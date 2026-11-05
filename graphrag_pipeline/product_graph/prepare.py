"""只读已有products.parquet，准备独立GraphRAG文档和商品映射，不重新划分数据。"""
import hashlib,json
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from .common import file_sha,write_json


def document_text(row):
    """标题/特征/描述拼接；details仅选商品属性，不读取评论、评分或测试交互。"""
    parts=[]
    allowed={'brand','label','manufacturer','artist','artists','genre','genres','format','material','style',
             'color','colour','target audience','item form','composer','performer','conductor'}
    for field in ['title','features','description','details']:
        value=row.get(field)
        if field=='details' and isinstance(value,str):
            try:value=json.loads(value)
            except json.JSONDecodeError:value={}
        if isinstance(value,dict):value='\n'.join(str(k)+': '+str(v) for k,v in value.items() if str(k).strip().casefold() in allowed)
        elif isinstance(value,list):value='\n'.join(str(x) for x in value)
        if value:parts.append(field+': '+str(value))
    return '\n'.join(parts)


def prepare(source,run,limit=3,item_ids=None):
    """limit=None明确表示全目录，默认只准备3件；run必须是全新目录。"""
    source=Path(source).resolve();run=Path(run).resolve()
    if run.exists():raise FileExistsError('Use a new run directory: '+str(run))
    source_manifest=json.loads((source/'manifest.json').read_text())
    expected=source_manifest.get('artifact_sha256',{}).get('products.parquet')
    if expected and expected!=file_sha(source/'products.parquet'):raise ValueError('Source product checksum mismatch')
    rows=pq.read_table(source/'products.parquet').to_pylist();catalog_count=len(rows)
    if item_ids is not None:
        wanted=set(item_ids)
        if len(wanted)!=len(item_ids):raise ValueError('Duplicate item_ids')
        rows=[r for r in rows if r['item_id'] in wanted]
        if len(rows)!=len(wanted):raise ValueError('Requested product missing')
    else:
        rows.sort(key=lambda r:hashlib.sha256(r['parent_asin'].encode()).hexdigest())
        if limit is not None:
            if limit<1:raise ValueError('limit must be positive')
            rows=rows[:limit]
    if not rows:raise ValueError('No products selected')
    mapping=[];documents=[];seen=set()
    for row in rows:
        parent=str(row['parent_asin'])
        if parent in seen:raise ValueError('Duplicate parent_asin')
        seen.add(parent);text=document_text(row)
        if not text.strip():raise ValueError('Empty product text: '+parent)
        # SDK API显式文档ID，不靠推断标题定位商品；每个商品是一份独立文档。
        documents.append({'id':parent,'title':parent,'text':text,'creation_date':''})
        mapping.append({'document_id':parent,'parent_asin':parent,'item_id':int(row['item_id']),
                        'title':row.get('title',''),'text_sha256':hashlib.sha256(text.encode()).hexdigest()})
    run.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(documents),run/'documents.parquet')
    write_json(run/'product_mapping.json',mapping)
    manifest={'source_directory':str(source),'data_fingerprint':source_manifest['fingerprint'],
              'catalog_size':catalog_count,'selected_products':len(rows),'scope':'full_catalog' if len(rows)==catalog_count else 'sample',
              'input_sha256':{name:file_sha(run/name) for name in ['documents.parquet','product_mapping.json']},
              'source_products_sha256':file_sha(source/'products.parquet')}
    write_json(run/'manifest.json',manifest);return manifest
