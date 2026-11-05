"""通过官方TextUnit来源恢复商品→实体映射，保留SDK原始表，不伪造逐字证据。"""
import csv,json
from pathlib import Path
from collections import defaultdict
import pyarrow.parquet as pq
from .common import load_manifest,read_json,write_json,digest


def table_rows(output,name):
    """GraphRAG3.3默认表后端可能是CSV；同时支持Parquet，显式解码列表字段。"""
    output=Path(output)
    if (output/(name+'.parquet')).exists():return pq.read_table(output/(name+'.parquet')).to_pylist()
    with (output/(name+'.csv')).open(newline='') as stream:return list(csv.DictReader(stream))


def references(value):
    """统一SDK列表表示，拒绝未知格式，避免把一个ID错误拆成字符。"""
    if value is None or value=='':return []
    if isinstance(value,str):
        parsed=json.loads(value)
        if not isinstance(parsed,list):raise ValueError('Expected JSON list of references')
        return [str(x) for x in parsed]
    if isinstance(value,(list,tuple)):return [str(x) for x in value]
    raise ValueError('Unknown reference list format')


def convert(run, allow_partial=False):
    """文档ID→商品、文本块→文档、实体→文本块三次确定性关联，不按文本相似度猜归属。"""
    run=Path(run);manifest=load_manifest(run);mapping=read_json(run/'product_mapping.json')
    status=read_json(run/'index_status.json')
    partial = status.get('status') == 'partial'
    if status.get('status') != 'complete' and not (allow_partial and partial and not status.get('workflow_errors')):
        raise ValueError('SDK index is not complete; inspect index_status.json')
    products={r['document_id']:r for r in mapping};tu={};by_item={r['item_id']:{} for r in mapping}
    units=table_rows(run/'sdk_output','text_units')
    for row in units:
        docs=[str(row['document_id'])] if row.get('document_id') else references(row.get('document_ids'))
        if len(docs)!=1 or docs[0] not in products:raise ValueError('Text unit crosses or references unknown product')
        key=str(row['id'])
        if key in tu and tu[key]['document_id']!=docs[0]:raise ValueError('Shared chunk ID across products; check prepend_metadata')
        tu[key]={'text_unit_id':key,'document_id':docs[0],'item_id':products[docs[0]]['item_id'],'text':row['text']}
    registry=[]
    for row in table_rows(run/'sdk_output','entities'):
        entity={'canonical_id':'entity_'+digest([str(row.get('type','UNKNOWN')),str(row['title'])])[:24],'name':str(row['title']),'type':str(row.get('type','UNKNOWN')),
                'description':row.get('description',''),'source_text_unit_ids':references(row.get('text_unit_ids'))}
        if not entity['source_text_unit_ids']:raise ValueError('Entity lacks source text unit IDs')
        registry.append(entity)
        for key in entity['source_text_unit_ids']:
            if key not in tu:raise ValueError('Unknown entity text unit reference')
            item=tu[key]['item_id']
            by_item[item][entity['canonical_id']]={**entity,'source_text_unit_ids':[t for t in entity['source_text_unit_ids'] if tu[t]['item_id']==item]}
    failed_units = sorted({e['text_unit_id'] for e in status.get('logged_errors', []) if e.get('text_unit_id')})
    failed_items = {tu[key]['item_id'] for key in failed_units}
    records=[]
    for product in mapping:
        values=list(by_item[product['item_id']].values())
        records.append({**product,'status':'partial' if product['item_id'] in failed_items else 'success','empty_entities':not values,'entities':values})
    artifact={'schema_version':1,'kind':'official_graphrag_product_entities','data_fingerprint':manifest['data_fingerprint'],
              'scope':'partial' if partial else manifest['scope'],'catalog_size':manifest['catalog_size'],'selected_products':len(records),
              'empty_products':sum(r['empty_entities'] for r in records),'records':records}
    if partial:
        artifact.update(index_status='partial', failed_text_unit_ids=failed_units,
                        affected_item_ids=sorted(failed_items))
    artifact['fingerprint']=digest(artifact);write_json(run/'product_entities.json',artifact)
    write_json(run/'entity_registry.json',registry)
    # 原始关系权重来自GraphRAG，不解释成经过校准的置信概率。
    names=defaultdict(list)
    for entity in registry:names[entity['name'].casefold()].append(entity['canonical_id'])
    relations=[]
    for row in table_rows(run/'sdk_output','relationships'):
        refs=references(row.get('text_unit_ids'))
        if any(key not in tu for key in refs):raise ValueError('Unknown relationship text unit reference')
        source=names[str(row['source']).casefold()];target=names[str(row['target']).casefold()]
        # 关系端点只有名字；同名异类型时不猜测具体ID，保留候选列表和歧义标记。
        relations.append({**row,'text_unit_ids':refs,'source_entity_ids':source,'target_entity_ids':target,
                          'ambiguous_endpoint':len(source)!=1 or len(target)!=1})
    write_json(run/'entity_relationships.json',relations)
    write_json(run/'text_unit_provenance.json',list(tu.values()))
    return {'products':len(records),'entities':len(registry),'empty_products':artifact['empty_products']}
