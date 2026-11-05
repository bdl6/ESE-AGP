"""独立实现的离线验收：映射来源、IDF、边方向、篡改检测；不请求模型。"""
import json
import math
import pytest
import pyarrow as pa
import pyarrow.parquet as pq
from product_graph.common import write_json,load_manifest,digest
from product_graph.prepare import prepare
from product_graph.convert import convert
from product_graph.semantic import build
from product_graph.official import configuration


@pytest.fixture
def ready(tmp_path):
    """三商品两类实体，商品1和2共享品牌；商品3不同，用于非零IDF。"""
    source=tmp_path/'source';source.mkdir()
    write_json(source/'manifest.json',{'fingerprint':'dataset'})
    pq.write_table(pa.Table.from_pylist([{'item_id':i,'parent_asin':'P'+str(i),'title':'Album '+str(i),
        'description':['First','Second'],'details':json.dumps({'Label':'BrandA' if i<3 else 'BrandB','Customer Reviews':'DO_NOT_USE'})} for i in range(1,4)]),source/'products.parquet')
    run=tmp_path/'run';prepare(source,run,limit=None)
    out=run/'sdk_output';out.mkdir();write_json(run/'index_status.json',{'status':'complete'})
    pq.write_table(pa.Table.from_pylist([{'id':'t'+str(i),'document_id':'P'+str(i),'text':'Label: BrandA' if i<3 else 'Label: BrandB'} for i in range(1,4)]),out/'text_units.parquet')
    pq.write_table(pa.Table.from_pylist([{'title':'BrandA','type':'brand','description':'Label','text_unit_ids':['t1','t2']},
                                      {'title':'BrandB','type':'brand','description':'Label','text_unit_ids':['t3']}]),out/'entities.parquet')
    pq.write_table(pa.Table.from_pylist([{'source':'Album 1','target':'BrandA','weight':1.,'text_unit_ids':['t1']}]),out/'relationships.parquet')
    return run


def test_provenance_mapping_and_shared_entity_graph(ready):
    result=convert(ready);assert result['products']==3 and result['entities']==2
    a=json.loads((ready/'product_entities.json').read_text());rows={r['item_id']:r for r in a['records']}
    assert rows[1]['entities'][0]['canonical_id']==rows[2]['entities'][0]['canonical_id']
    assert rows[1]['entities'][0]['canonical_id']!=rows[3]['entities'][0]['canonical_id']
    g=build(ready);assert g['num_edges']==2 and g['idf_document_count']==3
    full=json.loads((ready/'semantic_graph.json').read_text())
    assert full['edge_index']==[[1,2],[2,1]] and full['edge_weight']==[1.,1.]
    assert (ready/'semantic_graph_preview.json').exists()


def test_changed_input_rejected(ready):
    (ready/'product_mapping.json').write_text('[]')
    with pytest.raises(ValueError,match='changed'):load_manifest(ready)


def test_unknown_chunk_rejected(ready):
    out=ready/'sdk_output';pq.write_table(pa.Table.from_pylist([{'title':'Wrong','type':'brand','text_unit_ids':['missing']}]),out/'entities.parquet')
    with pytest.raises(ValueError,match='Unknown'):convert(ready)


def test_failed_sdk_not_exported_as_success(ready):
    write_json(ready/'index_status.json',{'status':'partial'})
    with pytest.raises(ValueError,match='not complete'):convert(ready)


def test_sample_idf_uses_selected_products(tmp_path):
    run=tmp_path
    rows=[{'item_id':i,'entities':[{'canonical_id':'e'}] if i<3 else []} for i in [1,2,3]]
    a={'catalog_size':10000,'scope':'sample','data_fingerprint':'data','records':rows};a['fingerprint']=digest(a)
    write_json(run/'product_entities.json',a)
    g=build(run);assert g['idf_document_count']==3 and g['num_items']==10000 and g['scope']=='sample'


def test_sdk_workflows_and_storage_are_isolated(tmp_path):
    c=configuration(tmp_path,2,1)
    assert c['extract_graph']['max_gleanings']==1
    assert c['concurrent_requests']==2
    assert 'generate_text_embeddings' not in c['workflows']
    assert c['chunking']['prepend_metadata']==['id']
    for name in ['input_storage','output_storage','update_output_storage']:
        assert str(tmp_path) in c[name]['base_dir']


def test_evidence_refs_are_local_to_each_product(ready):
    convert(ready)
    records=json.loads((ready/'product_entities.json').read_text())['records']
    rows={r['item_id']:r for r in records}
    assert rows[1]['entities'][0]['source_text_unit_ids']==['t1']
    assert rows[2]['entities'][0]['source_text_unit_ids']==['t2']


def test_same_name_different_types_not_collapsed(ready):
    out=ready/'sdk_output'
    pq.write_table(pa.Table.from_pylist([{'title':'Same','type':'brand','description':'brand','text_unit_ids':['t1']},
        {'title':'Same','type':'artist','description':'artist','text_unit_ids':['t2']}]),out/'entities.parquet')
    convert(ready);entities=json.loads((ready/'entity_registry.json').read_text())
    assert len({e['canonical_id'] for e in entities})==2
    assert {e['type'] for e in entities}=={'brand','artist'}


def test_custom_prompts_have_sdk_placeholders_only():
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]/'prompts'
    extract=(root/'extract_graph.txt').read_text().format(entity_types='brand,product',input_text='title: Album')
    summary=(root/'summarize_descriptions.txt').read_text().format(entity_name='Label',description_list='[]',max_length=60)
    assert '<|COMPLETE|>' in extract and 'title: Album' in extract
    assert 'Label' in summary
