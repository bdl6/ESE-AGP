"""联合抽取、证据、跨轮聚合、描述和消歧的可核查测试。"""
from pathlib import Path
import json
import pytest
import pyarrow as pa
import pyarrow.parquet as pq
from ese_agp.data.sessions import PreparedData,SessionExample
from ese_agp.entities.joint import validate_joint,validate_description,grounded
from ese_agp.entities.aggregation import aggregate,canonicalize
from ese_agp.entities.disambiguation import candidates,validate_resolution
from ese_agp.entities.workflow import run_workflow,StageRunner,document_chunks,RequestBudgetExceeded
from ese_agp.graphs.behavior import build_behavior_graph,save_graph


def joint(name='R&B',item=1,brand='LabelX'):
    """构造实体和关系都有实际证据的片段。"""
    text=f'{name} released by {brand}'
    payload={'entities':[{'name':name,'type':'genre','evidence':name},{'name':brand,'type':'brand','evidence':brand}],
             'relations':[{'entity_1':name,'entity_2':brand,'relation':'released by','strength':.8,'evidence':text}]}
    return text,payload,validate_joint(payload,text,item,0,0)


def test_evidence_case_whitespace_restores_source_offsets():
    """规范匹配后存储原文，不保存模型改写的证据。"""
    e=grounded('Mac  McAnally','mac mcanally',1,2,10)
    assert e['quote']=='Mac  McAnally';assert (e['start'],e['end'])==(10,23)
    with pytest.raises(ValueError):grounded('Album','Unknown',1,0,0)


def test_relations_strength_and_endpoint_validation():
    """拒绝虚构端点、bool强度和越界强度，保留独立有效实体。"""
    text,payload,result=joint();assert len(result['relations'])==1
    payload['relations'][0]['strength']=1.1
    assert not validate_joint(payload,text,1,0,0)['relations']
    payload['relations'][0]['strength']=True
    assert not validate_joint(payload,text,1,0,0)['relations']
    payload['relations'][0].update(strength=.5,entity_1='Invented')
    assert not validate_joint(payload,text,1,0,0)['relations']


def test_round_block_aggregation_keeps_provenance():
    """跨轮重复实体去重，跨商品证据保留，关系强度观测可追溯。"""
    a=joint(item=1)[2];b=joint(item=2)[2];entities,relations=aggregate([a,a,b])
    assert len(entities)==4;assert all(len(e['evidence'])==1 for e in entities.values())
    relation=next(iter(relations.values()));assert len(relation['evidence'])==1
    assert relation['strength_observations']==[.8,.8]


def test_disambiguation_requires_both_sides_and_type():
    """相似度只生成候选，合并必须通过双方证据和置信度检查。"""
    a=joint(name='R&B',item=1)[2];b=joint(name='Rhythm and Blues',item=2)[2]
    entities,relations=aggregate([a,b]);left,right=[n for n in entities.values() if n['type']=='genre']
    refs=[left['evidence'][0]['evidence_id'],right['evidence'][0]['evidence_id']]
    payload={'same_entity':True,'canonical_name':right['name'],'confidence':.95,'reason':'同义流派表达','evidence_ids':refs}
    decision=validate_resolution(payload,left,right,.9);assert decision['accepted']
    registry,rewritten,mapping=canonicalize(entities,relations,[decision])
    assert mapping[left['entity_id']]==mapping[right['entity_id']]
    assert len(registry)==3;assert len(rewritten)==2
    payload['evidence_ids']=refs[:1]
    with pytest.raises(ValueError):validate_resolution(payload,left,right,.9)
    payload['evidence_ids']=refs;payload['confidence']=.6
    assert not validate_resolution(payload,left,right,.9)['accepted']
    pairs,stats=candidates(entities,0);assert not pairs;assert stats['unprocessed_pairs']>0


def test_descriptions_cannot_cite_invented_sources():
    """描述引用必须来自实际展示给模型的片段。"""
    evidence=[joint()[2]['entities'][0]['evidence'][0]]
    with pytest.raises(ValueError):validate_description({'description':'text','evidence_ids':['fake']},evidence)
    assert validate_description({'description':'text','evidence_ids':[evidence[0]['evidence_id']]},evidence)


def test_document_chunks_do_not_truncate():
    """6000字符之后仍处理，末尾重叠不会产生无意义额外块。"""
    text='a'*7100;chunks=list(document_chunks(text,1600,200))
    assert chunks[-1][0]+len(chunks[-1][1])==len(text)
    assert chunks[0][1][-200:]==chunks[1][1][:200]


class WorkflowClient:
    """离线假服务覆盖所有真实阶段；不用于真实实验指标。"""
    def __init__(self):
        """累计模拟调用次数。"""
        self.calls=0
    def complete(self,prompt,text):
        """按阶段返回结构化且有引用的响应。"""
        self.calls+=1;p=json.loads(text)
        if 'TASK=joint' in prompt:
            if p['round']>0:result={'entities':[],'relations':[]}
            else:
                name='R&B' if 'R&B' in p['text'] else 'Rhythm and Blues'
                result=joint(name=name)[1]
        elif 'TASK=review_description' in prompt:
            result={'supported':True,'description':p['description'],'evidence_ids':[e['evidence_id'] for e in p['evidence']],'reason':'由展示证据支持'}
        elif 'TASK=describe' in prompt:
            result={'description':'根据商品原文得到的实体描述','evidence_ids':[p['evidence'][0]['evidence_id']] if 'evidence' in p else p['available_evidence_ids']}
        else:
            result={'same_entity':True,'canonical_name':p['right']['name'],'confidence':.99,'reason':'相同音乐流派的简称与全称',
                'evidence_ids':[p['left_evidence'][0]['evidence_id'],p['right_evidence'][0]['evidence_id']]}
        return {'choices':[{'finish_reason':'stop','message':{'content':json.dumps(result)}}],
                'usage':{'prompt_tokens':10,'completion_tokens':20}}


def test_full_workflow_cache_resume_and_canonical_graph(tmp_path):
    """两商品抽取→描述→真正别名合并→关系重写→规范ID语义图→断点缓存。"""
    d=PreparedData([SessionExample((1,),2,'u',0,1)],[],[],{'a':1,'b':2,'unused':3},[1,2,3],{})
    d.directory=tmp_path/'data';d.directory.mkdir()
    pq.write_table(pa.Table.from_pylist([{'item_id':1,'title':'R&B released by LabelX'},
                                       {'item_id':2,'title':'Rhythm and Blues released by LabelX'}]),d.directory/'products.parquet')
    behavior=tmp_path/'behavior.json';save_graph(build_behavior_graph(d),behavior)
    cfg={'entities':{'output':str(tmp_path/'workflow'),'item_ids':[1,2],'model':'fake','base_url':'https://offline.invalid',
         'max_requests':50,'max_output_tokens':1024,'max_rounds':3,'max_alias_pairs':5},
         'graph':{'behavior_path':str(behavior),'semantic_path':str(tmp_path/'semantic.json'),'path':str(tmp_path/'fused.json')}}
    client=WorkflowClient();result=run_workflow(d,cfg,client)
    assert result['complete'];assert result['accepted_alias_merges']==2;assert result['semantic_edges']==2
    registry=json.loads((tmp_path/'workflow/entity_registry.json').read_text())
    assert any(set(e['aliases'])=={'R&B','Rhythm and Blues'} for e in registry['entities'])
    calls=client.calls;run_workflow(d,cfg,client);assert client.calls==calls
    chunks=json.loads((tmp_path/'workflow/joint_chunks.json').read_text())
    assert all(t['round']<2 for t in chunks['traces'])
    status=json.loads((tmp_path/'workflow/status.json').read_text());assert status['complete']


def test_request_budget_enforced_without_extra_call(tmp_path):
    """额度耗尽时不请求模型，已缓存阶段不重复计费。"""
    client=WorkflowClient();runner=StageRunner(tmp_path,{'max_requests':1},client,'identity')
    payload={'name':'X','evidence':[{'evidence_id':'e1'}]}
    from ese_agp.entities.joint import DESCRIPTION_PROMPT
    validator=lambda x:validate_description(x,payload['evidence'])
    runner.ask('description',DESCRIPTION_PROMPT,payload,validator)
    runner.ask('description',DESCRIPTION_PROMPT,payload,validator);assert client.calls==1
    with pytest.raises(RequestBudgetExceeded):runner.ask('other',DESCRIPTION_PROMPT,payload,validator)
    assert client.calls==1


def test_hierarchical_description_reads_all_evidence(tmp_path):
    """超过8个证据时每组均进入模型，最终引用合法原始证据。"""
    from ese_agp.entities.workflow import describe
    client=WorkflowClient();runner=StageRunner(tmp_path,{'max_requests':10},client,'hierarchical')
    evidence=[{'evidence_id':'e'+str(i),'item_id':i,'start':0,'end':1,'quote':'X'} for i in range(17)]
    result=describe(runner,{'name':'X','type':'brand','evidence':evidence},'description')
    assert client.calls==8;assert set(result['evidence_ids'])=={'e'+str(i) for i in range(17)}


def test_same_name_different_products_not_automatically_merged():
    """同名实体跨商品先保持独立，避免把字符串相等当成消歧。"""
    a=joint(item=1)[2];b=joint(item=2)[2];entities,relations=aggregate([a,b])
    registry,_,mapping=canonicalize(entities,relations,[])
    assert len(registry)==4;assert len(set(mapping.values()))==4


def test_label_type_maps_to_brand_and_numeric_category_rejected():
    """厂牌同义类型明确映射，数字日期不作推荐类别。"""
    payload={'entities':[{'type':'label','name':'Sony Legacy','evidence':'Label: Sony Legacy'},
                         {'type':'category','name':'2006','evidence':'2006'}],'relations':[]}
    result=validate_joint(payload,'Label: Sony Legacy, 2006',1,0,0)
    assert len(result['entities'])==1;assert result['entities'][0]['type']=='brand'
    assert len(result['rejected'])==1


def test_document_excludes_administrative_metadata():
    """日期、尺寸和数量不能作为材料/格式等实体的噪声输入。"""
    from ese_agp.entities.workflow import product_document
    text=product_document({'title':'Album','details':json.dumps({'Label':'Sony Legacy','Date First Available':'2006','Number of discs':'1','Product Dimensions':'5 inches'})})
    assert 'Sony Legacy' in text
    for noise in ['2006','Number of discs','5 inches']:assert noise not in text
