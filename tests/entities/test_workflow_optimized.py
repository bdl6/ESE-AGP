"""优化版的线程安全、预算、候选与完整流程测试；不调用真实服务。"""
import json
import threading
import time
import pytest
from test_workflow import WorkflowClient
from ese_agp.entities import workflow_optimized as w
from ese_agp.entities.joint import DESCRIPTION_PROMPT,validate_description
from ese_agp.entities.optimized_candidates import candidates
from ese_agp.data.sessions import PreparedData,SessionExample
from ese_agp.graphs.behavior import build_behavior_graph,save_graph


def test_parallel_cache_and_exact_accounting(tmp_path):
    """不同键真并发，同一键只请求一次，累计token和预算不会丢失。"""
    class Slow(WorkflowClient):
        def __init__(self):super().__init__();self.guard=threading.Lock();self.active=0;self.peak=0
        def complete(self,prompt,text):
            with self.guard:self.active+=1;self.peak=max(self.peak,self.active)
            try:time.sleep(.01);return super().complete(prompt,text)
            finally:
                with self.guard:self.active-=1
    client=Slow();runner=w.StageRunner(tmp_path,{'max_requests':30},client,'parallel')
    def task(index):
        payload={'name':'X'+str(index),'evidence':[{'evidence_id':'e1'}]}
        return runner.ask('describe',DESCRIPTION_PROMPT,payload,lambda x:validate_description(x,payload['evidence']))
    list(w.parallel_results(task,list(range(20))+[0,0,1,1],4))
    assert client.peak>1 and client.peak<=4
    assert client.calls==20 and runner.state['attempts']==20
    assert runner.state['prompt_tokens']==200 and runner.state['completion_tokens']==400
    assert json.loads((tmp_path/'request_state.json').read_text())==runner.state
    for path in (tmp_path/'stages').glob('*.json'):assert json.loads(path.read_text())['status']=='success'


def test_concurrent_budget_never_overshoots(tmp_path):
    """并发预留额度时，真实请求数不超过硬上限。"""
    client=WorkflowClient();runner=w.StageRunner(tmp_path,{'max_requests':5},client,'budget')
    def task(index):return runner.ask('x',DESCRIPTION_PROMPT,{'name':str(index),'evidence':[{'evidence_id':'e'}]},lambda x:x)
    with pytest.raises(w.RequestBudgetExceeded):list(w.parallel_results(task,range(30),4))
    assert runner.state['attempts']==client.calls==5


def test_candidates_type_aliases_and_no_automatic_merge():
    """类型隔离；保留简称、包含与同名星形候选，而非直接合并实体。"""
    values=[('a','brand','Sony Legacy'),('b','brand','Legacy Recordings'),('c','format','CD'),('d','format','Audio CD'),
        ('e','genre','R&B'),('f','genre','Rhythm and Blues'),('g','artist','Sony Legacy')]
    entities={k:{'name':name,'normalized_name':name.lower(),'type':kind} for k,kind,name in values}
    pairs,stats=candidates(entities,100)
    assert ('a','b') in pairs and ('c','d') in pairs and ('e','f') in pairs
    assert all(entities[a]['type']==entities[b]['type'] for a,b in pairs)
    entities={str(i):{'name':'Label','normalized_name':'label','type':'brand'} for i in range(10)}
    pairs,stats=candidates(entities,100)
    assert len(pairs)==9 and stats['exact_name_star_pairs']==9


@pytest.mark.parametrize('fail_first',[False,True])
def test_full_workflow_deterministic_cache_and_failure_isolation(tmp_path,monkeypatch,fail_first):
    """完整图能生成，失败商品被隔离，顺序归并保证缓存恢复输出不变。"""
    data=PreparedData([SessionExample((1,),2,'u',0,1)],[],[],{'a':1,'b':2,'unused':3},[1,2,3],{});data.directory=tmp_path
    monkeypatch.setattr(w,'select_documents',lambda data,cfg: ([(1,'R&B released by LabelX'),(2,'Rhythm and Blues released by LabelX')],{1,2}))
    monkeypatch.setattr(w.time,'sleep',lambda seconds:None)
    behavior=tmp_path/'behavior.json';save_graph(build_behavior_graph(data),behavior)
    cfg={'entities':{'output':str(tmp_path/'entities'),'max_requests':100,'max_rounds':3,'max_alias_pairs':100,'concurrency':4,'request_attempts':1},
        'graph':{'behavior_path':str(behavior),'semantic_path':str(tmp_path/'semantic.json'),'path':str(tmp_path/'fused.json')}}
    class Client(WorkflowClient):
        def complete(self,prompt,text):
            if fail_first and 'TASK=joint' in prompt and json.loads(text)['item_id']==1:raise TimeoutError('isolated')
            return super().complete(prompt,text)
    client=Client();result=w.run_workflow(data,cfg,client)
    assert result['completed_products']==(1 if fail_first else 2)
    assert result['complete'] is (not fail_first)
    assert json.loads((tmp_path/'entities/extraction_timing.json').read_text())['products_extracted']==(1 if fail_first else 2)
    if not fail_first:
        assert result['semantic_edges']==2
        calls=client.calls;before=(tmp_path/'fused.json').read_bytes();w.run_workflow(data,cfg,client)
        assert client.calls==calls and (tmp_path/'fused.json').read_bytes()==before


@pytest.mark.parametrize('preserve_order',[False,True])
def test_slow_head_does_not_block_submitting_next_task(preserve_order):
    """首任务等待第三个任务启动；旧FIFO实现会超时，新实现可立即补位。"""
    third_started=threading.Event()
    def task(index):
        if index==0:
            assert third_started.wait(2),'slow head blocked submission'
        if index==2:third_started.set()
        return index
    result=list(w.parallel_results(task,range(4),2,preserve_order=preserve_order))
    assert sorted(result)==list(range(4))
    if preserve_order:assert result==list(range(4))


def test_scheduler_propagates_fatal_error():
    """程序异常不能被工作线程吞掉。"""
    def task(index):
        if index==1:raise RuntimeError('fatal task')
        return index
    with pytest.raises(RuntimeError,match='fatal task'):
        list(w.parallel_results(task,range(8),2,preserve_order=False))


@pytest.mark.parametrize('fail_first',[False,True])
def test_extraction_only_marks_remaining_stages_and_resumes(tmp_path,monkeypatch,fail_first):
    """联合抽取完成不能标成五步完成；同目录可继续完整流程而不重复抽取请求。"""
    data=PreparedData([SessionExample((1,),2,'u',0,1)],[],[],{'a':1,'b':2},[1,2],{});data.directory=tmp_path
    monkeypatch.setattr(w,'select_documents',lambda data,cfg: ([(1,'R&B released by LabelX'),(2,'Rhythm and Blues released by LabelX')],{1,2}))
    monkeypatch.setattr(w.time,'sleep',lambda seconds:None)
    cfg={'entities':{'output':str(tmp_path/'out'),'max_requests':100,'max_rounds':3,'max_alias_pairs':100,
                    'concurrency':4,'request_attempts':1,'extract_only':True,'build_graphs':False,'concise_outputs':True}}
    class Client(WorkflowClient):
        def __init__(self):super().__init__();self.joint_calls=0
        def complete(self,prompt,text):
            if 'TASK=joint' in prompt:
                self.joint_calls+=1
                if fail_first and json.loads(text)['item_id']==1:raise TimeoutError('isolated')
            return super().complete(prompt,text)
    client=Client();first=w.run_workflow(data,cfg,client)
    assert not first['complete'];assert 'disambiguation' in first['remaining_stages']
    assert (tmp_path/'out/products/2.json').exists()
    calls=client.joint_calls;cfg['entities']['extract_only']=False
    second=w.run_workflow(data,cfg,client)
    assert second['complete'] is (not fail_first)
    assert not second['graphs_built'];assert second['semantic_edges'] is None
    if not fail_first:assert client.joint_calls==calls



def test_readonly_cache_reuses_only_exact_success(tmp_path):
    """新全目录任务可复用旧成功请求，改变输入不能误命中。"""
    old=tmp_path/'old';old.mkdir();client=WorkflowClient()
    runner=w.StageRunner(old,{'max_requests':10},client,'old_identity')
    payload={'name':'X','evidence':[{'evidence_id':'e1'}]}
    check=lambda x:validate_description(x,payload['evidence'])
    runner.ask('describe',DESCRIPTION_PROMPT,payload,check);assert client.calls==1
    new=tmp_path/'new';new.mkdir()
    runner=w.StageRunner(new,{'max_requests':10,'cache_sources':[{'folder':str(old),'identity':'old_identity'}]},client,'new_identity')
    runner.ask('describe',DESCRIPTION_PROMPT,payload,check)
    assert client.calls==1 and runner.state['attempts']==0 and runner.state['reused_stage_hits']==1
    runner.ask('describe',DESCRIPTION_PROMPT,{**payload,'name':'Y'},check);assert client.calls==2


def test_full_catalog_includes_training_unseen_metadata(tmp_path):
    """目录范围读取未在训练出现商品的元数据，不读取测试行为构图。"""
    import pyarrow as pa
    import pyarrow.parquet as pq
    data=PreparedData([SessionExample((1,),2,'u',0,1)],[],[],{'a':1,'b':2,'c':3},[1,2,3],{});data.directory=tmp_path
    pq.write_table(pa.Table.from_pylist([{'item_id':i,'title':str(i)} for i in [1,2,3]]),tmp_path/'products.parquet')
    docs,known=w.select_documents(data,{'item_scope':'catalog','limit':3})
    assert known=={1,2,3};assert {i for i,t in docs}==known
