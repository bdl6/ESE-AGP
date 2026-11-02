"""验证错误隔离边界、截断拆块和不完整产物标记；全部使用离线模拟响应。"""
import json
import urllib.error
import pytest
from test_workflow import WorkflowClient, joint
from ese_agp.entities import workflow as w
from ese_agp.data.sessions import PreparedData, SessionExample
from ese_agp.graphs.behavior import build_behavior_graph, save_graph


@pytest.fixture
def setup_workflow(tmp_path, monkeypatch):
    """准备两个商品，不访问真实模型服务。"""
    data=PreparedData([SessionExample((1,),2,'u',0,1)],[],[],{'a':1,'b':2},[1,2],{})
    data.directory=tmp_path
    monkeypatch.setattr(w,'select_documents',lambda data,cfg: ([(1,'R&B released by LabelX'),(2,'Rhythm and Blues released by LabelX')],{1,2}))
    monkeypatch.setattr(w.time,'sleep',lambda seconds:None)
    behavior=tmp_path/'behavior.json';save_graph(build_behavior_graph(data),behavior)
    cfg={'entities':{'output':str(tmp_path/'workflow'),'item_ids':[1,2],'model':'fake','base_url':'https://offline.invalid',
         'max_requests':100,'max_output_tokens':4096,'max_rounds':3,'max_alias_pairs':5,'request_attempts':2},
         'graph':{'behavior_path':str(behavior),'semantic_path':str(tmp_path/'semantic.json'),'path':str(tmp_path/'fused.json')}}
    return data,cfg


@pytest.mark.parametrize('failure_stage',['joint','describe','resolve','describe_canonical'])
def test_isolated_failure_still_writes_partial_graph(setup_workflow,failure_stage):
    """各LLM阶段失败均写明部分完成，单个业务失败不会丢掉其他商品。"""
    data,cfg=setup_workflow
    class Broken(WorkflowClient):
        def complete(self,prompt,text):
            p=json.loads(text)
            fail=(failure_stage=='joint' and 'TASK=joint' in prompt and p['item_id']==1)
            fail|=(failure_stage=='describe' and 'TASK=describe' in prompt and p.get('name')=='R&B')
            fail|=(failure_stage=='resolve' and 'TASK=resolve' in prompt)
            # 合并后的同名厂牌含双方证据，初始描述只有单个商品的证据。
            fail|=(failure_stage=='describe_canonical' and 'TASK=describe' in prompt and len(p.get('evidence',[]))>1)
            if fail:raise TimeoutError('simulated timeout')
            return super().complete(prompt,text)
    result=w.run_workflow(data,cfg,Broken())
    assert not result['complete'];assert result['stage_errors']>0
    output=__import__('pathlib').Path(cfg['entities']['output'])
    artifact=json.loads((output/'entities.json').read_text())
    assert artifact['scope']=='pilot_or_partial'
    assert __import__('pathlib').Path(cfg['graph']['semantic_path']).exists()
    assert (output/'failures.json').exists()
    if failure_stage in {'joint','describe'}:
        assert result['completed_products']==1;assert result['failed_products']==1
        assert next(r for r in artifact['records'] if r['item_id']==1)['status']=='failed'
        assert next(r for r in artifact['records'] if r['item_id']==2)['status']=='success'
        assert result['semantic_edges']==0


def test_budget_exhaustion_is_global_stop(setup_workflow):
    """额度耗尽不跳过整个列表伪装成处理完成。"""
    data,cfg=setup_workflow;cfg['entities']['max_requests']=1
    with pytest.raises(w.RequestBudgetExceeded):w.run_workflow(data,cfg,WorkflowClient())


def test_program_error_not_silently_swallowed(setup_workflow,monkeypatch):
    """程序缺陷必须暴露，不能把所有异常都当商品错误。"""
    data,cfg=setup_workflow
    def broken(*args,**kwargs):raise AssertionError('bug')
    monkeypatch.setattr(w,'extract_product',broken)
    with pytest.raises(AssertionError):w.run_workflow(data,cfg,WorkflowClient())


def test_truncation_splits_and_preserves_source_offsets(tmp_path):
    """截断不重复相同请求；两个子块完整覆盖原文，证据偏移仍精确。"""
    text='LabelX '+' '*45+'LabelY';seen=[]
    class Dense:
        def complete(self,prompt,body):
            p=json.loads(body);seen.append(p['text'])
            if len(p['text'])>50:return {'choices':[{'finish_reason':'length','message':{'content':'{'}}]}
            name='LabelX' if 'LabelX' in p['text'] else 'LabelY'
            value={'entities':[{'name':name,'type':'brand','evidence':name}],'relations':[]}
            return {'choices':[{'finish_reason':'stop','message':{'content':json.dumps(value)}}]}
    cfg={'max_requests':10,'max_rounds':1,'min_chunk_chars':10,'max_split_depth':2}
    runner=w.StageRunner(tmp_path,cfg,Dense(),'split')
    results,traces=w.extract_product(runner,1,text,cfg,lambda *args:None)
    assert len(seen)==3;assert seen.count(text)==1
    assert len(results)==2
    for result in results:
        for entity in result['entities']:
            proof=entity['evidence'][0]
            assert text[proof['start']:proof['end']]==proof['quote']
    assert all(t['split_depth']==1 for t in traces)


def test_transient_retry_and_auth_global_stop(tmp_path,monkeypatch):
    """连接失败有限重试，鉴权错误直接暴露供修复。"""
    monkeypatch.setattr(w.time,'sleep',lambda seconds:None)
    class Flaky(WorkflowClient):
        def complete(self,prompt,text):
            if not hasattr(self,'failed'):self.failed=True;raise ConnectionError('temporary')
            return super().complete(prompt,text)
    from ese_agp.entities.joint import DESCRIPTION_PROMPT,validate_description
    payload={'name':'X','evidence':[{'evidence_id':'e1'}]}
    runner=w.StageRunner(tmp_path,{'max_requests':5},Flaky(),'retry')
    runner.ask('describe',DESCRIPTION_PROMPT,payload,lambda x:validate_description(x,payload['evidence']))
    assert runner.state['attempts']==2
    class Auth:
        def complete(self,*args):raise urllib.error.HTTPError('https://offline.invalid',401,'denied',{},None)
    runner.client=Auth()
    with pytest.raises(RuntimeError,match='configuration/authentication'):runner.ask('other',DESCRIPTION_PROMPT,payload,lambda x:x)
