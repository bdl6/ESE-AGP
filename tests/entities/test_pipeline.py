"""实体抽取和语义融合端到端测试，使用离线假服务，不生成真实抽取成果。"""
import json
import urllib.error
from pathlib import Path
import pytest
from ese_agp.data.sessions import PreparedData, SessionExample
from ese_agp.data.io import data_fingerprint
from ese_agp.entities.client import ChatClient
from ese_agp.entities.pipeline import extract_requests
from ese_agp.graphs.behavior import build_behavior_graph, graph_fingerprint, save_graph, load_graph
from ese_agp.graphs.semantic import build_semantic_graph
from ese_agp.graphs.fusion import fuse_graphs


class FakeClient:
    """明确的模拟服务；实体直接来自构造的测试文本。"""
    def __init__(self):
        """记录HTTP等价调用次数。"""
        self.calls=0
    def complete(self,prompt,text):
        """每块两轮都返回同一实体，用于验证去重和缓存。"""
        self.calls+=1
        e={'type':'artist','name':text,'evidence':text}
        return {'choices':[{'finish_reason':'stop','message':{'content':json.dumps({'entities':[e]})}}],
                'usage':{'prompt_tokens':10,'completion_tokens':5}}


@pytest.fixture
def fixture(tmp_path):
    """3件训练商品、1件孤立商品，1和2共享实体。"""
    d=PreparedData([SessionExample((1,),2,'u',0,1),SessionExample((2,),3,'v',0,1)],[],[],{str(i):i for i in range(1,5)},[1,2,3,4],{})
    requests=tmp_path/'pilot';requests.mkdir()
    import hashlib
    rows=[{'item_id':i,'text':text,'text_sha256':hashlib.sha256(text.encode()).hexdigest()} for i,text in [(1,'Singer'),(2,'Singer'),(3,'Other')]]
    (requests/'requests.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    (requests/'manifest.json').write_text(json.dumps({'data_fingerprint':data_fingerprint(d)}))
    cfg={'requests':str(requests/'requests.jsonl'),'output':str(tmp_path/'result'),'max_requests':6,'model':'fake','base_url':'https://offline.invalid/v1'}
    return d,cfg


def test_extract_cache_semantic_fusion_end_to_end(fixture,tmp_path):
    """两轮抽取、断点缓存、IDF连接、融合、正式加载完整串通。"""
    d,cfg=fixture;client=FakeClient()
    first=extract_requests(d,cfg,client);assert first['scope']=='full';assert first['completed_products']==3;assert client.calls==6
    extract_requests(d,cfg,client);assert client.calls==6
    path=Path(cfg['output'])/'entities.json'
    semantic=build_semantic_graph(d,path)
    assert semantic['edge_index']==[[1,2],[2,1]];assert semantic['edge_weight']==[1.,1.]
    behavior=build_behavior_graph(d);fused=fuse_graphs(behavior,semantic,.5)
    weights=dict(zip(zip(*fused['edge_index']),fused['edge_weight']))
    assert weights=={(1,2):1.,(2,1):.5,(2,3):.5}
    gp=tmp_path/'graph.json';save_graph(fused,gp);assert load_graph(gp,d)==fused


def test_quota_failure_is_not_success_empty_entities(fixture):
    """遗漏检查未完成不能标成功，重启不能绕过请求次数上限。"""
    d,cfg=fixture;cfg['max_requests']=1;client=FakeClient()
    result=extract_requests(d,cfg,client)
    assert result['budget_exhausted'];assert result['completed_products']==0;assert result['failed_products']==1
    extract_requests(d,cfg,client);assert client.calls==1
    with pytest.raises(ValueError,match='Partial'):build_semantic_graph(d,Path(cfg['output'])/'entities.json')


def test_explicit_partial_graph_rejected_by_training_loader(fixture,tmp_path):
    """允许诊断图不等于允许它进入完整实验训练。"""
    d,cfg=fixture;cfg['max_requests']=2;extract_requests(d,cfg,FakeClient())
    semantic=build_semantic_graph(d,Path(cfg['output'])/'entities.json',allow_partial=True)
    path=tmp_path/'partial.json';save_graph(semantic,path)
    with pytest.raises(ValueError,match='Partial'):load_graph(path,d)
    assert load_graph(path,d,allow_partial=True)['scope']=='pilot_or_partial'
    with pytest.raises(ValueError,match='Partial'):fuse_graphs(build_behavior_graph(d),semantic)


def test_schema_retry_and_bad_auth_stop(fixture,monkeypatch):
    """格式错误最多3次，认证错误立即停止，不遍历全量商品浪费配额。"""
    monkeypatch.setattr('ese_agp.entities.pipeline.time.sleep',lambda seconds:None)
    d,cfg=fixture
    class Invalid(FakeClient):
        """格式错误响应。"""
        def complete(self,prompt,text):
            """缺少结果不能作为空实体成功。"""
            self.calls+=1;return {}
    bad=Invalid();cfg['max_requests']=3;result=extract_requests(d,cfg,bad)
    assert bad.calls==3;assert result['completed_products']==0
    cfg['output']=cfg['output']+'_auth';cfg['max_requests']=6
    class Unauthorized(FakeClient):
        """认证错误。"""
        def complete(self,prompt,text):
            """只暴露HTTP状态，异常体不写入结果。"""
            self.calls+=1;raise urllib.error.HTTPError('https://offline.invalid',401,'secret-error',{},None)
    auth=Unauthorized();result=extract_requests(d,cfg,auth)
    assert auth.calls==1;assert result['fatal_service_error']


def test_identity_lock_and_tamper_checks(fixture):
    """配置变化、并发运行和结果篡改均被识别。"""
    d,cfg=fixture;extract_requests(d,cfg,FakeClient())
    output=Path(cfg['output']);(output/'extract.lock').write_text('123')
    with pytest.raises(FileExistsError):extract_requests(d,cfg,FakeClient())
    (output/'extract.lock').unlink();changed=dict(cfg,model='changed')
    with pytest.raises(ValueError,match='identity'):extract_requests(d,changed,FakeClient())
    path=output/'entities.json';body=json.loads(path.read_text());body['scope']='bad';path.write_text(json.dumps(body))
    with pytest.raises(ValueError,match='fingerprint'):build_semantic_graph(d,path)


def test_pair_budget_and_topk_tie(fixture):
    """超预算明确报错；top-k并列按商品编号处理。"""
    d,cfg=fixture;extract_requests(d,cfg,FakeClient());path=Path(cfg['output'])/'entities.json'
    with pytest.raises(ValueError,match='budget'):build_semantic_graph(d,path,max_candidate_pairs=0)
    assert build_semantic_graph(d,path,top_k=1)['num_edges']==2


def test_client_requires_secure_url_and_env(monkeypatch):
    """初始化不请求，拒绝不安全地址和缺失密钥。"""
    cfg={'base_url':'http://example.com','api_key_env':'TEST_ENTITY_KEY'}
    with pytest.raises(ValueError,match='HTTPS'):ChatClient(cfg)
    cfg['base_url']='https://offline.invalid/v1';monkeypatch.delenv('TEST_ENTITY_KEY',raising=False)
    with pytest.raises(ValueError,match='Missing'):ChatClient(cfg)


def test_fenced_json_strict_evidence():
    """合法围栏可去除，移除围栏后仍严格核对原文证据。"""
    from ese_agp.entities.pipeline import decode_response
    raw={'choices':[{'finish_reason':'stop','message':{'content':'```json\n{"entities":[]}\n```'}}]}
    assert decode_response(raw,'text')==[]
    raw['choices'][0]['message']['content']='```json\n{"entities":[{"type":"format","name":"CD","evidence":"CD"}]}\n```'
    with pytest.raises(ValueError):decode_response(raw,'no format here')


def test_outage_stops_batch_and_keeps_success_cache(fixture,monkeypatch):
    """重试3次后暂停整个批次，后续商品成功缓存不会丢失。"""
    monkeypatch.setattr('ese_agp.entities.pipeline.time.sleep',lambda seconds:None)
    d,cfg=fixture;extract_requests(d,cfg,FakeClient())
    # 删除首件缓存模拟需重试，其余成功缓存必须重用。
    first=next((Path(cfg['output'])/'cache').glob('1_*.json'));first.unlink();cfg['max_requests']=12
    class Offline(FakeClient):
        """模拟连接拒绝。"""
        def complete(self,prompt,text):
            """只累计本次尝试，不生成虚假响应。"""
            self.calls+=1;raise urllib.error.URLError(ConnectionRefusedError(111,'refused'))
    client=Offline();result=extract_requests(d,cfg,client)
    assert client.calls==3;assert result['service_unavailable'];assert result['completed_products']==2


def test_validation_retry_supplies_error_feedback(fixture,monkeypatch):
    """错误反馈进入下一次请求，而非温度0重复同一无效输出。"""
    monkeypatch.setattr('ese_agp.entities.pipeline.time.sleep',lambda seconds:None)
    d,cfg=fixture;cfg['max_requests']=9
    class Repair(FakeClient):
        """第一条格式错误，后续按反馈返回有效结果。"""
        def complete(self,prompt,text):
            """记录错误修复提示。"""
            if self.calls==0:self.calls+=1;return {}
            if self.calls==1:assert '未通过校验' in prompt
            return super().complete(prompt,text)
    result=extract_requests(d,cfg,Repair());assert result['completed_products']==3


def test_idf_relative_weights_and_deterministic_topk(fixture):
    """手算两种不同频率实体的叠加，检查IDF和并列邻居选择。"""
    import math
    d,cfg=fixture;extract_requests(d,cfg,FakeClient());path=Path(cfg['output'])/'entities.json'
    body=json.loads(path.read_text())
    for record in body['records']:
        names=['common']+(['rare'] if record['item_id'] in {1,2} else [])
        record['entities']=[{'type':'artist','name':name,'normalized_name':name,'evidence':name} for name in names]
    body['fingerprint']=graph_fingerprint(body);path.write_text(json.dumps(body))
    graph=build_semantic_graph(d,path)
    weights=dict(zip(zip(*graph['edge_index']),graph['edge_weight']))
    common=math.log(5/4);rare=math.log(5/3)
    assert weights[1,2]==pytest.approx((common+rare)/(2*common+rare))
    assert weights[1,3]==pytest.approx(common/(2*common+rare))
    top=build_semantic_graph(d,path,top_k=1)
    assert top['edge_index']==[[1,2,3],[2,1,1]]


def test_invalid_entity_is_logged_without_discarding_valid_ones():
    """逐实体隔离幻觉，同时保留有原文证据的实体；不接受全无依据的结果。"""
    from ese_agp.entities.pipeline import decode_response
    entities=[{'type':'artist','name':'Ray Stevens','evidence':'Ray Stevens'},
              {'type':'format','name':'CD','evidence':'CD'}]
    raw={'choices':[{'finish_reason':'stop','message':{'content':json.dumps({'entities':entities})}}]}
    rejected=[];valid=decode_response(raw,'Ray Stevens album',rejected)
    assert len(valid)==1 and valid[0]['name']=='Ray Stevens';assert len(rejected)==1
    with pytest.raises(ValueError):decode_response(raw,'unrelated',[])


def test_existing_quote_does_not_justify_invented_name():
    """有原文引用仍不允许凭专辑名推断原文不存在的艺人。"""
    from ese_agp.entities.extraction import validate_entities
    with pytest.raises(ValueError,match='not supported'):
        validate_entities({'entities':[{'type':'artist','name':'Beyoncé','evidence':'Dangerously In Love'}]},'Dangerously In Love')
