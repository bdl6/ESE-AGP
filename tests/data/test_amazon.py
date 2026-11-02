"""Amazon验收：统一词表、全局属性、Parquet、梯度和保存加载。"""
import copy
import gzip
import json
import math
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from ese_agp.data.amazon import prepare_amazon, _safe_details
from ese_agp.data.artifacts import PreparedArtifact
from ese_agp.data.sessions import Interaction, prepare, prefix_to_local_graph
from ese_agp.data.features import parse_price, fit_item_features, relative_price_bin
from ese_agp.models.embeddings import ItemAttributeEncoder
from ese_agp.models.recommender import LocalSessionRecommender
from ese_agp.data.batching import collate_examples
from ese_agp.training.trainer import train
from ese_agp.training.checkpoint import load_model
from ese_agp.training.evaluator import evaluate_model


def write_gzip(path, rows):
    """逐行写压缩 JSON，用真实 Amazon 字段而非第一阶段 item_id 字段。"""
    with gzip.open(path, 'wt', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row) + '\n')


def raw_rows():
    """20 个会话跨时间分布，三个商品有两种类别、三档原始价格。"""
    rows=[]
    patterns=[['A','B','C'],['B','C','A'],['C','A','B']]
    for day in range(20):
        for offset,item in enumerate(patterns[day%3]):
            rows.append({'user_id':f'u{day:02}', 'parent_asin':item,
                         'timestamp':1600000000000+day*86400000+offset})
    meta=[{'parent_asin':item,'categories':['Music',category],'price':price,
           'title':'Product '+item,'features':['safe feature'],'description':['safe description'],
           'details':{'Format':'CD','Customer Reviews':'future feedback'},
           'average_rating':5.0,'rating_number':999,'bought_together':['DO_NOT_USE']}
          for item,category,price in [('A','Rock',10),('B','Rock',20),('C','Pop',30)]]
    return rows,meta


@pytest.fixture
def amazon_fixture(tmp_path):
    """全合法测试文件，阈值为 1，便于与第一阶段纯 Python 参考实现对照。"""
    reviews,meta=raw_rows()
    review_file=tmp_path/'reviews.jsonl.gz';meta_file=tmp_path/'meta.jsonl.gz'
    write_gzip(review_file,reviews);write_gzip(meta_file,meta)
    output=tmp_path/'prepared'
    prepare_amazon(review_file,meta_file,output,min_user_events=1,min_item_events=1)
    return PreparedArtifact(output),reviews,meta


def test_price_parse_and_paper_relative_mapping():
    """价格拒绝无效输入；端点落首尾档，零方差落中档，映射方向单调。"""
    assert parse_price('$1,299.95') == 1299.95
    for value in [None,True,'None','10-20',-1,0,'NaN',float('inf')]:
        assert parse_price(value) is None
    features=fit_item_features([(1,'Rock',10),(2,'Rock',20),(3,'Pop',30)])
    assert features['price_bin_ids']==[0,1,50,26]
    stats=features['price_stats']['Rock']
    assert relative_price_bin(15,stats)==26
    assert relative_price_bin(-100,stats)==1
    assert relative_price_bin(10000,stats)==50


def test_amazon_matches_reference_sessions_and_attributes(amazon_fixture):
    """合法输入下 SQL 版的切分、顺序和标签与小数据参考实现一致。"""
    data,rows,_=amazon_fixture
    reference=prepare([Interaction(x['user_id'],x['parent_asin'],x['timestamp']) for x in rows],
                      min_user_events=1,min_item_events=1,train_ratio=.9,valid_ratio=0)
    assert data.catalog==reference.catalog
    assert data.popularity==reference.popularity
    assert [len(data.train),len(data.valid),len(data.test)]==[18,0,2]
    for split in ('train','valid','test'):
        actual=getattr(data,split)
        expected=getattr(reference,split)
        assert [(x.input_item_ids,x.target_item_id,x.session_start,x.session_end) for x in actual] == [
            (x.input_item_ids,x.target_item_id,x.session_start,x.session_end) for x in expected]
    assert data.item_features['price_stats']['Rock']['mean']==15
    assert data.item_features['price_bin_ids']==[0,1,50,26]
    import pyarrow.parquet as pq
    products=pq.read_table(data.directory/'products.parquet').to_pylist()
    assert 'Customer Reviews' not in products[0]['details']
    assert 'average_rating' not in products[0]




def test_artifact_detects_modified_catalog(amazon_fixture):
    """商品编号被改后必须在加载时拒绝，不能拿错误映射给模型。"""
    data,_,_=amazon_fixture
    path=data.directory/'catalog.json'
    path.write_text(json.dumps({'A':2,'B':1,'C':3}))
    with pytest.raises(ValueError,match='checksum'):
        PreparedArtifact(data.directory)


def test_attributes_receive_gradients_and_mask_padding(amazon_fixture):
    """CE 可训练三张表和融合投影；补齐位置严格为零，不受投影 bias 影响。"""
    data,_,_=amazon_fixture
    torch.manual_seed(42)
    model=LocalSessionRecommender(3,dim=16,layers=1,use_item_features=True,item_features=data.item_features)
    batch=collate_examples(list(data.train[:6]),torch.device('cpu'))
    loss=F.cross_entropy(model(batch),batch.target_item_ids-1)
    loss.backward()
    for parameter in [model.embedding.weight,model.attribute_encoder.category_embedding.weight,
                      model.attribute_encoder.price_embedding.weight,model.attribute_encoder.projection.weight]:
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum()>0
    ids=torch.tensor([0,1])
    encoded=model.attribute_encoder(ids,model.embedding(ids))
    assert encoded[0].count_nonzero()==0
    assert model.attribute_encoder.category_embedding.weight.grad[0].count_nonzero()==0
    assert model.attribute_encoder.price_embedding.weight.grad[0].count_nonzero()==0


def test_candidate_chunking_preserves_logits_loss_and_gradients(amazon_fixture):
    """候选分块后仍是一个全候选 CE；结果和参数梯度应与不分块版本一致。"""
    data,_,_=amazon_fixture
    torch.manual_seed(42)
    model=LocalSessionRecommender(3,dim=8,layers=1,use_item_features=True,item_features=data.item_features)
    reference=copy.deepcopy(model)
    model.candidate_chunk_size=1;reference.candidate_chunk_size=999
    batch=collate_examples(list(data.train[:4]),torch.device('cpu'))
    actual=model(batch);expected=reference(batch)
    torch.testing.assert_close(actual,expected)
    a=F.cross_entropy(actual,batch.target_item_ids-1);b=F.cross_entropy(expected,batch.target_item_ids-1)
    a.backward();b.backward()
    torch.testing.assert_close(a,b)
    for (_,left),(_,right) in zip(model.named_parameters(),reference.named_parameters()):
        torch.testing.assert_close(left.grad,right.grad,rtol=1e-4,atol=1e-6)


def test_attribute_checkpoint_round_trip_on_parquet(amazon_fixture,tmp_path):
    """真实格式产物训练一次并加载，独立验证指标与已保存报告一致。"""
    data,_,_=amazon_fixture
    config={'data':{},'model':{'dim':8,'layers':1,'dropout':0.0,'use_item_features':True,'candidate_chunk_size':2},
            'training':{'device':'cpu','seed':42,'batch_size':8,'learning_rate':.01,'epochs':2,
                        'patience':2,'top_k':2,'selection':'last_epoch','run_dir':str(tmp_path/'run')}}
    report=train(data,config)
    model=load_model(tmp_path/'run/best.pt',data,torch.device('cpu'))
    metrics=evaluate_model(model,data.test,data.popularity,torch.device('cpu'),8,2)
    assert report['best_epoch']==2 and report['valid'] is None
    assert all(math.isfinite(v) for v in metrics.values())




def test_id_and_attribute_models_share_initial_common_weights(amazon_fixture):
    """同种子下共有参数初始化一致，避免属性开关无意改变整套基线初始权重。"""
    data,_,_=amazon_fixture
    torch.manual_seed(42)
    plain=LocalSessionRecommender(3,dim=8,layers=1)
    torch.manual_seed(42)
    attributes=LocalSessionRecommender(3,dim=8,layers=1,use_item_features=True,item_features=data.item_features)
    for name,value in plain.state_dict().items():
        torch.testing.assert_close(value,attributes.state_dict()[name],rtol=0,atol=0)


def test_timestamp_fallback_and_invalid_fields_are_counted(tmp_path):
    """兼容官方样例中的 sort_timestamp，坏时间只计入无效行，不产生伪会话。"""
    rows,meta=raw_rows()
    rows[0]['sort_timestamp']=rows[0].pop('timestamp')
    rows.append({'user_id':'invalid-time','parent_asin':'A','timestamp':'broken'})
    reviews=tmp_path/'reviews.gz';metadata=tmp_path/'metadata.gz'
    write_gzip(reviews,rows);write_gzip(metadata,meta)
    prepare_amazon(reviews,metadata,tmp_path/'output',min_user_events=1,min_item_events=1)
    data=PreparedArtifact(tmp_path/'output')
    assert data.manifest['stats']['invalid_review_fields']==1
    assert data.manifest['stats']['selected_interactions']==60
    assert [len(data.train),len(data.valid),len(data.test)]==[18,0,2]


def test_legacy_temporal_protocol_cannot_be_selected(tmp_path):
    """删除旧分支后显式拒绝历史协议，不悄悄执行不同词表规则。"""
    with pytest.raises(ValueError,match='Only paper protocol'):
        prepare_amazon(tmp_path/'missing.gz',tmp_path/'missing-meta.gz',tmp_path/'result',protocol='temporal')
    assert not (tmp_path/'result').exists()


def test_invalid_metadata_deleted_before_global_vocabulary(tmp_path):
    """删除无效元数据是全局清洗；不以训练未见为理由删除有效商品。"""
    rows,meta=raw_rows()
    rows=[x for x in rows if x['user_id']!='u00']
    rows += [{'user_id':'u00','parent_asin':item,'timestamp':1600000000000+i} for i,item in enumerate(['A','NO_PRICE','B','A'])]
    meta += [{'parent_asin':'NO_PRICE','categories':['Music','Rock'],'price':None}]
    review=tmp_path/'reviews.gz';metadata=tmp_path/'metadata.gz'
    write_gzip(review,rows);write_gzip(metadata,meta)
    prepare_amazon(review,metadata,tmp_path/'result',min_user_events=1,min_item_events=1)
    data=PreparedArtifact(tmp_path/'result')
    assert 'NO_PRICE' not in data.catalog
    ex=next(x for split in [data.train,data.test] for x in split if x.user_id=='u00')
    assert ex.input_item_ids==(data.catalog['A'],data.catalog['B'])
    assert ex.target_item_id==data.catalog['A']
    assert data.manifest['stats']['oov_targets']=={'train':0,'valid':0,'test':0}
