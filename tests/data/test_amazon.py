"""Day02 验收：Amazon schema、训练内属性、OOV、Parquet、梯度和保存加载。"""
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
                      min_user_events=1,min_item_events=1)
    assert data.catalog==reference.catalog
    assert data.popularity==reference.popularity
    assert [len(data.train),len(data.valid),len(data.test)]==[16,2,2]
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


def test_missing_metadata_and_heldout_items_never_bridge_or_fit(tmp_path):
    """缺元数据/价格的商品保留 OOV 位置；验证商品的极端价格不能改变训练统计。"""
    rows,meta=raw_rows()
    # 第一会话 A→unknown→B→A，前缀应为 [A,0,B]，不能补出 A→B。
    rows=[x for x in rows if x['user_id']!='u00']
    rows += [{'user_id':'u00','parent_asin':item,'timestamp':1600000000000+i}
             for i,item in enumerate(['A','NO_PRICE','B','A'])]
    # 第 17 个会话属于验证期，把其目标换成仅验证出现的超贵商品。
    for row in rows:
        if row['user_id']=='u16' and row['timestamp']%86400000==(1600000000000+16*86400000+2)%86400000:
            row['parent_asin']='HELDOUT'
    meta += [{'parent_asin':'NO_PRICE','categories':['Music','Rock'],'price':None},
             {'parent_asin':'HELDOUT','categories':['Music','Rock'],'price':999999},
             {'parent_asin':'A','categories':['Other'],'price':500000}]
    review_file=tmp_path/'reviews.gz';meta_file=tmp_path/'meta.gz'
    write_gzip(review_file,rows);write_gzip(meta_file,meta)
    prepare_amazon(review_file,meta_file,tmp_path/'a',min_user_events=1,min_item_events=1)
    data=PreparedArtifact(tmp_path/'a')
    assert data.catalog=={'A':1,'B':2,'C':3}
    assert data.train[0].input_item_ids==(1,0,2)
    assert prefix_to_local_graph([data.train[0].input_item_ids]).edge_index==([],[])
    assert data.valid[0].target_item_id==0
    assert data.manifest['stats']['metadata_invalid_price']==1
    assert data.item_features['price_stats']['Rock']['mean']==15
    # 只改 heldout 商品/文本，训练映射、价格统计和样本不变。
    changed=[dict(row,parent_asin='OTHER_HELDOUT') if row['user_id'] in ['u16','u17','u18','u19'] else row for row in rows]
    write_gzip(review_file,changed)
    prepare_amazon(review_file,meta_file,tmp_path/'b',min_user_events=1,min_item_events=1)
    second=PreparedArtifact(tmp_path/'b')
    assert list(data.train)==list(second.train)
    assert data.catalog==second.catalog
    assert data.item_features==second.item_features
    assert data.popularity==second.popularity


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
                        'patience':2,'top_k':2,'run_dir':str(tmp_path/'run')}}
    report=train(data,config)
    model=load_model(tmp_path/'run/best.pt',data,torch.device('cpu'))
    metrics=evaluate_model(model,data.valid,data.popularity,torch.device('cpu'),8,2)
    assert metrics==report['valid']


def test_sql_split_uses_untruncated_boundaries(tmp_path):
    """跨界会话即使截断后起点落入验证期，仍必须按原始起点整段删除。"""
    base=1600000000000
    specifications=[('a',[0,10]),('b',[0,20]),('c',[0,30]),('cross',[25,35,40]),
                    ('valid',[45,50]),('test',[55,60])]
    rows=[{'user_id':user,'parent_asin':'A' if i%2==0 else 'B','timestamp':base+t}
          for user,times in specifications for i,t in enumerate(times)]
    meta=[{'parent_asin':item,'categories':['Music'],'price':price} for item,price in [('A',10),('B',20)]]
    review=tmp_path/'reviews.gz';metadata=tmp_path/'meta.gz'
    write_gzip(review,rows);write_gzip(metadata,meta)
    prepare_amazon(review,metadata,tmp_path/'result',min_user_events=1,min_item_events=1,
                   max_session_length=2,train_ratio=.5,valid_ratio=1/3)
    data=PreparedArtifact(tmp_path/'result')
    assert data.manifest['stats']['crossing_sessions_removed']==1
    assert {x.user_id for x in data.train}=={'a','b','c'}
    assert [x.user_id for x in data.valid]==['valid']
    assert [x.user_id for x in data.test]==['test']


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
    assert [len(data.train),len(data.valid),len(data.test)]==[16,2,2]
