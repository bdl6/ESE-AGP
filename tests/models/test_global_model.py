"""全局稀疏传播、梯度、缓存和 checkpoint 测试。"""
import torch
import pytest
from torch.nn import functional as F
from ese_agp.models.global_graph import DirectedGraphLayer
from ese_agp.graphs.behavior import build_behavior_graph, save_graph, graph_fingerprint
from ese_agp.models.recommender import LocalSessionRecommender
from ese_agp.data.batching import collate_examples
from ese_agp.training.trainer import train
from ese_agp.training.checkpoint import load_model
from ese_agp.data.sessions import PreparedData, SessionExample


def data():
    """循环监督与孤立商品的小型数据。"""
    rows=[SessionExample((i,),i%3+1,'u',0,1) for i in (1,2,3)]
    return PreparedData(rows,rows,[],{str(i):i for i in range(1,5)},[1,2,3,4],{})


def test_sparse_direction_weight_gradient():
    """手算接收聚合，并与稠密参考比较，边权可微。"""
    layer=DirectedGraphLayer(2,0)
    with torch.no_grad():
        layer.neighbor.weight.copy_(torch.eye(2)); layer.self_path.weight.zero_(); layer.self_path.bias.zero_()
    edges=torch.tensor([[1,3],[2,2]]); weights=torch.tensor([.25,.75],requires_grad=True)
    incoming=torch.sparse_coo_tensor(edges.flip(0),weights,(5,5)).coalesce()
    h=torch.tensor([[0.,0.],[1.,0.],[0.,0.],[0.,2.],[3.,3.]])
    out=layer(h,incoming)
    torch.testing.assert_close(out[2],torch.tensor([.25,1.5]))
    torch.testing.assert_close(out,F.elu(incoming.to_dense()@h))
    out.sum().backward(); assert weights.grad.abs().sum()>0


def test_global_gradient_and_cache():
    """CE 更新全局参数，虚拟行无梯度，缓存仅限冻结评估。"""
    torch.manual_seed(42); d=data(); m=LocalSessionRecommender(4,dim=8,use_global=True,graph=build_behavior_graph(d))
    b=collate_examples(d.train,torch.device('cpu'))
    F.cross_entropy(m(b),b.target_item_ids-1).backward()
    for layer in m.global_encoder.layers:
        assert layer.neighbor.weight.grad.abs().sum()>0
        assert torch.isfinite(layer.neighbor.weight.grad).all()
    assert m.embedding.weight.grad[0].count_nonzero()==0
    assert m._global_cache is None
    m.eval()
    with torch.no_grad():
        a=m(b); cache=m._global_cache
        torch.testing.assert_close(a,m(b)); assert m._global_cache is cache
    m.train(); assert m._global_cache is None
    m.eval()
    with torch.no_grad(): m(b)
    m.load_state_dict(m.state_dict()); assert m._global_cache is None
    m.to('cpu'); assert m._global_cache is None


def test_global_training_checkpoint(tmp_path):
    """两轮实际训练与独立加载一致，拒绝图身份变化。"""
    d=data(); g=build_behavior_graph(d); gp=tmp_path/'g.json'; save_graph(g,gp)
    cfg={'data':{},'model':{'dim':8,'layers':1,'use_global':True,'global_layers':1},'graph':{'path':str(gp)},
         'training':{'device':'cpu','seed':42,'selection':'last_epoch','batch_size':3,'epochs':2,'patience':3,'top_k':2,'learning_rate':.01,'run_dir':str(tmp_path/'run')}}
    assert train(d,cfg)['graph_fingerprint']==g['fingerprint']
    a=load_model(tmp_path/'run/best.pt',d,torch.device('cpu')); b=load_model(tmp_path/'run/best.pt',d,torch.device('cpu'))
    batch=collate_examples(d.train,torch.device('cpu'))
    with torch.no_grad(): torch.testing.assert_close(a(batch),b(batch))
    d.train=d.train[:-1]; other=build_behavior_graph(d); other['data_fingerprint']=g['data_fingerprint']; other['fingerprint']=graph_fingerprint(other)
    gp.unlink(); save_graph(other,gp)
    with pytest.raises(ValueError,match='graph fingerprint'): load_model(tmp_path/'run/best.pt',data(),torch.device('cpu'))


def test_matched_init_common_weights():
    """初始化对照保留共有参数一致，只修改两张属性表。"""
    f={'category_ids':[0,1,2,1,2],'price_bin_ids':[0,1,2,3,4],'num_price_bins':50}
    torch.manual_seed(42); a=LocalSessionRecommender(4,dim=64,use_item_features=True,item_features=f)
    torch.manual_seed(42); b=LocalSessionRecommender(4,dim=64,use_item_features=True,item_features=f,attribute_init='matched')
    for name,p in a.named_parameters():
        if 'category_embedding' not in name and 'price_embedding' not in name:
            torch.testing.assert_close(p,dict(b.named_parameters())[name],rtol=0,atol=0)
    assert .03<b.attribute_encoder.price_embedding.weight[1:].std()<.07
    assert b.attribute_encoder.category_embedding.weight[0].count_nonzero()==0


def test_empty_global_is_finite():
    """无边时仍有自身通路，虚拟行始终为零。"""
    d=data(); d.train=[]; m=LocalSessionRecommender(4,dim=8,use_global=True,graph=build_behavior_graph(d))
    h=m.encode_global(); assert torch.isfinite(h).all(); assert h[0].count_nonzero()==0
