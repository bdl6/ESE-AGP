"""行为图边权、数据隔离和产物身份测试。"""
import copy, json
import pytest
from ese_agp.data.sessions import PreparedData, SessionExample
from ese_agp.graphs.behavior import build_behavior_graph, save_graph, load_graph


def ex(seq):
    """最后一项作为监督标签。"""
    return SessionExample(tuple(seq[:-1]),seq[-1],'u',0,1)


def fixture_data():
    """手工数据覆盖重复转移、OOV 和孤立节点。"""
    return PreparedData([ex([1,2,3]),ex([1,2,4]),ex([1,3]),ex([1,0,4])],[ex([4,1])],[ex([3,4])],{str(i):i for i in range(1,6)},[1,2,3,4,5],{})


def test_counts_weights_no_oov_bridge():
    """手算源归一，训练标签入图且 heldout 不入图。"""
    g=build_behavior_graph(fixture_data())
    assert list(zip(*g['edge_index']))==[(1,2),(1,3),(2,3),(2,4)]
    assert g['counts']==[2,1,1,1]
    assert g['edge_weight']==[2/3,1/3,.5,.5]
    assert g['num_isolated']==1


def test_heldout_does_not_change_edges():
    """heldout 改变可改变数据身份，但不得改变训练边。"""
    d=fixture_data(); a=build_behavior_graph(d); d.test=[ex([5,5])]; d.valid=[]
    b=build_behavior_graph(d)
    for key in ['edge_index','counts','edge_weight']: assert a[key]==b[key]


def test_graph_integrity(tmp_path):
    """校验文件篡改、目录错配、无提示覆盖。"""
    d=fixture_data(); g=build_behavior_graph(d); path=tmp_path/'g.json'
    save_graph(g,path); assert load_graph(path,d)==g
    with pytest.raises(FileExistsError): save_graph(g,path)
    other=copy.deepcopy(d); other.catalog['extra']=6
    with pytest.raises(ValueError,match='fingerprints'): load_graph(path,other)
    g['edge_weight'][0]=.1; path.write_text(json.dumps(g))
    with pytest.raises(ValueError,match='fingerprint'): load_graph(path,d)


def test_empty_and_self_edges():
    """重复相邻商品产生自转移，空图合法。"""
    d=fixture_data(); d.train=[ex([1,1]),ex([2,0,3])]
    assert build_behavior_graph(d)['edge_index']==[[1],[1]]
    d.train=[]; assert build_behavior_graph(d)['edge_index']==[[],[]]
