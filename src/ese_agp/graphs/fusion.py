"""尺度对齐后的行为图和语义图按beta相加，合并同向重复边。"""
from collections import Counter
import math
from ese_agp.graphs.behavior import graph_fingerprint


def fuse_graphs(behavior, semantic, beta=0.5, allow_partial=False):
    """不在融合后重新归一；只有一种关系的行可能总权重小于1。"""
    if not isinstance(beta,(int,float)) or not math.isfinite(beta) or not 0<=beta<=1:raise ValueError('beta must be in [0,1]')
    for graph in [behavior,semantic]:
        if graph['fingerprint']!=graph_fingerprint(graph):raise ValueError('Graph fingerprint mismatch')
    if behavior['num_items']!=semantic['num_items'] or behavior['data_fingerprint']!=semantic['data_fingerprint']:
        raise ValueError('Graph dataset/catalog mismatch')
    if behavior['kind']!='train_adjacent_directed_source_normalized' or semantic['kind']!='semantic_idf_topk_source_normalized':
        raise ValueError('Expected behavior and semantic graphs')
    scope=semantic['scope']
    if scope!='full' and not allow_partial:raise ValueError('Partial semantic graph requires explicit allow_partial')
    weights=Counter()
    for graph,scale in [(behavior,beta),(semantic,1-beta)]:
        for (u,v),weight in zip(zip(*graph['edge_index']),graph['edge_weight']):weights[u,v]+=scale*weight
    edges=sorted((u,v,w) for (u,v),w in weights.items() if w>0)
    graph={'schema_version':1,'kind':'behavior_semantic_beta_fusion','scope':scope,'beta':beta,
           'num_items':behavior['num_items'],'data_fingerprint':behavior['data_fingerprint'],
           'behavior_fingerprint':behavior['fingerprint'],'semantic_fingerprint':semantic['fingerprint'],
           'num_edges':len(edges),'num_isolated':behavior['num_items']-len({i for u,v,w in edges for i in (u,v)}),
           'edge_index':[[u for u,v,w in edges],[v for u,v,w in edges]],'edge_weight':[w for u,v,w in edges]}
    graph['fingerprint']=graph_fingerprint(graph)
    return graph
