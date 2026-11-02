"""共享实体IDF语义图：逐来源累积候选，top-k后源归一，不构造N平方矩阵。"""
from collections import defaultdict, Counter
import json
import math
from pathlib import Path
from ese_agp.data.io import data_fingerprint
from ese_agp.graphs.behavior import graph_fingerprint


def build_semantic_graph(data, entity_path, top_k=20, allow_partial=False, max_candidate_pairs=50000000):
    """完整实验要求训练商品实体全部成功；试抽取图必须显式允许并标记partial。"""
    if not isinstance(top_k,int) or top_k<1: raise ValueError('top_k must be positive')
    artifact=json.loads(Path(entity_path).read_text())
    if artifact.get('fingerprint')!=graph_fingerprint(artifact):raise ValueError('Entity fingerprint mismatch')
    fingerprint=data_fingerprint(data)
    if artifact['data_fingerprint']!=fingerprint:raise ValueError('Entity dataset mismatch')
    universe=artifact.get('item_scope','train')
    if universe not in {'train','catalog'}:raise ValueError('Invalid item_scope')
    known=set(data.catalog.values()) if universe=='catalog' else {i for ex in data.train for i in (*ex.input_item_ids,ex.target_item_id) if i}
    rows=artifact['records'];ids=[r['item_id'] for r in rows]
    if len(ids)!=len(set(ids)) or any(i not in known for i in ids):raise ValueError('Invalid entity product IDs')
    good={r['item_id'] for r in rows if r['status']=='success'}
    complete=good==known and artifact.get('scope')=='full'
    if not complete and not allow_partial:raise ValueError('Partial entities require explicit allow_partial; not a full experiment')
    postings=defaultdict(set);by_item={}
    for row in rows:
        if row['status']!='success':continue
        # 类型参与实体身份，避免同名艺人和品牌被自动合并。
        keys={e.get('canonical_id') or e['type']+'::'+e['normalized_name'] for e in row['entities']}
        by_item[row['item_id']]=keys
        for key in keys:postings[key].add(row['item_id'])
    n=len(data.catalog)
    idf={key:math.log((n+1)/(len(items)+1)) for key,items in postings.items()}
    estimated=sum(len(items)*(len(items)-1) for key,items in postings.items() if idf[key]>0)
    if estimated>max_candidate_pairs:raise ValueError('Semantic candidate-pair budget exceeded; inspect generic entities before increasing budget')
    edges=[]
    for source,keys in sorted(by_item.items()):
        scores=Counter()
        for key in sorted(keys):
            if idf[key]<=0:continue
            for target in sorted(postings[key]):
                if source!=target:scores[target]+=idf[key]
        best=sorted(scores.items(),key=lambda pair:(-pair[1],pair[0]))[:top_k]
        total=sum(w for _,w in best)
        edges.extend((source,target,weight/total) for target,weight in best if weight>0)
    edges.sort()
    graph={'schema_version':1,'kind':'semantic_idf_topk_source_normalized','scope':'full' if complete else 'pilot_or_partial',
           'num_items':n,'data_fingerprint':fingerprint,'entity_fingerprint':artifact['fingerprint'],
           'top_k':top_k,'idf_formula':'log((catalog_size+1)/(entity_document_frequency+1))',
           'candidate_pairs':estimated,'num_entities':len(postings),'num_edges':len(edges),
           'num_isolated':n-len({i for u,v,w in edges for i in (u,v)}),
           'edge_index':[[u for u,v,w in edges],[v for u,v,w in edges]],'edge_weight':[w for u,v,w in edges]}
    graph['fingerprint']=graph_fingerprint(graph)
    return graph
