"""独立构建商品语义图：共享规范实体IDF加权、Top-K、按来源归一化。"""
from collections import defaultdict,Counter
import math
from pathlib import Path
from .common import read_json,write_json,digest


def build(run,top_k=20,max_pairs=500000000):
    """IDF总文档数取本次实际覆盖商品数；样例图明确标sample，不冒充全量语料。"""
    if top_k<1:raise ValueError('top_k must be positive')
    run=Path(run);a=read_json(run/'product_entities.json')
    signed={k:v for k,v in a.items() if k!='fingerprint'}
    if digest(signed)!=a['fingerprint']:raise ValueError('Entity artifact changed')
    rows=a['records'];n=len(rows);postings=defaultdict(set)
    if len({r['item_id'] for r in rows})!=n:raise ValueError('Duplicate product IDs')
    for row in rows:
        for entity in row['entities']:postings[entity['canonical_id']].add(row['item_id'])
    idf={e:math.log((n+1)/(len(items)+1)) for e,items in postings.items()}
    estimated=sum(len(items)*(len(items)-1) for e,items in postings.items() if idf[e]>0)
    if estimated>max_pairs:raise ValueError('Candidate pair budget exceeded')
    edges=[]
    for row in sorted(rows,key=lambda r:r['item_id']):
        scores=Counter();source=row['item_id']
        for entity in {e['canonical_id'] for e in row['entities']}:
            if idf[entity]<=0:continue
            for target in postings[entity]:
                if source!=target:scores[target]+=idf[entity]
        best=sorted(scores.items(),key=lambda pair:(-pair[1],pair[0]))[:top_k];total=sum(score for _,score in best)
        edges.extend((source,target,score/total) for target,score in best if score>0)
    graph={'schema_version':1,'kind':'official_graphrag_shared_entity_semantic','scope':a['scope'],
           'data_fingerprint':a['data_fingerprint'],'num_items':a['catalog_size'],'covered_products':n,
           'idf_document_count':n,'top_k':top_k,'candidate_pairs':estimated,'num_edges':len(edges),
           'edge_index':[[u for u,v,w in edges],[v for u,v,w in edges]],'edge_weight':[w for u,v,w in edges],
           'entity_fingerprint':a['fingerprint']}
    if a.get('index_status') == 'partial':
        graph.update(index_status='partial', failed_text_unit_ids=a['failed_text_unit_ids'],
                     affected_item_ids=a['affected_item_ids'], products_with_entities=n-a['empty_products'])
    graph['fingerprint']=digest(graph);write_json(run/'semantic_graph.json',graph)
    write_json(run/'semantic_graph_preview.json',{k:v for k,v in graph.items() if k not in ['edge_index','edge_weight']}|
               {'edge_index':[x[:10] for x in graph['edge_index']],'edge_weight':graph['edge_weight'][:10],'preview_only':True})
    return {k:v for k,v in graph.items() if k not in ['edge_index','edge_weight']}
