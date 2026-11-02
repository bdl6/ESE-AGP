"""消歧候选优化实验：词项/三元字符检索、相似度过滤、每实体top-k；不直接合并。"""
from collections import defaultdict
from difflib import SequenceMatcher
import re


def candidates(entities, limit, top_k=8, min_similarity=.65):
    """同名组星形连边避免平方比较；其余候选按同类型索引召回。候选仍需LLM判断。"""
    if top_k<1 or not 0<=min_similarity<=1:raise ValueError('Invalid candidate settings')
    names={k:re.sub(r'[^\w]','',v['normalized_name'].casefold()) for k,v in entities.items()}
    words={k:re.findall(r'\w+',v['normalized_name'].casefold()) for k,v in entities.items()}
    stop={'the','and','of','music','records','record','recordings','cd','album','various'}
    blocks=defaultdict(list);exact=defaultdict(list);signals={}
    for key,node in sorted(entities.items()):
        name=names[key];tokens={w for w in words[key] if w not in stop and len(w)>1}
        acronym=''.join(w[0] for w in words[key] if w not in {'the','and','of'})
        sig={'word:'+w for w in tokens}|{'word:'+w for w in words[key] if len(w)>1 and w not in {'the','and','of'}}|{'tri:'+name[i:i+3] for i in range(max(0,len(name)-2))}
        # 全称缩写既与其他缩写匹配，也与缩写本身的紧凑名称匹配。
        if len(acronym)>=2:sig.add('abbr:'+acronym)
        if len(name)>=2:sig.add('abbr:'+name)
        signals[key]=sig;exact[(node['type'],name)].append(key)
        for signal in sig:blocks[(node['type'],signal)].append(key)
    mandatory=set();ranked_neighbors={}
    for group in exact.values():
        for key in group[1:]:mandatory.add(tuple(sorted((group[0],key))))
    for key,node in sorted(entities.items()):
        pool=set()
        for signal in signals[key]:
            group=blocks[(node['type'],signal)]
            if len(group)<=128:pool.update(group)
        scored=[]
        for other in sorted(pool-{key}):
            if names[other]==names[key]:continue
            ratio=SequenceMatcher(None,names[key],names[other]).ratio()
            abbr=any(s.startswith('abbr:') for s in signals[key]&signals[other])
            # 单词只有泛化词重合不能自动成为候选；缩写碰撞也只是候选，不作合并。
            a={w for w in words[key] if w not in stop and len(w)>1};b={w for w in words[other] if w not in stop and len(w)>1}
            overlap=len(a&b)/max(1,len(a|b))
            containment=min(len(names[key]),len(names[other]))>=2 and (names[key] in names[other] or names[other] in names[key])
            score=max(ratio,.95 if containment else 0,.55+.4*overlap if overlap else 0,1.0 if abbr else 0)
            if score>=min_similarity:scored.append((score,other))
        ranked_neighbors[key]=sorted(scored,key=lambda x:(-x[0],x[1]))[:top_k]
    proposals=set(mandatory)
    for key,neighbors in ranked_neighbors.items():
        proposals.update(tuple(sorted((key,other))) for score,other in neighbors)
    # 同名星形边优先，截断明确记录；本策略不是穷尽的别名召回。
    ordered=sorted(proposals,key=lambda pair:(pair not in mandatory,-SequenceMatcher(None,names[pair[0]],names[pair[1]]).ratio(),pair))
    return ordered[:limit],{'candidate_pairs':len(ordered),'considered_pairs':min(limit,len(ordered)),
        'unprocessed_pairs':max(0,len(ordered)-limit),'strategy':'typed_word_trigram_acronym_topk_exact_name_star_v1',
        'top_k_per_entity':top_k,'min_similarity':min_similarity,'exact_name_star_pairs':len(mandatory),'exhaustive':False}
