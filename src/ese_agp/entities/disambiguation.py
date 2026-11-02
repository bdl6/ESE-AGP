"""同类型候选生成与基于双方证据的LLM消歧，不按相似度直接合并。"""
from difflib import SequenceMatcher
import math
import re
from ese_agp.entities.extraction import normalize_entity


def candidates(entities, limit):
    """类内短列表全比较；大列表使用词项/缩写阻塞控制候选量。"""
    groups={}
    for key,node in entities.items():groups.setdefault(node['type'],[]).append(key)
    proposals=set()
    for kind,ids in groups.items():
        ids=sorted(ids);blocks={}
        for key in ids:
            words=re.findall(r'\w+',entities[key]['normalized_name'])
            signals=set(words)|{''.join(w[0] for w in words if w not in {'and','the','of'})}
            if len(ids)<=32:signals.add('__small_group__')
            for signal in signals:
                if signal:blocks.setdefault(signal,[]).append(key)
        for members in blocks.values():
            # 泛化词过大块不展开N平方；明确在输出中记录启发式策略。
            if len(members)>64:continue
            for i,a in enumerate(members):
                for b in members[i+1:]:proposals.add((a,b))
    ranked=sorted(proposals,key=lambda pair:(-SequenceMatcher(None,entities[pair[0]]['normalized_name'],entities[pair[1]]['normalized_name']).ratio(),pair))
    return ranked[:limit],{'candidate_pairs':len(ranked),'considered_pairs':min(limit,len(ranked)),
        'unprocessed_pairs':max(0,len(ranked)-limit),'strategy':'same_type_small_group_or_word_acronym_blocking; not exhaustive'}


def validate_resolution(payload, left, right, threshold):
    """合并要求高置信度、同类型和双方证据；保留拒绝与不确定决定。"""
    fields={'same_entity','canonical_name','confidence','reason','evidence_ids'}
    if set(payload)!=fields or not isinstance(payload['same_entity'],bool):raise ValueError('Resolution schema invalid')
    confidence=payload['confidence']
    if isinstance(confidence,bool) or not isinstance(confidence,(int,float)) or not math.isfinite(confidence) or not 0<=confidence<=1:raise ValueError('Resolution confidence invalid')
    if not isinstance(payload['reason'],str) or not payload['reason'].strip():raise ValueError('Resolution reason missing')
    refs=payload['evidence_ids']
    a={e['evidence_id'] for e in left['evidence']};b={e['evidence_id'] for e in right['evidence']}
    if not isinstance(refs,list) or any(not isinstance(i,str) or i not in a|b for i in refs):raise ValueError('Unknown resolution evidence')
    accepted=payload['same_entity'] and left['type']==right['type'] and confidence>=threshold
    if accepted and (payload['canonical_name'] not in {left['name'],right['name']} or not a.intersection(refs) or not b.intersection(refs)):
        raise ValueError('Merge lacks supported name or both-sided evidence')
    return {**payload,'accepted':bool(accepted),'left':left['entity_id'],'right':right['entity_id']}
