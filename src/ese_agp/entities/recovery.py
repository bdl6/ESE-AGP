"""只读恢复已通过抽取校验的实体链接，不调用LLM，不假称补齐描述或消歧。"""
from copy import deepcopy
from ese_agp.entities.aggregation import aggregate, canonicalize
from ese_agp.graphs.behavior import graph_fingerprint


def restore_links(chunks, source_registry, decisions, failures, documents):
    """已消歧实体沿用规范ID；被过度排除的实体单独保留，等待后续消歧。"""
    entities, relations = aggregate(chunks)
    registry = {e['canonical_id']: deepcopy(e) for e in source_registry}
    mapping = {member: key for key, entity in registry.items() for member in entity['members']}
    if not set(mapping) <= set(entities):
        raise ValueError('Registry members do not match extraction')
    pending_canonical = {f['canonical_id'] for f in failures if 'canonical_id' in f}
    for key, entity in registry.items():
        entity['description_status'] = 'pending' if key in pending_canonical else 'validated'
        if key in pending_canonical:
            # 合并描述失败时不能使用canonicalize继承的单成员描述冒充完成。
            entity['description'] = None
        entity['disambiguation_status'] = 'processed_with_stage_errors' if any(f['stage']=='resolve' for f in failures) else 'processed'
    missing = {key: entity for key, entity in entities.items() if key not in mapping}
    extra, _, extra_mapping = canonicalize(missing, {}, [])
    for entity in extra.values():
        entity.update(description=None, description_status='pending', disambiguation_status='pending')
    registry.update(extra)
    # 复用原批准的消歧结果重新映射全部关系；新恢复实体不擅自跨商品合并。
    _, rewritten, complete_mapping = canonicalize(entities, relations, decisions)
    mapping.update(extra_mapping)
    if complete_mapping != mapping:
        raise ValueError('Recovered canonical mapping changed existing identities')
    # 商品归属以原文证据为准；逐条复核绝对偏移。
    by_item = {}
    for key, entity in sorted(registry.items()):
        for item in sorted({e['item_id'] for e in entity['evidence']}):
            if item not in documents:
                raise ValueError('Evidence references unknown product')
            evidence = [e for e in entity['evidence'] if e['item_id']==item]
            for e in evidence:
                if documents[item][e['start']:e['end']] != e['quote']:
                    raise ValueError('Evidence span does not match original document')
            by_item.setdefault(item, []).append({k:entity.get(k) for k in (
                'canonical_id','name','type','normalized_name','description','description_status','disambiguation_status') } | {'evidence': evidence})
    return registry, rewritten, by_item
