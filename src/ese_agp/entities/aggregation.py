"""跨块、跨轮聚合实体和关系，保留每条原文证据及强度观测。"""
from copy import deepcopy
from ese_agp.entities.extraction import normalize_entity
from ese_agp.entities.joint import stable_id


def merge_evidence(left, right):
    """按原文位置去重，重叠分块的同一证据只保留一次。"""
    return list({e['evidence_id']:e for e in [*left,*right]}.values())


def aggregate(results):
    """同一商品内相同类型/名称先聚合；跨商品同名也须后续消歧，防止同名异义。"""
    entities={};relations={}
    for result in results:
        for entity in result['entities']:
            key=entity['entity_id']
            if key not in entities:entities[key]=deepcopy(entity);entities[key]['aliases']=[entity['name']]
            else:
                entities[key]['evidence']=merge_evidence(entities[key]['evidence'],entity['evidence'])
                entities[key]['aliases']=sorted(set(entities[key]['aliases']+[entity['name']]))
        for relation in result['relations']:
            key=relation['relation_id']
            if key not in relations:relations[key]=deepcopy(relation);relations[key]['strength_observations']=[relation['strength']]
            else:
                relations[key]['evidence']=merge_evidence(relations[key]['evidence'],relation['evidence'])
                relations[key]['strength_observations'].append(relation['strength'])
                relations[key]['strength']=max(relations[key]['strength_observations'])
    return entities,relations


def canonicalize(entities, relations, decisions):
    """只合并经批准的同类型别名；规范名取组内最长已有名称，不创造新名字。"""
    parent={key:key for key in entities}
    def find(key):
        """查找并压缩消歧并查集路径。"""
        while parent[key]!=key:parent[key]=parent[parent[key]];key=parent[key]
        return key
    for decision in decisions:
        if decision.get('accepted'):
            a,b=decision['left'],decision['right']
            if entities[a]['type']!=entities[b]['type']:raise ValueError('Cannot merge different entity types')
            parent[find(b)]=find(a)
    groups={}
    for key in entities:groups.setdefault(find(key),[]).append(key)
    registry={};mapping={}
    for members in groups.values():
        chosen=sorted(members,key=lambda key:(-len(entities[key]['name']),entities[key]['normalized_name']))[0]
        source=entities[chosen];canonical_id=stable_id('canonical',source['type'],source['normalized_name'],sorted(members))
        node=deepcopy(source);node.update(canonical_id=canonical_id,members=sorted(members),aliases=[],evidence=[],descriptions=[])
        for member in members:
            mapping[member]=canonical_id;entity=entities[member]
            node['aliases']+=entity['aliases'];node['evidence']=merge_evidence(node['evidence'],entity['evidence'])
            if 'description' in entity:node['descriptions'].append(entity['description'])
        node['aliases']=sorted(set(node['aliases']));registry[canonical_id]=node
    rewritten={}
    for edge in relations.values():
        head,tail=mapping[edge['head']],mapping[edge['tail']]
        key=stable_id('relation',head,tail,normalize_entity(edge['relation']))
        if key not in rewritten:rewritten[key]={**deepcopy(edge),'relation_id':key,'head':head,'tail':tail}
        else:
            rewritten[key]['evidence']=merge_evidence(rewritten[key]['evidence'],edge['evidence'])
            rewritten[key]['strength_observations']+=edge['strength_observations']
            rewritten[key]['strength']=max(rewritten[key]['strength_observations'])
    return registry,rewritten,mapping
