"""恢复必须保留共享实体链接，并拒绝失真的原文证据。"""
import pytest
from ese_agp.entities.aggregation import aggregate,canonicalize
from ese_agp.entities.recovery import restore_links


def fixture():
    def entity(item):
        return {'entity_id':str(item),'name':'Sony','normalized_name':'sony','type':'brand',
            'evidence':[{'item_id':item,'start':0,'end':4,'quote':'Sony','evidence_id':'ev'+str(item)}]}
    chunks=[{'entities':[entity(i)],'relations':[]} for i in (1,2,3)]
    entities,relations=aggregate(chunks[:2])
    for e in entities.values():e['description']={'description':'品牌Sony','evidence_ids':[e['evidence'][0]['evidence_id']]}
    decisions=[{'left':'1','right':'2','accepted':True}]
    registry,_,_=canonicalize(entities,relations,decisions)
    key=next(iter(registry))
    return chunks,list(registry.values()),decisions,[{'stage':'describe_canonical','canonical_id':key}],{1:'Sony',2:'Sony',3:'Sony'}


def test_shared_description_failure_retains_all_links():
    chunks,registry,decisions,failures,documents=fixture()
    nodes,_,by_item=restore_links(chunks,registry,decisions,failures,documents)
    assert set(by_item)=={1,2,3}
    assert by_item[1][0]['canonical_id']==by_item[2][0]['canonical_id']
    assert by_item[1][0]['description'] is None
    assert by_item[1][0]['description_status']=='pending'
    assert by_item[3][0]['disambiguation_status']=='pending'
    assert len(nodes)==2


def test_recovery_rejects_corrupted_evidence():
    args=list(fixture());args[-1][1]='Fake'
    with pytest.raises(ValueError,match='span'):
        restore_links(*args)
