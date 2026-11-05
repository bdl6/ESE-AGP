"""标题兜底不能捏造品牌或关系，证据必须精确定位原文。"""
from ese_agp.entities.joint import title_entity_result


def test_title_only_product_with_evidence():
    text='title: Beats, Rhymes & Life\nfeatures:\n'
    result=title_entity_result(55,text)
    assert len(result['entities'])==1
    entity=result['entities'][0];assert entity['type']=='product'
    evidence=entity['evidence'][0]
    assert text[evidence['start']:evidence['end']]==evidence['quote']
    assert not result['relations']


def test_no_title_means_no_fabricated_entity():
    assert title_entity_result(1,'description: no title') is None
