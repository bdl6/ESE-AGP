"""实体准备的边界与证据检查，不调用真实服务。"""
import pytest
from ese_agp.entities.extraction import text_chunks, normalize_entity, validate_entities, cache_key


def test_chunk_overlap_and_limit():
    """保留重叠且总文本裁到6000。"""
    text=''.join(chr(1000+i) for i in range(7000)); chunks=text_chunks(text)
    assert chunks[0][-200:]==chunks[1][:200]
    assert chunks[-1][-1]==text[5999]
    assert text_chunks('')==[]


def test_normalization_and_cache_identity():
    """不进行猜测式同义合并，服务/模型变化必须让缓存失效。"""
    assert normalize_entity('  ＣＤ  Rock ')== 'cd rock'
    assert cache_key('a','m','p')!=cache_key('a','n','p')
    assert cache_key('a','m','p')!=cache_key('a','m','q')


def test_entity_schema_evidence_dedup():
    """空结果与失败不同，必须有原文连续证据。"""
    e={'type':'artist','name':'Singer','evidence':'Singer'}
    assert len(validate_entities({'entities':[e,e]},'Singer album'))==1
    assert validate_entities({'entities':[]},'album')==[]
    with pytest.raises(ValueError): validate_entities({'entities':[e]},'unrelated')
    with pytest.raises(ValueError): validate_entities({'entities':[dict(e,type='unknown')]},'Singer')
