"""论文协议验收：随机会话切分、全局过滤、训练未见商品与无验证训练。"""
import gzip
import json

import pytest
import torch

from ese_agp.amazon import prepare_amazon
from ese_agp.artifacts import PreparedArtifact
from ese_agp.model import collate_examples
from ese_agp.training import train, load_model


def make_files(tmp_path):
    """20 个跨月份会话，避免随机划分恰好看起来像时间划分。"""
    rows = [{'user_id': f'u{day:02}', 'parent_asin': item,
             'timestamp': 1600000000000 + day * 86400000 * 30 + offset}
            for day in range(20) for offset, item in enumerate(['A', 'B', 'C'])]
    meta = [{'parent_asin': item, 'categories': ['Music', 'Rock'], 'price': price}
            for item, price in [('A', 10), ('B', 20), ('C', 30)]]
    reviews, metadata = tmp_path / 'reviews.gz', tmp_path / 'meta.gz'
    write_rows(reviews, rows)
    write_rows(metadata, meta)
    return reviews, metadata, rows, meta


def write_rows(path, rows):
    """保存真实 Amazon schema 的 gzip JSONL。"""
    with gzip.open(path, 'wt', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row) + '\n')


def build(reviews, metadata, output, seed=42):
    """使用论文全局 user>=3、item>=5、随机 9:1 配置生成产物。"""
    prepare_amazon(reviews, metadata, output, protocol='paper', train_ratio=.9,
                   valid_ratio=0, split_seed=seed)
    return PreparedArtifact(output)


def test_paper_split_is_reproducible_disjoint_and_not_temporal(tmp_path):
    """整段会话仅归属一个集合；种子可复现，换种子改变分配，不按时间截断。"""
    reviews, metadata, _, _ = make_files(tmp_path)
    first = build(reviews, metadata, tmp_path / 'first')
    second = build(reviews, metadata, tmp_path / 'second')
    changed = build(reviews, metadata, tmp_path / 'changed', seed=7)
    assert [len(first.train), len(first.valid), len(first.test)] == [18, 0, 2]
    assert list(first.train) == list(second.train)
    assert first.fingerprint == second.fingerprint
    assert {x.session_id for x in first.train}.isdisjoint(x.session_id for x in first.test)
    assert {x.session_id for x in first.test} != {x.session_id for x in changed.test}
    assert max(x.session_end for x in first.train) > min(x.session_end for x in first.test)
    assert first.manifest['parameters']['filter_scope'] == 'global_before_split'


def test_paper_global_catalog_is_distinct_from_training_seen_items(tmp_path):
    """仅测试期交互的商品可通过全局频次过滤：ID 已知，但训练交互未见。"""
    reviews, metadata, rows, meta = make_files(tmp_path)
    original = build(reviews, metadata, tmp_path / 'original')
    user = original.test[0].user_id
    start = original.test[0].session_start
    rows = [row for row in rows if row['user_id'] != user]
    rows += [{'user_id': user, 'parent_asin': 'Z', 'timestamp': start + i} for i in range(5)]
    meta += [{'parent_asin': 'Z', 'categories': ['Music', 'Rock'], 'price': 100}]
    write_rows(reviews, rows)
    write_rows(metadata, meta)
    data = build(reviews, metadata, tmp_path / 'result')
    z = data.catalog['Z']
    assert all(z not in (*ex.input_item_ids, ex.target_item_id) for ex in data.train)
    assert any(ex.target_item_id == z for ex in data.test)
    assert data.manifest['stats']['oov_targets'] == {'train': 0, 'valid': 0, 'test': 0}
    assert data.manifest['stats']['targets_unseen_in_train']['test'] == 1
    assert data.manifest['stats']['catalog_items_unseen_in_train'] == 1
    # 价格均值包含全局目录中的 Z，和 temporal 的训练目录拟合语义不同。
    assert data.item_features['price_stats']['Rock']['mean'] == 40


def test_paper_deletes_invalid_and_low_frequency_events_then_truncates(tmp_path):
    """论文先删无效/低频商品，再截断和留一，不沿用时间协议的 OOV 占位。"""
    reviews, metadata, rows, meta = make_files(tmp_path)
    rows = [row for row in rows if row['user_id'] != 'u00']
    sequence = ['A'] * 23 + ['INVALID', 'RARE', 'B', 'C']
    rows += [{'user_id': 'u00', 'parent_asin': item, 'timestamp': 1600000000000 + i}
             for i, item in enumerate(sequence)]
    meta += [{'parent_asin': 'INVALID', 'categories': ['Music'], 'price': None},
             {'parent_asin': 'RARE', 'categories': ['Music'], 'price': 7}]
    write_rows(reviews, rows)
    write_rows(metadata, meta)
    data = build(reviews, metadata, tmp_path / 'result')
    ex = next(x for split in (data.train, data.test) for x in split if x.user_id == 'u00')
    assert len(ex.input_item_ids) == 19
    assert ex.input_item_ids[-1] == data.catalog['B']
    assert ex.target_item_id == data.catalog['C']
    assert 0 not in ex.input_item_ids
    assert 'INVALID' not in data.catalog and 'RARE' not in data.catalog


def test_paper_fixed_epochs_never_reads_test_set(tmp_path):
    """无验证集时固定轮数保存最终模型；测试样本访问即抛错，证明未拿测试选模型。"""
    reviews, metadata, _, _ = make_files(tmp_path)
    data = build(reviews, metadata, tmp_path / 'result')
    class ForbiddenTestSet:
        """训练期间禁止任何测试集访问。"""
        def __len__(self):
            raise AssertionError('Training must not inspect test examples')
        def __getitem__(self, index):
            raise AssertionError('Training must not inspect test examples')
    data.test = ForbiddenTestSet()
    config = {'data': {}, 'model': {'dim': 8, 'layers': 1, 'dropout': 0},
              'training': {'device': 'cpu', 'seed': 42, 'batch_size': 8,
                           'learning_rate': .001, 'epochs': 2, 'patience': 1,
                           'top_k': 2, 'selection': 'last_epoch', 'run_dir': str(tmp_path / 'run')}}
    report = train(data, config)
    assert report['best_epoch'] == 2
    assert report['best_valid_mrr'] is None and report['valid'] is None
    model = load_model(tmp_path / 'run' / 'best.pt', data, torch.device('cpu'))
    logits = model(collate_examples([data.train[0]], torch.device('cpu')))
    assert logits.shape == (1, len(data.catalog)) and torch.isfinite(logits).all()
    config['training']['selection'] = 'validation_mrr'
    with pytest.raises(ValueError, match='nonempty valid'):
        train(data, config)


def test_paper_rejects_implicit_validation_split(tmp_path):
    """论文两切分配置不能悄悄变成三切分。"""
    reviews, metadata, _, _ = make_files(tmp_path)
    with pytest.raises(ValueError, match='train_ratio=0.9'):
        prepare_amazon(reviews, metadata, tmp_path / 'result', protocol='paper')


def test_cli_rejects_empty_validation_metrics(tmp_path, monkeypatch):
    """论文数据默认 valid 评估必须明确报错，不能输出零样本指标。"""
    import sys
    import yaml
    from ese_agp.cli import main
    reviews, metadata, _, _ = make_files(tmp_path)
    data = build(reviews, metadata, tmp_path / 'result')
    config = {'data': {'output': str(data.directory)}, 'model': {},
              'training': {'run_dir': str(tmp_path / 'run'), 'top_k': 2}}
    config_path = tmp_path / 'config.yaml'
    config_path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(sys, 'argv', ['ese_agp', 'baseline', '--config', str(config_path)])
    with pytest.raises(ValueError, match='Selected split is empty'):
        main()
