"""真实数据产物读写：Parquet 保存样本，JSON 保存目录/特征/统计与文件指纹。"""
from collections.abc import Sequence
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

from ese_agp.data.sessions import SessionExample


def sha256_file(path):
    """分块计算文件摘要，不把压缩原始文件或 Parquet 整体读入 Python 字节串。"""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, value):
    """写入稳定排序、UTF-8、拒绝 NaN 的 JSON，用于可检查的小型清单。"""
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    indent=2, allow_nan=False) + '\n', encoding='utf-8')


class ParquetSessions(Sequence):
    """把预处理样本保留为紧凑 Arrow 列，按训练所需索引临时生成样本对象。

    只加载编号后的样本列，不加载原始评论或商品文本；不是把全部样本转成 Python list。
    当前样本列仍在内存中，数据更大时可再扩展 row-group 缓存或 IterableDataset。
    """
    def __init__(self, path):
        """path 指向一份固定切分的 Parquet，磁盘文件映射由 PyArrow 管理。"""
        self.table = pq.read_table(path, memory_map=True)
        self.columns = {name: self.table[name] for name in self.table.column_names}

    def __len__(self):
        """返回会话样本数，不是交互事件数。"""
        return self.table.num_rows

    def __getitem__(self, index):
        """支持单项或切片；转成与第一阶段相同的 SessionExample 接口。"""
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self))) ]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        row = {name: column[index].as_py() for name, column in self.columns.items()}
        row['input_item_ids'] = tuple(row['input_item_ids'])
        return SessionExample(**row)


class PreparedArtifact:
    """真实数据的运行时容器；对训练器提供与小数据一致的属性和训练目录。"""
    def __init__(self, directory, verify=True):
        """加载产物并验证文件内容；避免商品映射/特征被替换后继续用旧 checkpoint。"""
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / 'manifest.json').read_text())
        if self.manifest.get('schema_version') != 2:
            raise ValueError('Unsupported artifact schema')
        hashes = self.manifest['artifact_sha256']
        for filename, expected in hashes.items():
            path = self.directory / filename
            if path.resolve().parent != self.directory.resolve():
                raise ValueError('Invalid artifact file path')
            if verify and sha256_file(path) != expected:
                raise ValueError('Artifact checksum mismatch: ' + filename)
        identity = {key: self.manifest[key] for key in ('protocol', 'parameters', 'source_sha256', 'artifact_sha256')}
        actual = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        if actual != self.manifest['fingerprint']:
            raise ValueError('Manifest fingerprint mismatch')
        self.fingerprint = actual
        self.catalog = json.loads((self.directory / 'catalog.json').read_text())
        self.item_features = json.loads((self.directory / 'item_features.json').read_text())
        self.popularity = json.loads((self.directory / 'popularity.json').read_text())
        if sorted(self.catalog.values()) != list(range(1, len(self.catalog) + 1)):
            raise ValueError('Catalog IDs are not contiguous')
        for split in ('train', 'valid', 'test'):
            setattr(self, split, ParquetSessions(self.directory / (split + '.parquet')))
