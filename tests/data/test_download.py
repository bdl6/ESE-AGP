"""下载完整性测试：使用内存 HTTP 响应，不访问公网或依赖真实大文件。"""
import gzip
import io
import re
import zlib

import pytest

from ese_agp.data import download


class Response(io.BytesIO):
    """仅实现下载器使用的响应接口，响应体保存在当前内存流中。"""
    def __init__(self, data, start, end, total, status=206):
        """记录真实区间和固定 ETag，便于检查 Range 处理。"""
        super().__init__(data)
        self.status=status
        self.headers={'Content-Range':f'bytes {start}-{end}/{total}', 'ETag':'"fixture"'}


def fake_source(monkeypatch, payload, corrupt=False, status=206):
    """用确定性的分段响应替换 urlopen，测试过程不会产生任何外网请求。"""
    def open_url(request, timeout):
        """按请求区间提供独立内存流，模拟正常或被篡改的服务响应。"""
        start,end=map(int,re.fullmatch(r'bytes=(\d+)-(\d+)',request.get_header('Range')).groups())
        body=payload[start:end+1]
        if corrupt and end>start:
            body=b'\x00'*len(body)
        return Response(body,start,end,len(payload),status)
    monkeypatch.setattr(download.urllib.request,'urlopen',open_url)
    monkeypatch.setattr(download.time,'sleep',lambda _:None)


def test_download_resumes_prefix_and_checks_gzip(monkeypatch,tmp_path):
    """已有连续前缀可复用，拼接结果必须逐字节等于源 gzip，且可完整解压。"""
    original=bytes(range(256))*20
    payload=gzip.compress(original,mtime=0)
    destination=tmp_path/'source.gz'
    destination.write_bytes(payload[:10])
    fake_source(monkeypatch,payload)
    assert download.download_file('https://fixture.invalid/data',destination,workers=2,chunk_bytes=31)==len(payload)
    assert destination.read_bytes()==payload
    assert gzip.decompress(destination.read_bytes())==original


def test_download_rejects_server_without_range(monkeypatch,tmp_path):
    """服务端忽略 Range 返回 200 时直接拒绝，不能拼接成损坏的文件。"""
    payload=gzip.compress(b'data',mtime=0)
    fake_source(monkeypatch,payload,status=200)
    with pytest.raises(RuntimeError,match='range requests'):
        download.download_file('https://fixture.invalid/data',tmp_path/'source.gz')


def test_corrupt_download_never_replaces_existing_prefix(monkeypatch,tmp_path):
    """即使每段长度正确，gzip 完整性失败也不能把损坏内容覆盖到最终文件。"""
    payload=gzip.compress(bytes(range(256))*20,mtime=0)
    destination=tmp_path/'source.gz';prefix=payload[:10];destination.write_bytes(prefix)
    fake_source(monkeypatch,payload,corrupt=True)
    with pytest.raises((OSError,EOFError,zlib.error)):
        download.download_file('https://fixture.invalid/data',destination,workers=2,chunk_bytes=31)
    assert destination.read_bytes()==prefix
