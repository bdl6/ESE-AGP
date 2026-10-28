"""官方数据断点下载；分段请求严格检查 Content-Range，完成后校验 gzip 完整性。"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import json
import os
from pathlib import Path
import shutil
import time
import urllib.request

URLS = {
    'CDs_and_Vinyl.jsonl.gz': 'https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/CDs_and_Vinyl.jsonl.gz',
    'meta_CDs_and_Vinyl.jsonl.gz': 'https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_CDs_and_Vinyl.jsonl.gz',
}


def download_file(url, destination, workers=6, chunk_bytes=32*1024*1024):
    """保留已有同源文件前缀，下载剩余分段；只有完整拼接且 gzip CRC 通过才替换。

    已有目标必须是该 URL 的连续前缀，下载时不能有其他进程修改它。
    缓存目录记录 URL/总长/前缀长度与分段，网络失败可以再次运行，不丢已完成分段。
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={'Range':'bytes=0-0','Accept-Encoding':'identity'})
    with urllib.request.urlopen(request, timeout=45) as response:
        content_range = response.headers.get('Content-Range', '')
        if response.status != 206 or not content_range.startswith('bytes 0-0/'):
            raise RuntimeError('Server must support validated range requests')
        total = int(content_range.split('/')[-1])
        etag = response.headers.get('ETag')
        response.read()
    prefix = destination.stat().st_size if destination.exists() else 0
    if prefix > total:
        raise RuntimeError('Existing file is larger than official source')
    if prefix == total:
        with gzip.open(destination, 'rb') as stream:
            for _ in iter(lambda: stream.read(4*1024*1024), b''): pass
        return total
    cache = destination.with_name(destination.name + '.chunks')
    cache.mkdir(exist_ok=True)
    identity = {'url':url,'total':total,'prefix':prefix,'etag':etag}
    marker = cache/'source.json'
    if marker.exists() and json.loads(marker.read_text()) != identity:
        raise RuntimeError('Download identity changed; use another destination')
    marker.write_text(json.dumps(identity,indent=2))
    original_stat = (destination.stat().st_size,destination.stat().st_mtime_ns) if destination.exists() else None
    ranges = [(start, min(start+chunk_bytes,total)-1) for start in range(prefix,total,chunk_bytes)]
    def fetch(bounds):
        """下载并校验一个字节区间；成功的缓存段可在重试时复用。"""
        start,end = bounds
        path = cache/f'{start}-{end}.part'
        expected = end-start+1
        if path.exists() and path.stat().st_size == expected:
            return path
        for attempt in range(4):
            try:
                headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'}
                if etag and not etag.startswith('W/'): headers['If-Range']=etag
                with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=60) as response:
                    if response.status != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{total}':
                        raise RuntimeError('Unexpected Content-Range')
                    if etag and response.headers.get('ETag') != etag:
                        raise RuntimeError('Source ETag changed')
                    with path.open('wb') as stream:
                        shutil.copyfileobj(response,stream,1024*1024)
                if path.stat().st_size != expected:
                    raise RuntimeError('Incomplete range')
                return path
            except Exception:
                if attempt == 3: raise
                time.sleep(2**attempt)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(fetch,bounds) for bounds in ranges]
        for completed,future in enumerate(as_completed(futures),1):
            future.result()
            print(json.dumps({'download':destination.name,'parts_complete':completed,'parts_total':len(ranges)}),flush=True)
    assembled = destination.with_name(destination.name + '.assembling')
    with assembled.open('wb') as stream:
        if prefix:
            with destination.open('rb') as previous: shutil.copyfileobj(previous,stream,4*1024*1024)
        for start,end in ranges:
            with (cache/f'{start}-{end}.part').open('rb') as part:
                shutil.copyfileobj(part,stream,4*1024*1024)
    if assembled.stat().st_size != total:
        raise RuntimeError('Assembled size mismatch')
    with gzip.open(assembled,'rb') as stream:
        for _ in iter(lambda:stream.read(4*1024*1024),b''): pass
    current_stat=(destination.stat().st_size,destination.stat().st_mtime_ns) if destination.exists() else None
    if current_stat != original_stat:
        raise RuntimeError('Destination modified by another downloader')
    os.replace(assembled,destination)
    return total


def main():
    """命令入口：下载固定官方类目的交互/元数据，打印可检查的进度与完成标记。"""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='data/raw/amazon2023')
    parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args()
    if not 1<=args.workers<=8: parser.error('workers must be 1..8')
    sizes={name:download_file(url,Path(args.output)/name,args.workers) for name,url in URLS.items()}
    (Path(args.output)/'download_complete.json').write_text(json.dumps({'sources':URLS,'bytes':sizes},indent=2))
    print(json.dumps({'download_complete':True,'bytes':sizes}),flush=True)


if __name__=='__main__':
    main()
