"""本目录独立使用的JSON读写、摘要和输入身份校验。"""
import hashlib,json,os
from pathlib import Path


def digest(value):
    """对JSON内容生成稳定SHA256，不依赖字典插入顺序。"""
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def write_json(path,value):
    """先写临时文件再原子替换，避免中途退出留下残缺JSON。"""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    os.replace(temp,path)


def read_json(path):
    return json.loads(Path(path).read_text())


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def load_manifest(run):
    """拒绝输入文档或映射被静默更改，保证SDK输出可映射回准备阶段的商品。"""
    run=Path(run);m=read_json(run/'manifest.json')
    for name,expected in m['input_sha256'].items():
        if file_sha(run/name)!=expected:raise ValueError('Prepared input changed: '+name)
    return m
