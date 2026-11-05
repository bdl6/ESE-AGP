"""独立命令入口，使用本目录专用venv；不调用原项目CLI。"""
import argparse,json
from .prepare import prepare
from .official import index
from .convert import convert
from .semantic import build


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--source',required=True);p.add_argument('--run',required=True)
    g=p.add_mutually_exclusive_group();g.add_argument('--limit',type=int,default=3);g.add_argument('--all',action='store_true');g.add_argument('--item-ids',type=int,nargs='+')
    p=sub.add_parser('index');p.add_argument('--run',required=True);p.add_argument('--concurrency',type=int,default=64)
    p.add_argument('--gleanings',type=int,default=2);p.add_argument('--resume',action='store_true')
    for name in ['convert','build-semantic']:
        p=sub.add_parser(name);p.add_argument('--run',required=True)
        if name=='convert':p.add_argument('--allow-partial',action='store_true')
        if name=='build-semantic':p.add_argument('--top-k',type=int,default=20)
    a=parser.parse_args()
    if a.command=='prepare':result=prepare(a.source,a.run,None if a.all else a.limit,a.item_ids)
    elif a.command=='index':result=index(a.run,a.concurrency,a.gleanings,a.resume)
    elif a.command=='convert':result=convert(a.run,allow_partial=a.allow_partial)
    else:result=build(a.run,a.top_k)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
