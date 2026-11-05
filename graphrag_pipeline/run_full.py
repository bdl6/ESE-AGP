"""全目录GraphRAG后台编排：保存时间/日志，失败利用SDK缓存重试，完成才构图。"""
import argparse,json,shutil,time,traceback
from datetime import datetime,timezone
from pathlib import Path
from product_graph.prepare import prepare
from product_graph.official import index
from product_graph.convert import convert
from product_graph.semantic import build
from product_graph.common import write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',required=True);parser.add_argument('--run',required=True)
    args=parser.parse_args();run=Path(args.run).resolve();started=time.monotonic()
    manifest=prepare(args.source,run,limit=None)
    shutil.copytree(Path(__file__).parent/'product_graph',run/'source_snapshot',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(__file__,run/'run_full_executed.py')
    state={'status':'running','started_utc':datetime.now(timezone.utc).isoformat(),
        'selected_products':manifest['selected_products'],'concurrency':64,'gleanings':2,
        'sdk':'graphrag 3.3.0','attempt_errors':[]}
    def save(stage):
        state.update(stage=stage,elapsed_seconds=time.monotonic()-started,
            updated_utc=datetime.now(timezone.utc).isoformat())
        write_json(run/'pipeline_status.json',state)
        print(json.dumps(state,ensure_ascii=False),flush=True)
    try:
        save('index')
        for attempt in range(3):
            state['index_attempt']=attempt+1;save('index')
            try:
                state['index_result']=index(run,concurrency=64,gleanings=2,resume=attempt>0)
                break
            except Exception as error:
                state['attempt_errors'].append({'attempt':attempt+1,'error':str(error)[:1000]})
                traceback.print_exc();save('index_retry_pending' if attempt<2 else 'index_failed')
                if attempt==2:raise
                time.sleep(30)
        save('convert');state['convert_result']=convert(run)
        save('build_semantic');state['graph_result']=build(run,20)
        state.update(status='complete',finished_utc=datetime.now(timezone.utc).isoformat());save('complete')
        return 0
    except Exception as error:
        state.update(status='failed',error=str(error)[:1500],finished_utc=datetime.now(timezone.utc).isoformat())
        save('failed');traceback.print_exc();return 1

if __name__=='__main__':raise SystemExit(main())
