"""全目录实体流程，读取已核验缓存来源，64并发，完成后生成语义图与融合图。"""
import argparse,json,threading,time,shutil,signal
from pathlib import Path
from datetime import datetime,timezone
from ese_agp.entities.client import ChatClient
from ese_agp.entities import workflow_optimized as w
from ese_agp.entities.pipeline import atomic_json
from ese_agp.data.io import load_prepared
from ese_agp.utils.config import load_config


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--run-dir',required=True)
    a=parser.parse_args();run=Path(a.run_dir);run.mkdir(parents=True,exist_ok=False);c=load_config(a.config)
    shutil.copy2(__file__,run/'run_catalog_executed.py')
    shutil.copytree(Path(__file__).resolve().parents[2]/'src/ese_agp',run/'source_snapshot',ignore=shutil.ignore_patterns('__pycache__'))
    atomic_json(run/'config.json',c);start=time.time();state={'status':'running','started_utc':datetime.now(timezone.utc).isoformat(),'concurrency':64,'scope':'full v4 catalog metadata; no test behavior edges; no recommender training'}
    atomic_json(run/'status.json',state);local=threading.local()
    class Client:
        def complete(self,prompt,text):
            if not hasattr(local,'client'):local.client=ChatClient(c['entities'])
            return local.client.complete(prompt,text)
    def stop(signum,frame):raise KeyboardInterrupt('Signal '+str(signum))
    signal.signal(signal.SIGTERM,stop)
    try:
        data=load_prepared(c['data']['output']);state['requested_products']=len(data.catalog)
        atomic_json(run/'status.json',state)
        result=w.run_workflow(data,c,Client());state.update(status='complete' if result['complete'] else 'partial',result=result)
    except KeyboardInterrupt as error:state.update(status='interrupted',error=str(error))
    except Exception as error:
        state.update(status='failed',error=type(error).__name__+': '+str(error));__import__('traceback').print_exc()
    finally:
        state.update(elapsed_seconds=time.time()-start,finished_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(run/'status.json',state);atomic_json(run/'report.json',state)
        print(json.dumps(state,ensure_ascii=False),flush=True)
    return 0 if state['status']=='complete' else 1


if __name__=='__main__':raise SystemExit(main())
