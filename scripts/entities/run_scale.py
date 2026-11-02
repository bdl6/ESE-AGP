"""按24小时时限标定16/32/64并发，测完整五步后决定是否启动1万商品；不训练推荐模型。"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import signal
import shutil
import threading
import time
import traceback
from ese_agp.data.io import load_prepared
from ese_agp.entities.client import ChatClient
from ese_agp.entities.pipeline import atomic_json
from ese_agp.entities import workflow_optimized as w
from ese_agp.utils.config import load_config


class DeadlineExceeded(RuntimeError):
    """达到整体时限；不能当作单商品失败跳过后继续请求。"""


def main():
    """先标定再放大；任何结论均写入报告，不以联合抽取冒充完整五步。"""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--run-dir',required=True)
    parser.add_argument('--limit',type=int,default=10000);parser.add_argument('--pilot',type=int,default=64)
    parser.add_argument('--hours',type=float,default=24)
    args=parser.parse_args();run=Path(args.run_dir).resolve()
    if args.limit<1 or args.pilot<1 or args.hours<=0:raise ValueError('Invalid scale settings')
    run.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();deadline=start+args.hours*3600
    config=load_config(args.config);config['entities'].pop('item_ids',None);config['entities']['limit']=args.limit
    shutil.copy2(__file__,run/'run_scale_executed.py')
    shutil.copytree(Path(__file__).resolve().parents[2]/'src/ese_agp',run/'source_snapshot',ignore=shutil.ignore_patterns('__pycache__'))
    config['entities'].update(concise_outputs=True,build_graphs=False,max_requests=500000,max_alias_pairs=1000000)
    state={'status':'starting','target_products':args.limit,'deadline_hours':args.hours,
           'started_utc':datetime.now(timezone.utc).isoformat(),'phase_results':[],
           'scope':'all five entity stages; graph construction and recommender training excluded',
           'cache_note':'fresh application caches per concurrency; shared warm model prefix cache not reset'}
    def update(**values):
        state.update(values);state['elapsed_seconds']=time.monotonic()-start
        atomic_json(run/'status.json',state)
    def interrupted(signum,frame):raise KeyboardInterrupt('Signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted)
    def deadline_alarm(signum,frame):raise DeadlineExceeded('Wall-clock deadline reached')
    signal.signal(signal.SIGALRM,deadline_alarm);signal.setitimer(signal.ITIMER_REAL,args.hours*3600)
    event_lock=threading.Lock();local=threading.local();current={'phase':None};totals={'requests':0,'errors':0,'completion_tokens':0}
    def emit(event):
        event={'utc':datetime.now(timezone.utc).isoformat(),'phase':current['phase'],**event}
        with event_lock:
            with (run/'events.jsonl').open('a') as out:out.write(json.dumps(event,ensure_ascii=False)+'\n')
    class Client:
        def complete(self,prompt,text):
            remaining=deadline-time.monotonic()
            if remaining<=0:raise DeadlineExceeded('24h wall-clock deadline reached')
            if not hasattr(local,'client'):local.client=ChatClient(config['entities'])
            # 每线程独立适配器，只改变本线程超时，避免单次请求跨越整体期限。
            local.client.config={**config['entities'],'timeout_seconds':min(config['entities'].get('timeout_seconds',360),remaining)}
            t=time.monotonic();raw=None;error=None
            with event_lock:totals['requests']+=1
            try:raw=local.client.complete(prompt,text);return raw
            except Exception as exc:
                error=type(exc).__name__
                with event_lock:totals['errors']+=1
                raise
            finally:
                usage=(raw or {}).get('usage',{})
                with event_lock:totals['completion_tokens']+=usage.get('completion_tokens',0) or 0
                emit({'event':'request_end','stage':getattr(local,'stage',None),'seconds':time.monotonic()-t,
                      'error':error,'finish_reason':(raw or {}).get('choices',[{}])[0].get('finish_reason'),
                      'prompt_tokens':usage.get('prompt_tokens',0),'completion_tokens':usage.get('completion_tokens',0)})
    base_runner=w.StageRunner
    class Runner(base_runner):
        def ask(self,stage,prompt,payload,validator):
            previous=getattr(local,'stage',None);local.stage=stage
            try:return super().ask(stage,prompt,payload,validator)
            finally:local.stage=previous
    w.StageRunner=Runner
    try:
        update(status='loading_data')
        data=load_prepared(config['data']['output']);docs,known=w.select_documents(data,config['entities'])
        if len(docs)!=args.limit:raise ValueError('Not enough unique training products for requested limit')
        ordered=sorted(docs,key=lambda pair:(len(pair[1]),pair[0]));n=min(args.pilot,len(ordered))
        pilot=[ordered[min(len(ordered)-1,int((i+.5)*len(ordered)/n))][0] for i in range(n)]
        # 按文本长度分层，避免只拿短标题测速；生产列表按固定哈希选择，不取最短1万件。
        atomic_json(run/'selection.json',{'data_fingerprint':data.fingerprint,'products':len(docs),'pilot_item_ids':pilot,
            'production_item_ids':[i for i,text in docs],'total_document_chars':sum(len(t) for i,t in docs),
            'sample_method':'length-stratified pilot within fixed-hash production selection',
            'document_sha256':{str(i):hashlib.sha256(t.encode()).hexdigest() for i,t in docs}})
        atomic_json(run/'base_config.json',config)
        def phase(name,workers,ids,extract_only):
            current['phase']=name;cfg=deepcopy(config)
            cfg['entities'].update(item_ids=ids,output=str(run/name/'entities'),concurrency=workers,extract_only=extract_only)
            phase_dir=run/name;phase_dir.mkdir(exist_ok=True);atomic_json(phase_dir/'config.json',cfg)
            before=dict(totals);update(status='running',phase=name,concurrency=workers)
            begin=time.monotonic();result=w.run_workflow(data,cfg,Client());elapsed=time.monotonic()-begin
            count=result.get('extracted_products',result.get('completed_products',0))
            row={'name':name,'concurrency':workers,'seconds':elapsed,'successful_products':count,
                'failed_products':result.get('failed_products',0),'extract_only':extract_only,
                'requests':totals['requests']-before['requests'],'transport_errors':totals['errors']-before['errors'],
                'completion_tokens':totals['completion_tokens']-before['completion_tokens'],'result':result}
            state['phase_results'].append(row);atomic_json(phase_dir/'report.json',row);update()
            print(json.dumps({k:v for k,v in row.items() if k!='result'},ensure_ascii=False),flush=True)
            return row
        calibration=[]
        for workers in [16,32,64]:
            row=phase('calibration_'+str(workers),workers,pilot,True);calibration.append(row)
            if row['successful_products']<.95*n or row['transport_errors']>max(2,.02*row['requests']):break
        healthy=[r for r in calibration if r['successful_products']>=.95*n]
        if not healthy:raise RuntimeError('Calibration health gate failed; no bulk run launched')
        best=max(healthy,key=lambda r:r['successful_products']/r['seconds'])
        # 与最佳标定同目录续跑，复用联合抽取缓存，后续阶段仍完整执行。
        full=phase(best['name'],best['concurrency'],pilot,False)
        pilot_five_step_seconds=best['seconds']+full['seconds']
        successful=full['successful_products'];remaining=deadline-time.monotonic()
        estimate=pilot_five_step_seconds/max(1,successful)*args.limit
        update(selected_concurrency=best['concurrency'],pilot_five_step_seconds=pilot_five_step_seconds,
               estimated_10000_seconds=estimate,estimate_safety_factor=1.2,
               estimate_note='pilot extrapolation only; large-catalog candidate count and shared server load can differ')
        if successful<.95*n or estimate*1.2>remaining:
            update(status='needs_more_optimization',reason='Full five-step pilot does not justify bulk completion within remaining 24h; bulk not launched')
        else:
            bulk=phase('production',best['concurrency'],[i for i,text in docs],False)
            update(status='complete' if bulk['result'].get('complete') else 'partial',production_result=bulk['result'])
    except KeyboardInterrupt as exc:update(status='interrupted',error=str(exc))
    except Exception as exc:
        update(status='deadline_exceeded' if isinstance(exc,DeadlineExceeded) else 'failed',error=type(exc).__name__+': '+str(exc))
        traceback.print_exc()
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        state['elapsed_seconds']=time.monotonic()-start
        state['within_deadline']=state['elapsed_seconds']<=args.hours*3600
        atomic_json(run/'report.json',state)
        lines=['# 万商品实体流程速度优化','', '状态：'+state['status'],
               '范围：五步实体流程，排除构图和推荐模型训练。24小时从标定开始计算。',
               '预测不是完成承诺；没有通过完整流程测速门槛不会盲目启动万商品。','',
               '详细结果见report.json；各阶段config.json/report.json及entities目录保留。']
        (run/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return 0 if state['status']=='complete' else 1


if __name__=='__main__':raise SystemExit(main())
