"""在tmux内测量100件商品完整实体工作流，保存分阶段事件与成功/失败报告。"""
import argparse
import threading
from collections import defaultdict
from datetime import datetime,timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time
from zoneinfo import ZoneInfo
from ese_agp.data.io import load_prepared
from ese_agp.entities.client import ChatClient
from ese_agp.entities.pipeline import atomic_json
from ese_agp.entities import workflow_optimized as workflow
from ese_agp.utils.config import load_config


def now():
    """同时保存UTC与北京时间，避免服务器时区影响分析。"""
    utc=datetime.now(timezone.utc)
    return {'utc':utc.isoformat(),'beijing':utc.astimezone(ZoneInfo('Asia/Shanghai')).isoformat()}


def percentile(values,q):
    """返回排序后的最近秩分位数，空数据返回None。"""
    if not values:return None
    ordered=sorted(values);return ordered[max(0,min(len(ordered)-1,__import__('math').ceil(q*len(ordered))-1))]


def main():
    """配置、源代码快照和时钟固化之后开始测量；中断也保存报告。"""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--run-dir',required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[2]
    config=load_config(args.config);run=Path(args.run_dir).resolve();run.mkdir(parents=True,exist_ok=True)
    if (run/'timing_report.json').exists():raise FileExistsError('Existing finished report; choose another run-dir')
    log=logging.getLogger('entity-benchmark');log.setLevel(logging.INFO)
    fmt=logging.Formatter('%(asctime)s %(message)s')
    for handler in [logging.StreamHandler(sys.stdout),logging.FileHandler(run/'experiment.log',encoding='utf-8')]:
        handler.setFormatter(fmt);log.addHandler(handler)
    output=Path(config['entities']['output'])
    if output.exists():raise FileExistsError('Timing run requires cold application output directory')
    snapshot=run/'source_snapshot';snapshot.mkdir(exist_ok=False)
    source_hashes={}
    for source in sorted((root/'src/ese_agp').rglob('*.py')):
        relative=source.relative_to(root);target=snapshot/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        source_hashes[str(relative)]=hashlib.sha256(source.read_bytes()).hexdigest()
    shutil.copy2(__file__,snapshot/'benchmark_workflow.py')
    atomic_json(run/'config.json',config);atomic_json(run/'source_sha256.json',source_hashes)
    atomic_json(run/'environment.json',{'python':sys.version,'executable':sys.executable,'pid':os.getpid(),
        'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        'git_status':subprocess.check_output(['git','status','--short'],cwd=root,text=True),
        'application_cache':'cold; model server already loaded; model prefix cache not reset',
        'execution':'bounded concurrent workflow; deterministic collection; no recommender training',
        'concurrency':config['entities'].get('concurrency',4),
        'model':config['entities']['model'],'base_url':config['entities']['base_url']})
    begin=time.perf_counter();started=now();events=[];active=threading.local();active.stage=None;attempts=[];event_lock=threading.RLock()
    atomic_json(run/'status.json',{'status':'running','started':started,'pid':os.getpid(),'products_requested':100})

    def emit(event):
        """每条事件立即刷盘，运行中也能读取当前HTTP请求与阶段。"""
        event={'time':now(),'elapsed_seconds':time.perf_counter()-begin,**event}
        with event_lock:
            with (run/'events.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(event,ensure_ascii=False,allow_nan=False)+'\n')
            events.append(event)

    class MeasuredClient:
        """只测量网络调用，不改变提示词、响应或重试算法。"""
        def __init__(self,client):
            """保留原适配器。"""
            self.client=client;self.local=threading.local()
        def complete(self,prompt,text):
            """记录真实请求时延、token及错误类别，不输出密钥或商品原文。"""
            t=time.perf_counter();ok=False;error=None;usage={}
            emit({'event':'http_start','stage':active.stage,'prompt_chars':len(prompt),'text_chars':len(text),
                  'text_sha256':hashlib.sha256(text.encode()).hexdigest()})
            try:
                if not hasattr(self.local,'client'):self.local.client=ChatClient(config['entities'])
                raw=self.local.client.complete(prompt,text);usage=raw.get('usage',{});ok=True;return raw
            except Exception as exc:error=type(exc).__name__;raise
            finally:
                emit({'event':'http_end','stage':active.stage,'seconds':time.perf_counter()-t,
                      'success':ok,'error_type':error,'prompt_tokens':usage.get('prompt_tokens',0),
                      'completion_tokens':usage.get('completion_tokens',0)})

    original_runner=workflow.StageRunner
    class MeasuredRunner(original_runner):
        """测量含缓存/校验/重试的整个阶段调用，HTTP时间另行记录。"""
        def ask(self,stage,prompt,payload,validator):
            """相邻层级事件存在包含关系，报告不将HTTP和阶段耗时重复相加。"""
            previous=getattr(active,'stage',None);active.stage=stage;t=time.perf_counter();ok=False
            try:
                value=super().ask(stage,prompt,payload,validator);ok=True;return value
            finally:
                emit({'event':'stage_end','stage':stage,'seconds':time.perf_counter()-t,'success':ok,
                      'http_attempts':getattr(self.local,'attempts',0),'cache_hit':ok and getattr(self.local,'attempts',0)==0})
                active.stage=previous
    workflow.StageRunner=MeasuredRunner

    def instrument(name):
        """包裹CPU与图步骤，记录与模型调用无关的构图时间。"""
        original=getattr(workflow,name)
        def measured(*items,**kwargs):
            """记录原函数运行时间，保持返回值和异常不变。"""
            t=time.perf_counter();ok=False
            try:
                value=original(*items,**kwargs);ok=True;return value
            finally:emit({'event':'cpu_step_end','stage':name,'seconds':time.perf_counter()-t,'success':ok})
        setattr(workflow,name,measured)
    for name in ['select_documents','aggregate','retrieve_candidates','canonicalize','build_semantic_graph','fuse_graphs','load_graph','save_graph']:instrument(name)

    def interrupted(signum,frame):
        """用户停止也保存已耗时，不写成完成。"""
        raise KeyboardInterrupt('Signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted)
    result=None;error=None;state='failed'
    log.info('START products=100 model=%s output=%s',config['entities']['model'],output)
    try:
        t=time.perf_counter();data=load_prepared(config['data']['output']);emit({'event':'cpu_step_end','stage':'load_prepared','seconds':time.perf_counter()-t,'success':True})
        docs,known=workflow.select_documents(data,config['entities'])
        assert len(docs)==100 and len({i for i,t in docs})==100
        sizes=[len(t) for i,t in docs];chunks=sum(sum(1 for _ in workflow.document_chunks(t,config['entities']['chunk_size'],config['entities']['overlap'])) for i,t in docs)
        atomic_json(run/'input_statistics.json',{'products':100,'train_unique_products':len(known),'document_chars_total':sum(sizes),
            'document_chars_min':min(sizes),'document_chars_mean':statistics.mean(sizes),'document_chars_median':statistics.median(sizes),
            'document_chars_max':max(sizes),'chunks':chunks,'selected_item_ids':[i for i,t in docs]})
        client=MeasuredClient(ChatClient(config['entities']))
        for index in range(1):
            started_attempt=time.perf_counter()
            try:
                log.info('WORKFLOW attempt=%d',index+1)
                result=workflow.run_workflow(data,config,client)
                attempts.append({'attempt':index+1,'success':True,'seconds':time.perf_counter()-started_attempt});break
            except workflow.RequestBudgetExceeded:raise
            except RuntimeError as exc:
                attempts.append({'attempt':index+1,'success':False,'seconds':time.perf_counter()-started_attempt,'error':str(exc)[:300]})
                log.exception('Workflow attempt failed; cached successes retained')
                if index==2:raise
                emit({'event':'retry_wait','stage':'workflow','seconds':30});time.sleep(30)
        state='complete' if result.get('complete') else 'partial'
        log.info('WORKFLOW RESULT %s',json.dumps(result,ensure_ascii=False))
    except KeyboardInterrupt as exc:state='interrupted';error=str(exc);log.warning('Interrupted')
    except Exception as exc:error=type(exc).__name__+': '+str(exc)[:500];log.exception('Experiment incomplete')
    finally:
        duration=time.perf_counter()-begin;finished=now();groups=defaultdict(list)
        for event in events:
            if event['event'] in {'stage_end','http_end','cpu_step_end'}:groups[(event['event'],event['stage'])].append(event)
        summary=[]
        for (kind,stage),rows in sorted(groups.items()):
            times=[r['seconds'] for r in rows]
            summary.append({'kind':kind,'stage':stage,'count':len(rows),'seconds_total':sum(times),
                'seconds_mean':statistics.mean(times),'seconds_p50':percentile(times,.5),'seconds_p95':percentile(times,.95),
                'successes':sum(r['success'] for r in rows),'cache_hits':sum(r.get('cache_hit',False) for r in rows),
                'prompt_tokens':sum(r.get('prompt_tokens',0) or 0 for r in rows),
                'completion_tokens':sum(r.get('completion_tokens',0) or 0 for r in rows)})
        semantic_events=[e for e in events if e['event']=='cpu_step_end' and e['stage']=='build_semantic_graph' and e['success']]
        report={'status':state,'completed':state=='complete','started':started,'finished':finished,'elapsed_seconds':duration,
            'elapsed_hours':duration/3600,'time_to_semantic_graph_seconds':semantic_events[-1]['elapsed_seconds'] if semantic_events else None,
            'stage_summary':summary,'workflow_attempts':attempts,'result':result,'error':error,
            'scope':'100 fixed training products; diagnostic graph over full catalog index, not all training products',
            'timing_notes':['wall time includes errors, retries, waits and graph serialization',
                'HTTP durations overlap under concurrency; never sum to estimate wall clock',
                'application cache cold at start; model server remains warm',
                'description review and conservative cross-product disambiguation are current engineering choices',
                'no official GraphRAG SDK or community search; custom paper-step pipeline']}
        # 与冻结串行实验对比，同样定义联合抽取区间；失败或部分完成如实标记。
        baseline_dir=root/'runs/entity_workflow_100_recovery_20261031'
        baseline=json.loads((baseline_dir/'timing_report.json').read_text())
        baseline_events=[json.loads(line) for line in (baseline_dir/'events.jsonl').read_text().splitlines()]
        def joint_wall(records):
            starts=[e['elapsed_seconds'] for e in records if e['event']=='http_start' and e.get('stage')=='joint']
            ends=[e['elapsed_seconds'] for e in records if e['event']=='stage_end' and e.get('stage')=='joint']
            return max(ends)-min(starts) if starts and ends else None
        old_joint=joint_wall(baseline_events);new_joint=joint_wall(events)
        extraction_file=output/'extraction_timing.json'
        extraction=json.loads(extraction_file.read_text()) if extraction_file.exists() else None
        report['joint_stage_wall_seconds']=new_joint
        report['comparison']={'baseline_run':str(baseline_dir),'baseline_status':baseline['status'],
            'baseline_total_seconds':baseline['elapsed_seconds'],'baseline_joint_wall_seconds':old_joint,
            'optimized_extraction':extraction,'joint_speedup':old_joint/new_joint if extraction and new_joint else None,
            'total_speedup':baseline['elapsed_seconds']/duration if result else None,
            'same_100_item_ids':config['entities']['item_ids']==json.loads((baseline_dir/'config.json').read_text())['entities']['item_ids'],
            'quality_note':'candidate strategy changed; speed does not prove alias recall/semantic accuracy; not controlled service load'}
        if result and (output/'disambiguation.json').exists():
            old_dis=json.loads((root/'data/entities/entity_workflow_100_recovery_20261031/disambiguation.json').read_text())
            new_dis=json.loads((output/'disambiguation.json').read_text());mapping=new_dis['mapping']
            positives=[d for d in old_dis['decisions'] if d['accepted']]
            comparable=[d for d in positives if d['left'] in mapping and d['right'] in mapping]
            report['comparison']['baseline_positive_decisions_on_shared_entities']=len(comparable)
            report['comparison']['baseline_positive_decisions_same_final_cluster']=sum(mapping[d['left']]==mapping[d['right']] for d in comparable)
            report['comparison']['baseline_completed_products']=baseline['result']['completed_products']
            report['comparison']['optimized_completed_products']=result['completed_products']
            report['comparison']['baseline_semantic_edges']=baseline['result']['semantic_edges']
            report['comparison']['optimized_semantic_edges']=result['semantic_edges']
        atomic_json(run/'comparison.json',report['comparison'])
        atomic_json(run/'timing_report.json',report);atomic_json(run/'status.json',{'status':state,'completed':state=='complete','started':started,'finished':finished,'elapsed_seconds':duration,'error':error})
        import csv
        with (run/'stage_summary.csv').open('w',encoding='utf-8-sig',newline='') as stream:
            fields=['kind','stage','count','seconds_total','seconds_mean','seconds_p50','seconds_p95','successes','cache_hits','prompt_tokens','completion_tokens']
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(summary)
        lines=['# 100件商品完整实体工作流计时实验','',f'状态：{state}',f"开始（北京时间）：{started['beijing']}",
            f"结束（北京时间）：{finished['beijing']}",f'总耗时：{duration:.1f}秒（{duration/3600:.3f}小时）','',
            '100件商品固定，使用新版论文五步流程；不是完整训练目录。没有将仅实体抽取时间当作全部流程时间。',
            '原始事件见events.jsonl；结构化结果见timing_report.json；分阶段CSV见stage_summary.csv。',
            '并发请求时间相互重叠，HTTP累计秒数不能当作墙钟时间。应用缓存为空，服务已加载且未清理前缀缓存。',
            '当前实现含额外描述审查、保守消歧；该耗时不是官方GraphRAG普遍性能。',
            '中断/失败/预算不足不视为100件全部完成。人工语义质量审核没有自动完成。','',
            '|计时类别|阶段|次数|总秒数|p50秒|p95秒|','|---|---|---:|---:|---:|---:|']
        lines += [f"|{r['kind']}|{r['stage']}|{r['count']}|{r['seconds_total']:.2f}|{r['seconds_p50']:.2f}|{r['seconds_p95']:.2f}|" for r in summary]
        lines+=['','## 串行对比','',json.dumps(report['comparison'],ensure_ascii=False,indent=2)]
        if result:lines+=['','## 流程结果','',json.dumps(result,ensure_ascii=False,indent=2)]
        if error:lines+=['','## 未完成原因','',error]
        (run/'REPORT.md').write_text('\n'.join(lines)+'\n')
        log.info('END status=%s elapsed_seconds=%.1f report=%s',state,duration,run/'REPORT.md')
    return 0 if state=='complete' else 1


if __name__=='__main__':raise SystemExit(main())
