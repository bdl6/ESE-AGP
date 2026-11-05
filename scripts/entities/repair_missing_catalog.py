"""补齐空实体及报错商品：标题可溯源兜底，属性文本低并发重试，单件失败不停止。"""
import hashlib,json,subprocess,sys,threading,time,traceback
from pathlib import Path
from ese_agp.entities.joint import JOINT_PROMPT,title_entity_result
from ese_agp.entities import workflow_optimized as w
from ese_agp.entities.client import ChatClient
from ese_agp.entities.pipeline import atomic_json
from ese_agp.utils.config import load_config


def main():
    root=Path(__file__).resolve().parents[2]
    source=root/'data/entities/catalog_v4_64_20261102'
    baseline=root/'data/entities/catalog_v4_recovered_20261103/entities.json'
    out=root/'data/entities/catalog_v4_missing_repair_20261103';out.mkdir(exist_ok=True)
    documents=json.loads((source/'documents.json').read_text())
    rows=json.loads(baseline.read_text())['records'];wanted={r['item_id'] for r in rows if not r['entities']}
    docs={d['item_id']:d['text'] for d in documents['documents'] if d['item_id'] in wanted}
    failures=json.loads((source/'failures.json').read_text())['failures']
    failed_joint={f['item_id'] for f in failures if f['stage']=='joint'}
    seeds=[title_entity_result(i,t) for i,t in sorted(docs.items())]
    if any(x is None for x in seeds):raise ValueError('Missing title requires manual handling')
    retry=[]
    for item,text in sorted(docs.items()):
        rest='\n'.join(text.splitlines()[1:])
        rich=rest.replace('features:','').replace('description:','').replace('details:','').strip()
        if rich or item in failed_joint:retry.append((item,text))
    cfg=load_config(root/'configs/entity_workflow_catalog_v4_64.yaml')['entities']
    cfg.update(output=str(out),max_requests=10000,concurrency=4,cache_sources=[],request_attempts=4,json_mode=True)
    identity=hashlib.sha256((documents['identity']+'-repair571-v1').encode()).hexdigest()
    local=threading.local()
    class Client:
        def complete(self,prompt,text):
            if not hasattr(local,'client'):local.client=ChatClient(cfg)
            return local.client.complete(prompt,text)
    runner=w.StageRunner(out,cfg,Client(),identity)
    # 此提示词变化进入阶段缓存键；不能把旧的空输出误当成此次修复结果。
    w.JOINT_PROMPT=JOINT_PROMPT+'\n首轮必须识别title字段中的商品标题，按product输出。Label/Manufacturer如有值应检查是否存在品牌实体。JSON字符串中的双引号与反斜线必须正确转义。仅根据原文，禁止推断标题未表明的艺人。'
    state={'status':'running','source_identity':documents['identity'],'target_products':len(wanted),
        'title_seed_products':len(seeds),'llm_retry_products':len(retry),'concurrency':4,
        'processed':0,'retry_success':0,'retry_errors':[],'started':time.time()}
    patch={'source_identity':documents['identity'],'results':list(seeds)}
    atomic_json(out/'patches.json',patch);atomic_json(out/'repair_status.json',state)
    def task(pair):
        item,text=pair
        try:return item,w.extract_product(runner,item,text,cfg,lambda *args:None)[0],None
        except w.StageError as error:return item,[],str(error)
    for item,results,error in w.parallel_results(task,retry,4,preserve_order=False):
        state['processed']+=1
        if error:state['retry_errors'].append({'item_id':item,'error':error})
        else:
            state['retry_success']+=1;patch['results'].extend(results)
        atomic_json(out/'patches.json',patch);atomic_json(out/'repair_status.json',state)
    state['status']='building_graphs';atomic_json(out/'repair_status.json',state)
    command=[sys.executable,str(root/'scripts/entities/recover_catalog.py'),
        '--source',str(source),'--patches',str(out/'patches.json'),
        '--output',str(root/'data/entities/catalog_v4_repaired_20261103'),
        '--graph-output',str(root/'data/graphs/paper_drop_session_v4_repaired_20261103'),
        '--data',str(root/'data/processed/amazon_cds_paper_drop_session_v4'),
        '--behavior',str(root/'data/graphs/paper_drop_session_v4/behavior.json')]
    subprocess.run(command,cwd=root,check=True)
    state.update(status='completed_with_pending_workflow_stages',elapsed_seconds=time.time()-state['started'],usage=runner.state)
    atomic_json(out/'repair_status.json',state);print(json.dumps(state,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
