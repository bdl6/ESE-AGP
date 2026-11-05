"""独立优化实验：保留五步及原提示词，加入线程安全并发和候选筛选；不替换正式CLI。"""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import threading
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.error
import pyarrow.parquet as pq
from ese_agp.data.io import data_fingerprint
from ese_agp.entities.pipeline import atomic_json
from ese_agp.entities.joint import VERSION,JOINT_PROMPT,DESCRIPTION_PROMPT,RESOLVE_PROMPT,DESCRIPTION_REVIEW_PROMPT,validate_description_review,parse_json,validate_joint,validate_description
from ese_agp.entities.aggregation import aggregate,canonicalize
from ese_agp.entities.disambiguation import validate_resolution
from ese_agp.entities.optimized_candidates import candidates as retrieve_candidates
from ese_agp.graphs.behavior import graph_fingerprint,save_graph,load_graph
from ese_agp.graphs.semantic import build_semantic_graph
from ese_agp.graphs.fusion import fuse_graphs


class RequestBudgetExceeded(RuntimeError):
    """累计调用上限已用完，保留缓存后停止。"""


class StageError(RuntimeError):
    """单个业务阶段重试耗尽；允许隔离，不能吞掉预算或程序错误。"""
    def __init__(self, stage, message, truncated=False):
        super().__init__('Workflow stage failed: '+stage+'; '+message)
        self.stage=stage;self.truncated=truncated


class StageRunner:
    """各阶段的请求缓存与累计记账；运行级互斥由主工作流持有。"""
    def __init__(self, folder, config, client, identity):
        """不发请求；校验输出目录身份，防止跨数据/模型混用。"""
        self.folder=Path(folder);self.config=config;self.client=client
        self.state_lock=threading.RLock();self.key_locks={};self.local=threading.local()
        self.statepath=self.folder/'request_state.json'
        self.state=json.loads(self.statepath.read_text()) if self.statepath.exists() else {'identity':identity,'attempts':0,'prompt_tokens':0,'completion_tokens':0}
        if self.state['identity']!=identity:raise ValueError('Workflow identity changed; use new output')
        if not isinstance(config['max_requests'],int) or config['max_requests']<1:raise ValueError('Positive max_requests required')

    def ask(self, stage, prompt, payload, validator):
        """同一缓存键串行，独立键并发；预算预留和token记账持锁，网络请求不持锁。"""
        # 只缩短判断理由，不减少实体/关系/证据字段；提示词进入缓存键。
        if self.config.get('concise_outputs',False) and (stage=='resolve' or stage.endswith('_review')):
            prompt+='\n输出紧凑JSON，不输出解释段落。reason最多20个汉字或12个英文单词；证据ID和其他字段仍须完整。'
        key=hashlib.sha256(json.dumps([stage,prompt,payload],ensure_ascii=False,sort_keys=True).encode()).hexdigest()
        with self.state_lock:lock=self.key_locks.setdefault(key,threading.RLock())
        self.local.attempts=0
        with lock:return self._ask(stage,prompt,payload,validator)

    def _ask(self, stage, prompt, payload, validator):
        """成功阶段可恢复；原始输出和校验错误逐尝试保留，不修补残缺JSON。"""
        text=json.dumps(payload,ensure_ascii=False,sort_keys=True)
        key=hashlib.sha256(json.dumps([self.state['identity'],stage,prompt,text],ensure_ascii=False).encode()).hexdigest()
        path=self.folder/'stages'/(key+'.json')
        if path.exists():
            cached=json.loads(path.read_text())
            if cached.get('fingerprint')!=graph_fingerprint(cached):raise ValueError('Stage cache corrupted')
            if cached.get('status')=='success':return validator(cached['payload'])
        # 只读复用相同数据/文本/模型配置的成功缓存；具体请求还须完整prompt和payload相等。
        for source in self.config.get('cache_sources',[]):
            source_key=hashlib.sha256(json.dumps([source['identity'],stage,prompt,text],ensure_ascii=False).encode()).hexdigest()
            source_path=Path(source['folder'])/'stages'/(source_key+'.json')
            if source_path.exists():
                value=json.loads(source_path.read_text())
                if value.get('fingerprint')!=graph_fingerprint(value):raise ValueError('Source stage cache corrupted')
                if value.get('status')=='success':
                    result=validator(value['payload'])
                    with self.state_lock:
                        self.state['reused_stage_hits']=self.state.get('reused_stage_hits',0)+1
                    return result
        record={'stage':stage,'status':'failed','attempts':cached.get('attempts',[]) if path.exists() else []};feedback='';truncated=False
        retries=self.config.get('request_attempts',3)
        if not isinstance(retries,int) or retries<1:raise ValueError('request_attempts must be positive')
        for attempt in range(retries):
            with self.state_lock:
                if self.state['attempts']>=self.config['max_requests']:raise RequestBudgetExceeded('Cumulative workflow request limit reached')
                self.state['attempts']+=1;self.local.attempts+=1;atomic_json(self.statepath,self.state)
            try:
                raw=self.client.complete(prompt+feedback,text)
                usage=raw.get('usage',{})
                with self.state_lock:
                    self.state['prompt_tokens']+=int(usage.get('prompt_tokens',0));self.state['completion_tokens']+=int(usage.get('completion_tokens',0));atomic_json(self.statepath,self.state)
                entry={'raw':raw};record['attempts'].append(entry)
                if raw.get('choices',[{}])[0].get('finish_reason')=='length':
                    truncated=True;record['error']='Model response truncated';break
                payload_out=parse_json(raw);validated=validator(payload_out)
                record.update(status='success',payload=payload_out);record['fingerprint']=graph_fingerprint(record);atomic_json(path,record)
                return validated
            except urllib.error.HTTPError as error:
                record['error']='HTTP '+str(error.code)
                if error.code in {401,403,404}:raise RuntimeError('Model service configuration/authentication error: HTTP '+str(error.code)) from error
                if error.code not in {408,429,500,502,503,504}:break
            except (urllib.error.URLError,TimeoutError,ConnectionError) as error:
                record['error']=type(error).__name__
                if attempt==retries-1:break
            except (ValueError,KeyError,IndexError,TypeError,AttributeError) as error:
                record['error']=str(error)[:300]
                feedback=' 上一轮未通过校验：'+str(error)[:200]+'。重新输出符合模式的完整JSON，证据复制原文，不补充无依据内容。'
            record['fingerprint']=graph_fingerprint(record);atomic_json(path,record)
            if attempt<retries-1:time.sleep(min(2**attempt,30))
        record['fingerprint']=graph_fingerprint(record);atomic_json(path,record)
        raise StageError(stage,record.get('error','unknown'),truncated)


def product_document(row):
    """只展开商品白名单字段；把details从JSON字符串还原，避免双重转义干扰证据。"""
    parts=[]
    for key in ['title','features','description','details']:
        value=row.get(key)
        if key=='details' and isinstance(value,str):
            try:value=json.loads(value)
            except json.JSONDecodeError:pass
        if isinstance(value,dict):
            # 推荐实体抽取不输入日期、尺寸、重量、型号、停产标记等无关管理字段。
            allowed={'label','manufacturer','brand','artist','artists','composer','composers','performer','performers',
                     'conductor','genre','genres','format','material','style','color','colour','target audience','item form'}
            selected={k:v for k,v in value.items() if str(k).strip().casefold() in allowed}
            parts.append(key+':\n'+'\n'.join(str(k)+': '+str(v) for k,v in sorted(selected.items())))
        elif isinstance(value,list):parts.append(key+':\n'+'\n'.join(str(v) for v in value))
        elif value:parts.append(key+': '+str(value))
    return '\n'.join(parts)


def document_chunks(text, size=1600, overlap=200):
    """处理完整文档，不先截掉6000字符；保留绝对起点，最后一个块不重复起尾。"""
    if not isinstance(size,int) or size<1 or not 0<=overlap<size:raise ValueError('Invalid chunk settings')
    start=0
    while start<len(text):
        yield start,text[start:start+size]
        if start+size>=len(text):break
        start+=size-overlap


def select_documents(data, config):
    """只读训练中实际出现的商品；显式item_ids或固定哈希选择可复现样例。"""
    scope=config.get('item_scope','train')
    if scope not in {'train','catalog'}:raise ValueError('item_scope must be train or catalog')
    known=set(data.catalog.values()) if scope=='catalog' else {i for ex in data.train for i in (*ex.input_item_ids,ex.target_item_id) if i}
    if 'item_ids' in config:
        ids=config['item_ids']
        if not ids or len(ids)!=len(set(ids)) or any(i not in known for i in ids):raise ValueError('Requested items must be unique training products')
    else:
        limit=config.get('limit',100)
        if not isinstance(limit,int) or limit<1:raise ValueError('limit must be positive')
        ids=sorted(known,key=lambda i:hashlib.sha256(str(i).encode()).hexdigest())[:limit]
    wanted=set(ids);found={}
    for batch in pq.ParquetFile(data.directory/'products.parquet').iter_batches(batch_size=2048):
        for row in batch.to_pylist():
            if row['item_id'] in wanted:found[row['item_id']]=product_document(row)
    if set(found)!=wanted:raise ValueError('Training product metadata missing')
    return [(i,found[i]) for i in ids],known


def describe(runner, entity, stage):
    """跨块分层归纳，随后审查描述的原文支持，保留修订与理由。"""
    evidence=sorted(entity['evidence'],key=lambda e:(e['item_id'],e['start']))
    if not evidence:raise ValueError('Cannot describe entity without evidence')
    def reviewed(payload,allowed,suffix):
        """先归纳，再由独立提示词检查无依据断言；所有原始响应均缓存。"""
        draft=runner.ask(stage+suffix,DESCRIPTION_PROMPT,payload,lambda value:validate_description(value,allowed))
        review_payload={'name':entity['name'],'description':draft['description'],'evidence':allowed}
        review=runner.ask(stage+suffix+'_review',DESCRIPTION_REVIEW_PROMPT,review_payload,lambda value:validate_description_review(value,allowed))
        return {'description':review['description'],'evidence_ids':review['evidence_ids'],'draft':draft,
                'review_supported':review['supported'],'review_reason':review['reason']}
    summaries=[]
    for start in range(0,len(evidence),8):
        group=evidence[start:start+8]
        summaries.append(reviewed({'name':entity['name'],'type':entity['type'],'evidence':group},group,''))
    level=0
    while len(summaries)>1:
        combined=[]
        for start in range(0,len(summaries),8):
            group=summaries[start:start+8];ids={i for summary in group for i in summary['evidence_ids']}
            allowed=[e for e in evidence if e['evidence_id'] in ids]
            payload={'name':entity['name'],'type':entity['type'],'source_summaries':group,'available_evidence_ids':sorted(ids)}
            combined.append(reviewed(payload,allowed,'_reduce_'+str(level)))
        summaries=combined;level+=1
    return summaries[0]


def extract_product(runner, item, text, cfg, progress):
    """商品内任何子块最终失败则整件隔离；拆块后证据仍使用原文绝对偏移。"""
    results=[];traces=[]
    def extract(chunk, offset, index, depth=0):
        previous={};seen_relations=set();previous_relations={};local=[];local_traces=[]
        try:
            for round_index in range(cfg.get('max_rounds',3)):
                progress(item,index,round_index)
                payload={'item_id':item,'chunk_id':index,'round':round_index,'text':chunk,
                    'known_entities':[{'name':e['name'],'type':e['type']} for e in previous.values()],
                    'known_relations':list(previous_relations.values())}
                current=runner.ask('joint',JOINT_PROMPT,payload,lambda value:validate_joint(value,chunk,item,index,offset,previous))
                current.update(item_id=item,chunk_id=index,round=round_index);local.append(current)
                new_entities={e['type']+'::'+e['normalized_name']:e for e in current['entities']}
                new_relations={r['relation_id'] for r in current['relations']}
                additions=len(set(new_entities)-set(previous))+len(new_relations-seen_relations)
                previous.update(new_entities);seen_relations.update(new_relations)
                id_to_name={entity['entity_id']:entity['name'] for entity in previous.values()}
                for relation in current['relations']:
                    previous_relations[relation['relation_id']]={'entity_1':id_to_name[relation['head']],
                        'entity_2':id_to_name[relation['tail']],'relation':relation['relation']}
                local_traces.append({'item_id':item,'chunk':index,'round':round_index,'new_records':additions,
                    'rejected':len(current['rejected']),'split_depth':depth,'offset':offset})
                if additions==0:break
        except StageError as error:
            if not error.truncated or depth>=cfg.get('max_split_depth',3) or len(chunk)<2*cfg.get('min_chunk_chars',200):raise
            # 父块的半成品全部舍弃；子块分别补充抽取，避免混用父块的已知实体。
            middle=len(chunk)//2;overlap=min(50,len(chunk)//8)
            extract(chunk[:middle+overlap],offset,str(index)+'.0',depth+1)
            extract(chunk[middle-overlap:],offset+middle-overlap,str(index)+'.1',depth+1)
            return
        results.extend(local);traces.extend(local_traces)
    for index,(offset,chunk) in enumerate(document_chunks(text,cfg.get('chunk_size',1600),cfg.get('overlap',200))):
        extract(chunk,offset,index)
    if not any(r['entities'] for r in results):
        # 商品标题是元数据已有事实；空抽取时保留product，不推断其他属性或关系。
        from ese_agp.entities.joint import title_entity_result
        fallback=title_entity_result(item,text)
        if fallback is not None:results.append(fallback)
    return results,traces


def parallel_results(function, items, workers, preserve_order=True):
    """完成即补位；按序输出模式设有界重排缓冲，不按序模式不受慢任务阻塞。

    默认保留旧接口顺序。工作流用完成顺序接收以立即保存进度，最后显式排序归并。
    网络请求最多workers个；默认有序模式的待处理+重排缓冲最多2*workers。
    """
    if not isinstance(workers,int) or workers<1:raise ValueError('workers must be positive')
    iterator=enumerate(items);pending={};ready={};next_result=0;exhausted=False
    pool=ThreadPoolExecutor(max_workers=workers)
    try:
        while True:
            while not exhausted and len(pending)<workers and (not preserve_order or len(pending)+len(ready)<2*workers):
                try:index,item=next(iterator)
                except StopIteration:exhausted=True;break
                pending[pool.submit(function,item)]=index
            if not pending:
                if preserve_order:
                    while next_result in ready:yield ready.pop(next_result);next_result+=1
                break
            done,_=wait(pending,return_when=FIRST_COMPLETED)
            # 同一批完成结果按原索引处理，正式产物另行稳定排序。
            for future in sorted(done,key=lambda f:pending[f]):
                index=pending.pop(future);value=future.result()
                if preserve_order:ready[index]=value
                else:yield value
            if preserve_order:
                while next_result in ready:yield ready.pop(next_result);next_result+=1
    finally:
        for future in pending:future.cancel()
        pool.shutdown(wait=True,cancel_futures=True)


def candidates(entities,limit):
    """默认候选配置；run_workflow内使用实际配置覆盖。"""
    return retrieve_candidates(entities,limit)


def run_workflow(data, config, client):
    """断点恢复完整步骤；失败时保留阶段缓存，绝不把半完成流程写成完整产物。"""
    cfg=config['entities'];output=Path(cfg['output']);output.mkdir(parents=True,exist_ok=True)
    lock=output/'workflow.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    started=time.time();status={'stage':'starting','complete':False};progress_lock=threading.RLock()
    workers=cfg.get('concurrency',4)
    if not isinstance(workers,int) or not 1<=workers<=64:
        lock.unlink(missing_ok=True);raise ValueError('concurrency must be between 1 and 64')
    try:
        documents,known=select_documents(data,cfg)
        settings={k:cfg.get(k) for k in ['base_url','model','max_output_tokens','chunk_size','overlap','max_rounds','max_alias_pairs','alias_threshold','max_split_depth','min_chunk_chars','continue_on_error','concurrency','candidate_top_k','candidate_min_similarity','concise_outputs','item_scope']}
        identity=hashlib.sha256(json.dumps([VERSION+'-optimized-v4-title-grounded-fallback',data_fingerprint(data),documents,settings],ensure_ascii=False,sort_keys=True).encode()).hexdigest()
        runner=StageRunner(output,cfg,client,identity)
        rounds=cfg.get('max_rounds',3)
        if not isinstance(rounds,int) or rounds<1:raise ValueError('max_rounds must be positive')
        results=[];traces=[];failures=[];failed_items=set();item_progress={}
        def record_failure(error, **context):
            entry={'stage':error.stage,'error':str(error),'time':time.time(),**context}
            failures.append(entry);atomic_json(output/'failures.json',{'failures':failures})
        def progress(item,index,round_index):
            with progress_lock:
                status.update(stage='joint_extraction',item_id=item,chunk=index,round=round_index,
                    processed_products=len(item_progress),failed_products=len(failed_items),requested_products=len(documents),concurrency=workers)
                atomic_json(output/'status.json',status)
        # 先固化完整规范文档；每条证据的起止位置都能回到这些文本。
        atomic_json(output/'documents.json',{'identity':identity,'documents':[{'item_id':i,'text':t} for i,t in documents]})
        def extract_task(document):
            item,text=document
            try:return item,extract_product(runner,item,text,cfg,progress),None
            except StageError as error:return item,None,error
        extraction_started=time.perf_counter()
        for item,value,error in parallel_results(extract_task,documents,workers,preserve_order=False):
            with progress_lock:
                if error:
                    if not cfg.get('continue_on_error',True):raise error
                    failed_items.add(item);record_failure(error,item_id=item)
                    item_progress[item]={'status':'failed','error':str(error)}
                else:
                    item_results,item_traces=value;results.extend(item_results);traces.extend(item_traces)
                    item_progress[item]={'status':'extracted'}
                    atomic_json(output/'products'/(str(item)+'.json'),{'item_id':item,'results':item_results,'traces':item_traces})
                atomic_json(output/'item_progress.json',{'items':item_progress,'requested_products':len(documents)})
        atomic_json(output/'extraction_timing.json',{'elapsed_seconds':time.perf_counter()-extraction_started,
            'products_requested':len(documents),'products_extracted':len(documents)-len(failed_items),
            'failed_products':len(failed_items),'concurrency':workers,'complete':not failed_items})
        # 完成顺序用于实时进度；跨商品结果固定顺序后再聚合，避免并发改变产物顺序。
        results.sort(key=lambda r:(r['item_id'],str(r['chunk_id']),r['round']))
        traces.sort(key=lambda r:(r['item_id'],str(r['chunk']),r['round']))
        entities,relations=aggregate(results)
        atomic_json(output/'joint_chunks.json',{'identity':identity,'results':results,'traces':traces})
        if cfg.get('extract_only',False):
            status={'stage':'extraction_complete' if not failed_items else 'extraction_partial',
                'complete':False,'extraction_complete':not failed_items,'requested_products':len(documents),
                'extracted_products':len(documents)-len(failed_items),'failed_products':len(failed_items),
                'entities':len(entities),'relations':len(relations),'usage':runner.state,
                'elapsed_seconds':time.time()-started,'identity':identity,
                'remaining_stages':['entity_descriptions','disambiguation','canonical_descriptions','graphs']}
            atomic_json(output/'status.json',status);return status
        status.update(stage='entity_descriptions',entities=len(entities),relations=len(relations));atomic_json(output/'status.json',status)
        def describe_task(pair):
            key,entity=pair
            try:return key,describe(runner,entity,'describe_initial'),None
            except StageError as error:return key,None,error
        for key,value,error in parallel_results(describe_task,sorted(entities.items()),workers,preserve_order=False):
            if error:
                if not cfg.get('continue_on_error',True):raise error
                items={e['item_id'] for e in entities[key]['evidence']}
                # 描述失败只影响此实体的描述状态，保留已有原文实体链接。
                entities[key]['description']=None
                entities[key]['description_status']='pending'
                record_failure(error,entity_id=key,item_ids=sorted(items))
            else:entities[key]['description']=value
        # 仅排除抽取本身失败的商品；描述失败不得扩大为整件商品失败。
        entities={k:e for k,e in entities.items() if not any(v['item_id'] in failed_items for v in e['evidence'])}
        relations={k:r for k,r in relations.items() if r['head'] in entities and r['tail'] in entities}
        limit=cfg.get('max_alias_pairs',100)
        if not isinstance(limit,int) or limit<0:raise ValueError('max_alias_pairs must be nonnegative')
        threshold=cfg.get('alias_threshold',.9)
        if not isinstance(threshold,(int,float)) or not .5<=threshold<=1:raise ValueError('Invalid alias threshold')
        pairs,candidate_stats=retrieve_candidates(entities,limit,cfg.get('candidate_top_k',8),cfg.get('candidate_min_similarity',.65));decisions=[]
        status.update(stage='disambiguation',candidate_stats=candidate_stats);atomic_json(output/'status.json',status)
        def resolve_task(pair):
            left,right=pair;a,b=entities[left],entities[right]
            payload={'left':{k:a[k] for k in ['name','type','description']},'right':{k:b[k] for k in ['name','type','description']},
                'left_evidence':a['evidence'][:4],'right_evidence':b['evidence'][:4]}
            shown_a={**a,'evidence':payload['left_evidence']};shown_b={**b,'evidence':payload['right_evidence']}
            try:return pair,runner.ask('resolve',RESOLVE_PROMPT,payload,lambda value:validate_resolution(value,shown_a,shown_b,threshold)),None
            except StageError as error:return pair,None,error
        for pair,value,error in parallel_results(resolve_task,pairs,workers,preserve_order=False):
            if error:
                if not cfg.get('continue_on_error',True):raise error
                record_failure(error,left=pair[0],right=pair[1])
            else:decisions.append(value)
            status.update(resolved_pairs=len(decisions),resolve_tasks_processed=len(decisions)+sum(e['stage']=='resolve' for e in failures),failed_products=len(failed_items))
            atomic_json(output/'status.json',status)
        decisions.sort(key=lambda d:(d['left'],d['right']))
        registry,canonical_relations,mapping=canonicalize(entities,relations,decisions)
        def canonical_task(pair):
            key,entity=pair
            try:return key,describe(runner,entity,'describe_canonical'),None
            except StageError as error:return key,None,error
        merged=sorted((k,e) for k,e in registry.items() if len(e['members'])>1)
        status.update(stage='canonical_descriptions',merged_entities=len(merged));atomic_json(output/'status.json',status)
        for key,value,error in parallel_results(canonical_task,merged,workers,preserve_order=False):
            if error:
                if not cfg.get('continue_on_error',True):raise error
                items={e['item_id'] for e in registry[key]['evidence']}
                # 不把继承的单个成员描述伪装成合并描述；成员描述仍在descriptions中。
                registry[key]['description']=None
                registry[key]['description_status']='pending'
                record_failure(error,canonical_id=key,item_ids=sorted(items))
            else:registry[key]['description']=value
        for name,body in [('entity_registry',{'entities':list(registry.values())}),('relations',{'relations':list(canonical_relations.values())}),
                          ('disambiguation',{'decisions':decisions,'candidate_stats':candidate_stats,'mapping':mapping})]:
            body.update(identity=identity,data_fingerprint=data_fingerprint(data));body['fingerprint']=graph_fingerprint(body);atomic_json(output/(name+'.json'),body)
        by_item={i:[] for i,t in documents if i not in failed_items}
        for key,entity in sorted(registry.items()):
            for item in sorted({e['item_id'] for e in entity['evidence']}):
                if item not in by_item:continue
                by_item[item].append({'canonical_id':key,'type':entity['type'],'name':entity['name'],'normalized_name':entity['normalized_name'],
                    'description':entity['description'],'evidence':[e for e in entity['evidence'] if e['item_id']==item]})
        artifact={'schema_version':1,'kind':'item_entities','workflow_version':VERSION,'data_fingerprint':data_fingerprint(data),
            'identity':identity,'item_scope':cfg.get('item_scope','train'),'scope':'full' if set(by_item)==known and not failures and not candidate_stats.get('unprocessed_pairs',0) else 'pilot_or_partial','completed_products':len(by_item),
            'requested_products':len(documents),'failed_products':len(failed_items),'pending_products':0,'stage_errors':len(failures),'registry_fingerprint':json.loads((output/'entity_registry.json').read_text())['fingerprint'],
            'records':[{'item_id':i,'status':'failed' if i in failed_items else 'success','entities':by_item.get(i,[])} for i,t in documents]}
        artifact['fingerprint']=graph_fingerprint(artifact);atomic_json(output/'entities.json',artifact)
        semantic=None;fused=None
        if cfg.get('build_graphs',True):
            semantic=build_semantic_graph(data,output/'entities.json',config['graph'].get('top_k',20),allow_partial=artifact['scope']!='full',max_candidate_pairs=config['graph'].get('max_candidate_pairs',50000000))
            fused=fuse_graphs(load_graph(config['graph']['behavior_path'],data),semantic,config['graph'].get('beta',.5),allow_partial=artifact['scope']!='full')
            for graph,path in [(semantic,config['graph']['semantic_path']),(fused,config['graph']['path'])]:
                if Path(path).exists():
                    existing=load_graph(path,data,allow_partial=True)
                    if existing['fingerprint']!=graph['fingerprint']:raise ValueError('Refuse overwriting different graph')
                else:save_graph(graph,path)
        status={'stage':'partial' if failures or candidate_stats.get('unprocessed_pairs',0) else 'complete',
            'complete':not failures and not candidate_stats.get('unprocessed_pairs',0),'scope':artifact['scope'],'products':len(documents),
            'requested_products':len(documents),'completed_products':len(by_item),'failed_products':len(failed_items),
            'stage_errors':len(failures),'failures':failures,'entities':len(registry),
            'relations':len(canonical_relations),'accepted_alias_merges':sum(d['accepted'] for d in decisions),'alias_candidate_stats':candidate_stats,
            'semantic_edges':semantic['num_edges'] if semantic else None,'fused_edges':fused['num_edges'] if fused else None,
            'graphs_built':semantic is not None,'usage':runner.state,'elapsed_seconds':time.time()-started,
            'human_review_completed':False,'official_graphrag_sdk':False,'identity':identity}
        atomic_json(output/'item_progress.json',{'items':{i:{'status':'failed' if i in failed_items else 'success'} for i,t in documents},
            'requested_products':len(documents),'completed_products':len(by_item),'failed_products':len(failed_items),'stage_errors':len(failures)})
        atomic_json(output/'failures.json',{'failures':failures})
        atomic_json(output/'status.json',status)
        return status
    except Exception as error:
        status.update(stage='failed',error_type=type(error).__name__,error=str(error)[:300],complete=False)
        atomic_json(output/'status.json',status)
        raise
    finally:lock.unlink(missing_ok=True)
