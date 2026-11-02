"""按论文步骤编排联合抽取、补充、描述、消歧和规范实体语义构图；非官方GraphRAG SDK。"""
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
from ese_agp.entities.disambiguation import candidates,validate_resolution
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
        self.statepath=self.folder/'request_state.json'
        self.state=json.loads(self.statepath.read_text()) if self.statepath.exists() else {'identity':identity,'attempts':0,'prompt_tokens':0,'completion_tokens':0}
        if self.state['identity']!=identity:raise ValueError('Workflow identity changed; use new output')
        if not isinstance(config['max_requests'],int) or config['max_requests']<1:raise ValueError('Positive max_requests required')

    def ask(self, stage, prompt, payload, validator):
        """成功阶段可恢复；原始输出和校验错误逐尝试保留，不修补残缺JSON。"""
        text=json.dumps(payload,ensure_ascii=False,sort_keys=True)
        key=hashlib.sha256(json.dumps([self.state['identity'],stage,prompt,text],ensure_ascii=False).encode()).hexdigest()
        path=self.folder/'stages'/(key+'.json')
        if path.exists():
            cached=json.loads(path.read_text())
            if cached.get('fingerprint')!=graph_fingerprint(cached):raise ValueError('Stage cache corrupted')
            if cached.get('status')=='success':return validator(cached['payload'])
        record={'stage':stage,'status':'failed','attempts':cached.get('attempts',[]) if path.exists() else []};feedback='';truncated=False
        retries=self.config.get('request_attempts',3)
        if not isinstance(retries,int) or retries<1:raise ValueError('request_attempts must be positive')
        for attempt in range(retries):
            if self.state['attempts']>=self.config['max_requests']:raise RequestBudgetExceeded('Cumulative workflow request limit reached')
            self.state['attempts']+=1;atomic_json(self.statepath,self.state)
            try:
                raw=self.client.complete(prompt+feedback,text)
                usage=raw.get('usage',{})
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
    known={i for ex in data.train for i in (*ex.input_item_ids,ex.target_item_id) if i}
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
    return results,traces


def run_workflow(data, config, client):
    """断点恢复完整步骤；失败时保留阶段缓存，绝不把半完成流程写成完整产物。"""
    cfg=config['entities'];output=Path(cfg['output']);output.mkdir(parents=True,exist_ok=True)
    lock=output/'workflow.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    started=time.time();status={'stage':'starting','complete':False}
    try:
        documents,known=select_documents(data,cfg)
        settings={k:cfg.get(k) for k in ['base_url','model','max_output_tokens','chunk_size','overlap','max_rounds','max_alias_pairs','alias_threshold','max_split_depth','min_chunk_chars','continue_on_error']}
        identity=hashlib.sha256(json.dumps([VERSION,data_fingerprint(data),documents,settings],ensure_ascii=False,sort_keys=True).encode()).hexdigest()
        runner=StageRunner(output,cfg,client,identity)
        rounds=cfg.get('max_rounds',3)
        if not isinstance(rounds,int) or rounds<1:raise ValueError('max_rounds must be positive')
        results=[];traces=[];failures=[];failed_items=set();item_progress={}
        def record_failure(error, **context):
            entry={'stage':error.stage,'error':str(error),'time':time.time(),**context}
            failures.append(entry);atomic_json(output/'failures.json',{'failures':failures})
        def progress(item,index,round_index):
            status.update(stage='joint_extraction',item_id=item,chunk=index,round=round_index,
                processed_products=len(item_progress),failed_products=len(failed_items),requested_products=len(documents))
            atomic_json(output/'status.json',status)
        # 先固化完整规范文档；每条证据的起止位置都能回到这些文本。
        atomic_json(output/'documents.json',{'identity':identity,'documents':[{'item_id':i,'text':t} for i,t in documents]})
        for item,text in documents:
            try:
                item_results,item_traces=extract_product(runner,item,text,cfg,progress)
                results.extend(item_results);traces.extend(item_traces);item_progress[item]={'status':'extracted'}
            except StageError as error:
                if not cfg.get('continue_on_error',True):raise
                failed_items.add(item);record_failure(error,item_id=item)
                item_progress[item]={'status':'failed','error':str(error)}
            atomic_json(output/'item_progress.json',{'items':item_progress,'requested_products':len(documents)})
        entities,relations=aggregate(results)
        atomic_json(output/'joint_chunks.json',{'identity':identity,'results':results,'traces':traces})
        status.update(stage='entity_descriptions',entities=len(entities),relations=len(relations));atomic_json(output/'status.json',status)
        for key,entity in sorted(entities.items()):
            if any(e['item_id'] in failed_items for e in entity['evidence']):continue
            try:entity['description']=describe(runner,entity,'describe_initial')
            except StageError as error:
                if not cfg.get('continue_on_error',True):raise
                items={e['item_id'] for e in entity['evidence']};failed_items.update(items)
                record_failure(error,entity_id=key,item_ids=sorted(items))
        # 描述阶段失败的商品整件排除，避免把不完整五步产物作为成功样本。
        entities={k:e for k,e in entities.items() if not any(v['item_id'] in failed_items for v in e['evidence'])}
        relations={k:r for k,r in relations.items() if r['head'] in entities and r['tail'] in entities}
        limit=cfg.get('max_alias_pairs',100)
        if not isinstance(limit,int) or limit<0:raise ValueError('max_alias_pairs must be nonnegative')
        threshold=cfg.get('alias_threshold',.9)
        if not isinstance(threshold,(int,float)) or not .5<=threshold<=1:raise ValueError('Invalid alias threshold')
        pairs,candidate_stats=candidates(entities,limit);decisions=[]
        status.update(stage='disambiguation',candidate_stats=candidate_stats);atomic_json(output/'status.json',status)
        for left,right in pairs:
            a,b=entities[left],entities[right]
            payload={'left':{k:a[k] for k in ['name','type','description']},'right':{k:b[k] for k in ['name','type','description']},
                'left_evidence':a['evidence'][:4],'right_evidence':b['evidence'][:4]}
            # 校验只允许本次真正给模型看的证据ID，不能引用未展示片段。
            shown_a={**a,'evidence':payload['left_evidence']};shown_b={**b,'evidence':payload['right_evidence']}
            try:decisions.append(runner.ask('resolve',RESOLVE_PROMPT,payload,lambda value:validate_resolution(value,shown_a,shown_b,threshold)))
            except StageError as error:
                if not cfg.get('continue_on_error',True):raise
                record_failure(error,left=left,right=right)
                # 缺少判断依据时不合并；缺失判断在报告中单独计数。
        registry,canonical_relations,mapping=canonicalize(entities,relations,decisions)
        for key,entity in registry.items():
            if len(entity['members'])>1:
                try:entity['description']=describe(runner,entity,'describe_canonical')
                except StageError as error:
                    if not cfg.get('continue_on_error',True):raise
                    items={e['item_id'] for e in entity['evidence']};failed_items.update(items)
                    record_failure(error,canonical_id=key,item_ids=sorted(items))
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
            'identity':identity,'scope':'full' if set(by_item)==known and not failures and not candidate_stats.get('unprocessed_pairs',0) else 'pilot_or_partial','completed_products':len(by_item),
            'requested_products':len(documents),'failed_products':len(failed_items),'pending_products':0,'stage_errors':len(failures),'registry_fingerprint':json.loads((output/'entity_registry.json').read_text())['fingerprint'],
            'records':[{'item_id':i,'status':'failed' if i in failed_items else 'success','entities':by_item.get(i,[])} for i,t in documents]}
        artifact['fingerprint']=graph_fingerprint(artifact);atomic_json(output/'entities.json',artifact)
        semantic=build_semantic_graph(data,output/'entities.json',config['graph'].get('top_k',20),allow_partial=artifact['scope']!='full')
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
            'semantic_edges':semantic['num_edges'],'fused_edges':fused['num_edges'],'usage':runner.state,'elapsed_seconds':time.time()-started,
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
