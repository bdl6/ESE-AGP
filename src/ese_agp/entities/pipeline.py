"""实体抽取执行：单进程锁、累计请求限额、逐商品缓存、证据校验及遗漏检查。"""
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.error
from ese_agp.entities.extraction import PROMPT, PROMPT_VERSION, text_chunks, cache_key, validate_entities
from ese_agp.data.io import data_fingerprint
from ese_agp.graphs.behavior import graph_fingerprint

PIPELINE_VERSION='two-pass-evidence-v1'


def atomic_json(path, value):
    """同目录临时文件原子替换，避免中断留下半份缓存。"""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,allow_nan=False))
    temporary.replace(path)


def decode_response(raw, text, rejected=None):
    """拒绝截断、缺失内容和无原文证据的结果，成功空实体仍为合法结果。"""
    choice=raw['choices'][0]
    if choice.get('finish_reason')!='stop': raise ValueError('Incomplete model response')
    content=choice['message']['content'].strip()
    # 只移除完整包裹的Markdown代码围栏，仍使用严格JSON解析，不猜补残缺内容。
    if content.startswith('```') and content.endswith('```'):
        lines=content.splitlines()
        if lines[0].strip().lower() in {'```','```json'}:content='\n'.join(lines[1:-1]).strip()
    payload=json.loads(content)
    if rejected is None:return validate_entities(payload,text)
    if not isinstance(payload,dict) or set(payload)!={'entities'} or not isinstance(payload['entities'],list):
        raise ValueError('Invalid entity schema')
    valid=[]
    for entity in payload['entities']:
        try:
            checked=validate_entities({'entities':[entity]},text)
            valid.extend({k:v for k,v in item.items() if k!='normalized_name'} for item in checked)
        except (ValueError,TypeError,KeyError) as error:
            rejected.append({'entity':entity,'reason':str(error)[:240]})
    if payload['entities'] and not valid:raise ValueError('All returned entities lack valid original evidence')
    return validate_entities({'entities':valid},text)


def extract_requests(data, config, client):
    """只接受训练商品；max_requests为本输出目录跨重启累计HTTP尝试上限。"""
    limit=config['max_requests']
    if not isinstance(limit,int) or isinstance(limit,bool) or limit<1:
        raise ValueError('Explicit positive max_requests quota required')
    if not config.get('model') or not config.get('base_url'): raise ValueError('Model/base URL required')
    output=Path(config['output']);output.mkdir(parents=True,exist_ok=True)
    lock=output/'extract.lock'
    descriptor=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(descriptor,str(os.getpid()).encode());os.close(descriptor)
    try:
        requests=Path(config['requests'])
        pilot=json.loads((requests.parent/'manifest.json').read_text())
        fingerprint=data_fingerprint(data)
        if pilot['data_fingerprint']!=fingerprint: raise ValueError('Pilot dataset fingerprint mismatch')
        rows=[json.loads(line) for line in requests.read_text().splitlines() if line.strip()]
        known={i for ex in data.train for i in (*ex.input_item_ids,ex.target_item_id) if i}
        ids=[row['item_id'] for row in rows]
        if len(ids)!=len(set(ids)) or any(i not in known for i in ids): raise ValueError('Expected unique training products')
        identity=hashlib.sha256(json.dumps([fingerprint,hashlib.sha256(requests.read_bytes()).hexdigest(),
                 config['base_url'],config['model'],PROMPT_VERSION,PROMPT,PIPELINE_VERSION,
                 config.get('max_output_tokens',1024)],sort_keys=True).encode()).hexdigest()
        statepath=output/'state.json'
        state=json.loads(statepath.read_text()) if statepath.exists() else {'identity':identity,'attempts':0,'input_tokens':0,'output_tokens':0}
        if state['identity']!=identity: raise ValueError('Extraction identity changed; use new output directory')
        results=[];budget_exhausted=False;fatal_service_error=False;service_unavailable=False
        for row in rows:
            text=row['text'];item=row['item_id']
            if hashlib.sha256(text.encode()).hexdigest()!=row['text_sha256']: raise ValueError('Request text hash mismatch')
            key=cache_key(text,config['model'],config['base_url']+PIPELINE_VERSION)
            path=output/'cache'/f'{item}_{key}.json'
            if path.exists():
                record=json.loads(path.read_text())
                if record.get('status')=='success' and record.get('identity')==identity and record.get('fingerprint')==graph_fingerprint(record):
                    results.append(record);continue
            record={'item_id':item,'text_sha256':row['text_sha256'],'identity':identity,'status':'failed','entities':[],'responses':[],'rejected_entities':[]}
            unique={};failed=False
            for chunk in text_chunks(text):
                for stage in range(2):
                    prompt=PROMPT if stage==0 else PROMPT+' 执行遗漏检查，仅补充未提取的实体；已提取：'+json.dumps(list(unique),ensure_ascii=False)
                    succeeded=False;feedback=""
                    for attempt in range(3):
                        if state['attempts']>=limit:
                            budget_exhausted=True;break
                        # 先记账再请求；崩溃时最多多算一次，不能超出累计请求次数上限。
                        state['attempts']+=1;atomic_json(statepath,state)
                        try:
                            raw=client.complete(prompt+feedback,chunk)
                            usage=raw.get('usage',{})
                            state['input_tokens']+=int(usage.get('prompt_tokens',0))
                            state['output_tokens']+=int(usage.get('completion_tokens',0));atomic_json(statepath,state)
                            record['responses'].append({'stage':stage,'raw':raw})
                            entities=decode_response(raw,chunk,rejected=record['rejected_entities'])
                            for entity in entities: unique[entity['type']+'::'+entity['normalized_name']]=entity
                            succeeded=True;break
                        except urllib.error.HTTPError as error:
                            record['error']='HTTP '+str(error.code)
                            if error.code not in {408,429,500,502,503,504}:
                                fatal_service_error=True;break
                        except (TimeoutError,urllib.error.URLError,ConnectionError) as error:
                            record['error']=type(error).__name__
                            record['service_errno']=getattr(getattr(error,'reason',error),'errno',None)
                            if attempt==2:service_unavailable=True
                        except (ValueError,KeyError,TypeError,IndexError,AttributeError) as error:
                            # 不保存异常原文，避免服务错误中包含凭据或敏感响应。
                            record['error']=type(error).__name__
                            previous=raw.get('choices',[{}])[0].get('message',{}).get('content','') if isinstance(raw,dict) and raw.get('choices') else ''
                            feedback=' 上次输出未通过校验：'+str(error)[:240]+'。重新输出完整JSON，不带Markdown围栏；删除原文不存在的证据，不推断CD格式或人物身份。上次输出：'+str(previous)[:4000]
                        if attempt<2:time.sleep(min(2**attempt,4))
                    if not succeeded: failed=True;break
                if failed:break
            if not failed:
                record['status']='success';record['entities']=list(unique.values());record.pop('error',None)
            record['response_sha256']=graph_fingerprint({'responses':record['responses']})
            record['fingerprint']=graph_fingerprint(record)
            atomic_json(path,record);results.append(record)
            if budget_exhausted or fatal_service_error or service_unavailable:break
        # 熔断时也把后续已成功缓存计入汇总，避免误报丢失已有成果。
        done={r['item_id'] for r in results}
        for row in rows:
            if row['item_id'] in done:continue
            key=cache_key(row['text'],config['model'],config['base_url']+PIPELINE_VERSION)
            cached=output/'cache'/f"{row['item_id']}_{key}.json"
            if cached.exists():
                record=json.loads(cached.read_text())
                if record.get('status')=='success' and record.get('identity')==identity and record.get('fingerprint')==graph_fingerprint(record):results.append(record)
        success=sum(r['status']=='success' for r in results)
        artifact={'schema_version':1,'kind':'item_entities','data_fingerprint':fingerprint,'identity':identity,
                  'scope':'full' if set(ids)==known and success==len(known) else 'pilot_or_partial',
                  'requested_products':len(rows),'completed_products':success,'failed_products':len(results)-success,
                  'pending_products':len(rows)-len(results),'budget_exhausted':budget_exhausted,'fatal_service_error':fatal_service_error,'service_unavailable':service_unavailable,'usage':state,
                  'nonempty_products':sum(bool(r['entities']) for r in results if r['status']=='success'),
                  'accepted_entity_count':sum(len(r['entities']) for r in results if r['status']=='success'),
                  'records':results,'prompt_version':PROMPT_VERSION,'pipeline_version':PIPELINE_VERSION}
        artifact['fingerprint']=graph_fingerprint(artifact)
        atomic_json(output/'entities.json',artifact)
        return {k:v for k,v in artifact.items() if k!='records'}
    finally:lock.unlink(missing_ok=True)
