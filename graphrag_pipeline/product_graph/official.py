"""调用固定版本官方GraphRAG API；只运行实体关系索引，不运行问答/社区/向量流程。"""
import asyncio,importlib.metadata,logging,time,os
from pathlib import Path
import yaml
from .common import load_manifest,read_json,write_json,file_sha


WORKFLOWS=['create_base_text_units','create_final_documents','extract_graph']


def configuration(run,concurrency=64,gleanings=2,api_base='http://127.0.0.1:18888/v1'):
    """所有SDK写入目录均位于新run；保留官方抽取与描述归纳提示词。"""
    run=Path(run).resolve()
    if not 1<=concurrency<=64 or not 0<=gleanings<=3:raise ValueError('Invalid concurrency/gleanings')
    if api_base!='http://127.0.0.1:18888/v1':raise ValueError('This local profile only permits configured localhost service')
    return {'completion_models':{'default_completion_model':{'model_provider':'openai',
        'model':'Qwen3-30B-A3B-Instruct-2507','api_base':api_base,'api_key':'local-no-key',
        'call_args':{'temperature':0,'max_tokens':4096,'timeout':180},
        'retry':{'type':'exponential_backoff','max_retries':3,'base_delay':2,'max_delay':20}}},
        'embedding_models':{},'concurrent_requests':concurrency,'async_mode':'asyncio',
        'input_storage':{'type':'file','base_dir':str(run)},
        'output_storage':{'type':'file','base_dir':str(run/'sdk_output')},
        'update_output_storage':{'type':'file','base_dir':str(run/'sdk_update')},
        'cache':{'type':'json','storage':{'type':'file','base_dir':str(run/'sdk_cache')}},
        'reporting':{'type':'file','base_dir':str(run/'sdk_logs')},
        'vector_store':{'type':'lancedb','db_uri':str(run/'unused_vector_store')},
        # 每块都加文档ID，避免不同商品的相同段落产生相同文本块哈希。
        'chunking':{'type':'tokens','size':600,'overlap':80,'encoding_model':'cl100k_base','prepend_metadata':['id']},
        'extract_graph':{'completion_model_id':'default_completion_model','max_gleanings':gleanings,
            'prompt':str(run/'prompts/extract_graph.txt'),
            'entity_types':['product','brand','artist','genre','category','material','function','audience','scene','style','format']},
        'summarize_descriptions':{'completion_model_id':'default_completion_model','max_length':60,'prompt':str(run/'prompts/summarize_descriptions.txt')},
        'workflows':WORKFLOWS}


def index(run,concurrency=64,gleanings=2,resume=False):
    """SDK可能把单块异常写日志后继续；额外收集ERROR日志，不能把部分失败标为完整。"""
    # 使用LiteLLM随包成本表，避免本地模型启动时访问无关的远端价格表。
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP","True")
    import pandas as pd
    from graphrag.api import build_index
    from graphrag.config.models.graph_rag_config import GraphRagConfig
    from graphrag.callbacks.noop_workflow_callbacks import NoopWorkflowCallbacks
    run=Path(run).resolve();load_manifest(run)
    if importlib.metadata.version('graphrag')!='3.3.0':raise RuntimeError('Requires pinned GraphRAG 3.3.0')
    for name in ['extract_graph.txt','summarize_descriptions.txt']:
        source=Path(__file__).resolve().parents[1]/'prompts'/name;target=run/'prompts'/name
        if target.exists() and target.read_text()!=source.read_text():raise ValueError('Prompt changed; use a new run directory')
        target.parent.mkdir(exist_ok=True);target.write_text(source.read_text())
    config=configuration(run,concurrency,gleanings);settings=run/'settings.yaml'
    if settings.exists():
        if not resume:raise FileExistsError('Index exists; use --resume for identical settings')
        if yaml.safe_load(settings.read_text())!=config:raise ValueError('Settings changed; use a new run directory')
    else:settings.write_text(yaml.safe_dump(config,sort_keys=False,allow_unicode=True))
    checked=GraphRagConfig.model_validate(config)
    errors=[];begin=time.monotonic();state={'status':'running','sdk_version':'3.3.0','workflows':WORKFLOWS,
           'prompt_sha256':{f.name:file_sha(f) for f in (run/'prompts').glob('*.txt')}}
    write_json(run/'index_status.json',state)
    class ErrorCapture(logging.Handler):
        def emit(self,record):
            if record.levelno>=logging.ERROR:
                detail=getattr(record,'details',{})
                errors.append({'logger':record.name,'message':record.getMessage()[:500],
                    'text_unit_id':detail.get('source_id') if isinstance(detail,dict) else None})
    handler=ErrorCapture()
    class Progress(NoopWorkflowCallbacks):
        def pipeline_start(self,names):logging.getLogger('graphrag').addHandler(handler)
        def workflow_start(self,name,instance):
            state['stage']=name;write_json(run/'index_status.json',state)
    try:
        documents=pd.read_parquet(run/'documents.parquet')
        results=asyncio.run(build_index(config=checked,input_documents=documents,callbacks=[Progress()]))
        failed=[{'workflow':r.workflow,'error':str(r.error)} for r in results if r.error is not None]
        state.update(status='partial' if failed or errors else 'complete',workflow_errors=failed,logged_errors=errors)
        if failed or errors:raise RuntimeError('GraphRAG reported errors; inspect index_status.json')
        return {'status':state['status'],'seconds':time.monotonic()-begin,'workflows':WORKFLOWS}
    except Exception as error:
        state.update(status=state['status'] if state['status']=='partial' else 'failed',error=str(error)[:800]);raise
    finally:
        logging.getLogger('graphrag').removeHandler(handler)
        state['elapsed_seconds']=time.monotonic()-begin;write_json(run/'index_status.json',state)
