"""恢复本轮失败商品，复用成功缓存，并生成独立的重试图与报告。"""
from pathlib import Path
import json, subprocess, sys, time
from ese_agp.utils.config import load_config
from ese_agp.data.io import load_prepared
from ese_agp.graphs.semantic import build_semantic_graph
from ese_agp.graphs.fusion import fuse_graphs
from ese_agp.graphs.behavior import load_graph,save_graph


def main():
    """同步完成抽取与构图；模型断线时保留真实暂停状态，不伪报成功。"""
    p=Path(__file__).resolve().parents[2];start=time.time()
    cfg=load_config(p/'configs/day04_entities_retry.yaml')
    state={'status':'retrying','started_unix':start}
    (p/'runs/day04_retry_status.json').write_text(json.dumps(state))
    result=subprocess.run([sys.executable,'-u','-m','ese_agp','extract','--config','configs/day04_entities_retry.yaml'],cwd=p)
    result.check_returncode()
    a=json.loads((Path(cfg['entities']['output'])/'entities.json').read_text());data=load_prepared(cfg['data']['output'])
    semantic=build_semantic_graph(data,Path(cfg['entities']['output'])/'entities.json',allow_partial=True)
    fused=fuse_graphs(load_graph(cfg['graph']['behavior_path'],data),semantic,.5,allow_partial=True)
    save_graph(semantic,cfg['graph']['semantic_path']);save_graph(fused,cfg['graph']['path'])
    state.update(status='paused_service_unavailable' if a.get('service_unavailable') else 'retry_finished',
        successful=a['completed_products'],failed=a['failed_products'],pending=a['pending_products'],
        cumulative_attempts=a['usage']['attempts'],semantic_edges=semantic['num_edges'],fused_edges=fused['num_edges'],
        elapsed_seconds=time.time()-start,full_training_ready=False)
    # 验证原来5件成功缓存没有被重请求或改写。
    before=p/'data/entities/extracted_pilot_v2_before_retry_20261030/cache';preserved=0
    for f in before.glob('*.json'):
        if json.loads(f.read_text())['status']=='success':
            assert f.read_bytes()==(Path(cfg['entities']['output'])/'cache'/f.name).read_bytes();preserved+=1
    state['old_success_cache_preserved']=preserved
    (p/'runs/day04_retry_status.json').write_text(json.dumps(state,indent=2))
    lines=['# 实体试抽取重试结果','',f"成功{state['successful']}件，失败{state['failed']}件，未处理{state['pending']}件。",
        f"复用原成功缓存{preserved}件，累计请求尝试{state['cumulative_attempts']}。",
        f"语义边{state['semantic_edges']}条，beta融合边{state['fused_edges']}条。",'',
        '首轮故障：生成服务的vLLM EngineCore在sample_tokens RPC调用超时后退出；确切底层原因仍未确定。',
        '已恢复同一模型，使用eager模式排除编译/图捕获路径变量；未改模型版本。',
        '格式围栏解析、校验错误反馈和连接故障熔断已修复，71项测试通过。',
        '本次仍为100件pilot图，不是全量实体实验；人工质量审核尚未完成，不进入正式训练。']
    (p/'docs/DAY4_RETRY_RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(state),flush=True)


if __name__=='__main__':main()
