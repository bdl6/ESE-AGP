"""等待已启动的真实pilot结束，再自动生成显式partial语义图与融合诊断报告。"""
import json
import time
from pathlib import Path
from ese_agp.data.io import load_prepared
from ese_agp.utils.config import load_config
from ese_agp.graphs.semantic import build_semantic_graph
from ese_agp.graphs.fusion import fuse_graphs
from ese_agp.graphs.behavior import load_graph, save_graph


def main():
    """不发模型请求，不把100件试抽取包装成全量实验。"""
    root=Path(__file__).resolve().parents[2]
    cfg=load_config(root/'configs/day04_entities.yaml')
    artifact=Path(cfg['entities']['output'])/'entities.json'
    pid=json.loads((root/'runs/day04_extract_process.json').read_text())['pid']
    deadline=time.monotonic()+3600
    while not artifact.exists():
        if not Path('/proc') .joinpath(str(pid)).exists():raise RuntimeError('Extraction exited without final artifact')
        if time.monotonic()>deadline:raise TimeoutError('Extraction still running; inspect logs before continuing')
        time.sleep(5)
    entities=json.loads(artifact.read_text());data=load_prepared(cfg['data']['output'])
    semantic=build_semantic_graph(data,artifact,allow_partial=True)
    behavior=load_graph(cfg['graph']['behavior_path'],data)
    fused=fuse_graphs(behavior,semantic,cfg['graph']['beta'],allow_partial=True)
    save_graph(semantic,cfg['graph']['semantic_path']);save_graph(fused,cfg['graph']['path'])
    report={k:v for k,v in entities.items() if k!='records'}
    report.update(semantic_edges=semantic['num_edges'],fused_edges=fused['num_edges'],
                  human_reviews_completed=0,full_experiment_ready=False,status='pilot_graphs_complete_requires_quality_review')
    (root/'runs/day04_pilot_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    lines=['# Day04 实体试抽取与语义图结果','',
       '真实服务为本机Qwen3-30B-A3B-Instruct-2507；仅100件训练商品试抽取。',
       f"成功商品：{entities['completed_products']}；失败：{entities['failed_products']}；未处理：{entities['pending_products']}。",
       f"累计尝试：{entities['usage']['attempts']}；输入token：{entities['usage']['input_tokens']}；输出token：{entities['usage']['output_tokens']}。",
       f"语义边：{semantic['num_edges']}；beta融合后边：{fused['num_edges']}。",'',
       '这是明确标记的pilot/partial诊断图，训练加载器默认拒绝使用；不能当作全量语义增强结果。',
       '失败结果未当作成功空实体；原始响应保留在实体缓存中。尚未完成人工30条审核，不启动全量抽取。',
       '后续需检查角色误判、缺失原文证据、长文本失败，再决定提示词/文本格式修订及扩量安排。']
    (root/'docs/DAY4_PILOT_RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
