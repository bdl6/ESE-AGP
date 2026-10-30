"""神经推荐模型的批量评估，复用统一排名指标。"""
import torch
from torch.nn import functional as F
from ese_agp.data.batching import collate_examples
from ese_agp.training.metrics import evaluate_rankings

@torch.no_grad()
def evaluate_model(model, examples, popularity, device, batch_size=64, k=20):
    """评估神经模型并返回全样本/已知目标的 HR、MRR、OOV 比例等。

    Args:
        model: LocalSessionRecommender，参数已放到 device。
        examples: 验证或测试样本，允许未知目标和全未知历史。
        popularity: 只从训练集得到的热门商品 ID 排名，供无有效历史时降级。
        device: 批数据张量所在设备。
        batch_size: 每次前向处理的会话数。
        k: 排名截断位置，默认 20；候选不足 k 时返回全部候选。
    Returns:
        指标字典；额外的 model_known_history_ce 仅统计“目标已知且历史有效”的样本。

    装饰器 no_grad 关闭梯度记录，model.eval 关闭训练态 dropout，二者作用不同。
    全未知历史使用热门排名，未知目标在整体 HR/MRR 的分母中保留并计 miss。
    """
    model.eval()
    rankings, targets = [], []
    total_loss, known_count = 0.0, 0
    for start in range(0, len(examples), batch_size):
        rows = examples[start:start + batch_size]
        batch = collate_examples(rows, device)
        logits = model(batch)
        # CE 只诊断神经模型实际适用的样本；热门降级样本与未知标签均不混入该损失。
        known = (batch.target_item_ids > 0) & batch.has_known_history
        if known.any():
            # 商品编号 1..N 转为类别下标 0..N-1；先累计样本 loss 总和，再统一除以数量。
            total_loss += F.cross_entropy(logits[known], batch.target_item_ids[known] - 1, reduction="sum").item()
            known_count += int(known.sum())
        # 候选列原本按商品 ID 升序，stable=True 使分数并列时保持 ID 升序；+1 还原商品 ID。
        ordered = (torch.argsort(logits, dim=1, descending=True, stable=True)[:, :k] + 1).cpu().tolist()
        for row, has_history in zip(ordered, batch.has_known_history.cpu().tolist()):
            # 全未知历史即使能算出 logits 也没有有效兴趣信号，最终排名使用训练热门榜。
            rankings.append(row[:k] if has_history else popularity[:k])
        targets.extend(ex.target_item_id for ex in rows)
    # 交给与基线共用的评估器，确保候选、未知目标分母和指标定义一致。
    result = evaluate_rankings(rankings, targets, k=k)
    result["hr@1"] = evaluate_rankings(rankings, targets, k=1)["hr@1"]
    result["model_known_history_ce"] = total_loss / known_count if known_count else None
    result["num_model_loss_examples"] = known_count
    return result
