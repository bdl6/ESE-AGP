"""CPU/CUDA运行设备选择。"""
import torch

def choose_device(requested: str) -> torch.device:
    """把 auto/cpu/cuda 配置转换为 torch.device。

    auto 按 CUDA 实际可用性选择；显式要求 cuda 但不可用时直接报错，避免误用 CPU。
    """
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu or cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable in this venv")
    return torch.device("cuda" if requested == "auto" and torch.cuda.is_available() else "cpu" if requested == "auto" else requested)
