"""通用JSON结果写入。"""
import json
from pathlib import Path

def write_json(path: Path, payload: dict) -> None:
    """创建父目录并用 UTF-8 写可读 JSON；拒绝 NaN/Infinity，避免生成无效报告。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
