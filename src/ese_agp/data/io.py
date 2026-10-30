"""预处理产物加载、序列化及数据身份计算，不依赖训练器。"""
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from ese_agp.data.sessions import PreparedData, SessionExample

def prepared_payload(data: PreparedData) -> dict:
    """把预处理 dataclass 转成可写 JSON 的字典，并附上数据格式版本号。"""
    return {"schema_version": 1, **asdict(data)}


def data_fingerprint(data: PreparedData) -> str:
    """对完整预处理产物生成 SHA256 指纹，绑定样本、目录编号、切分和规则清单。

    排序字典键并去掉非必要空白，使序列化格式变化不轻易改变指纹。
    样本序列的顺序仍然有意义，改变目录或数据后不应沿用旧 checkpoint。
    """
    # 真实数据身份由已校验的文件清单给出，避免展开全量 Arrow 样本重新序列化。
    if hasattr(data, "fingerprint"):
        return data.fingerprint
    raw = json.dumps(prepared_payload(data), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_prepared(path: str | Path) -> PreparedData:
    """读取 prepare 命令的 JSON，恢复三份 SessionExample 列表和 PreparedData。

    JSON 数组会读成 list，这里把输入前缀还原为 tuple，保持数据结构一致。
    """
    if Path(path).is_dir():
        from ese_agp.data.artifacts import PreparedArtifact
        return PreparedArtifact(path)
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    # 数据结构升级时应显式处理版本，不能静默把不兼容的文件当旧格式读取。
    if payload.pop("schema_version", None) != 1:
        raise ValueError("Unsupported prepared dataset schema")
    for split in ("train", "valid", "test"):
        payload[split] = [SessionExample(**{**row, "input_item_ids": tuple(row["input_item_ids"])}) for row in payload[split]]
    return PreparedData(**payload)
