"""读取 YAML 配置，校验分区并统一数据和运行目录的路径基准。"""
from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    """返回含 data、model、training 三个分区的配置字典。

    参数 path 相对当前工作目录解析；配置内部的相对路径则以配置文件目录的
    上一级为基准。默认文件位于 configs/，所以内部路径相对项目根目录。
    """
    config_path = Path(path).resolve()
    with config_path.open(encoding="utf-8") as stream:
        # safe_load 只加载 YAML 数据，不构造任意 Python 对象。
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or set(config) != {"data", "model", "training"}:
        raise ValueError("Config must contain exactly data, model and training")
    # 以 configs/ 的上一级为路径基准，减少切换命令工作目录造成的路径错误。
    root = config_path.parent.parent
    for key in ("input", "reviews", "metadata", "output"):
        if key not in config["data"]:
            continue
        value = Path(config["data"][key])
        config["data"][key] = str(value if value.is_absolute() else root / value)
    value = Path(config["training"]["run_dir"])
    config["training"]["run_dir"] = str(value if value.is_absolute() else root / value)
    return config
