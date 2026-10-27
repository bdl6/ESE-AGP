from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    config_path = Path(path).resolve()
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or set(config) != {"data", "model", "training"}:
        raise ValueError("Config must contain exactly data, model and training")
    # Relative paths are anchored to the project's configs/ parent, not the shell cwd.
    root = config_path.parent.parent
    for key in ("input", "output"):
        value = Path(config["data"][key])
        config["data"][key] = str(value if value.is_absolute() else root / value)
    value = Path(config["training"]["run_dir"])
    config["training"]["run_dir"] = str(value if value.is_absolute() else root / value)
    return config
