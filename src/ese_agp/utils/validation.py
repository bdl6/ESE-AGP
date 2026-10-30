"""商品编号与整数输入的共用校验工具。"""
from numbers import Integral
from typing import Any

def _integer(value: Any, name: str, *, minimum: int) -> int:
    """校验非 bool 的整数及下界，并转成 Python int；同时接受 numbers.Integral 类型。"""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _catalog_item(value: Any, num_items: int) -> int:
    """检查编号在 0..N 范围内；0 可作为输入 OOV，但不会成为推荐候选。"""
    item_id = _integer(value, "item ID", minimum=0)
    if item_id > num_items:
        raise ValueError(f"item ID {item_id} is outside catalog 0..{num_items}")
    return item_id
