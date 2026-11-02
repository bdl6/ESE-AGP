"""把小型交互 JSONL 转成监督样本和会话局部图，不依赖 PyTorch。

流程：按用户排序 → 固定24小时窗口 → 全局迭代过滤 → 全量保留商品编号
→ 随机会话划分 → 前缀与标签 → 批内局部图。此内存读取器仅用于小数据排错；
真实Amazon数据使用amazon.py的磁盘SQL实现。
时间戳单位为毫秒。编号 0 同时用于 padding/OOV，但保留未知交互的原始位置。
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence


# 所有窗口运算统一为毫秒，避免把秒级时间戳误当成毫秒。
DAY_MS = 24 * 60 * 60 * 1000


@dataclass(frozen=True)
class Interaction:
    """一条原始交互：原始用户标识、原始商品标识和毫秒时间戳。

    frozen=True 防止记录被随手改写；user_id 用于构造会话，不输入用户 embedding。
    """
    user_id: str
    item_id: str
    timestamp: int

    def __post_init__(self) -> None:
        """dataclass 构造后校验标识与时间戳；特别排除 Python 中也属于 int 的 bool。"""
        if not isinstance(self.user_id, str) or not self.user_id:
            raise ValueError("user_id must be a nonempty string")
        if not isinstance(self.item_id, str) or not self.item_id:
            raise ValueError("item_id must be a nonempty string")
        if isinstance(self.timestamp, bool) or not isinstance(self.timestamp, int):
            raise ValueError("timestamp must be an integer number of milliseconds")


@dataclass(frozen=True)
class SessionExample:
    """一条“根据前缀预测最后一件商品”的监督样本。

    例如原始 [A,B,A,C] 得到 input_item_ids=(A,B,A)、target_item_id=C。
    商品均已映射为整数；验证/测试可含未知编号 0。起止时间保留截断前的值。
    """
    # 已知商品编号 1..N；0 表示不在商品词表中的 OOV，序列不含预测目标。
    input_item_ids: tuple[int, ...]
    # 标签仍是商品 ID；交叉熵类别下标在训练器中转换为 target_item_id-1。
    target_item_id: int
    user_id: str
    session_start: int
    session_end: int
    # 定位样本用的稳定标识，不是模型的输入特征。
    session_id: str = ""


@dataclass
class PreparedData:
    """预处理产物：三份样本、统一商品词表映射、训练热门榜及可审计的规则/统计清单。"""
    train: list[SessionExample]
    valid: list[SessionExample]
    test: list[SessionExample]
    # 原始商品ID → 统一商品词表编号1..N，覆盖全部保留会话。
    catalog: dict[str, int]
    # 按训练频次降序排列的商品 ID 列表，不是频次数组；并列按 ID 升序。
    popularity: list[int]
    # 记录切分时间、过滤次数、OOV 数量、参数与数据指纹，供复现和检查泄漏。
    manifest: dict


@dataclass(frozen=True)
class LocalGraph:
    """纯 Python 表示的批内局部图；模型侧再把这些列表变成张量。

    不同会话的节点彼此独立，同一会话的重复商品共享节点；0 是共享虚拟节点。
    B=会话数，T=最长前缀长度（至少为 1），M=有效局部节点数，L=去重边数。
    """
    # [M+1]：局部节点索引 → 商品词表ID，第 0 项为虚拟节点。
    node_item_ids: list[int]
    # 两行长度均为 L；每列是一条来源局部节点→接收局部节点的有向边。
    edge_index: tuple[list[int], list[int]]
    # [B,T]：点击位置 → 批内局部节点索引。
    position_to_node: list[list[int]]
    # [B,T]：仅已知真实点击为 True，OOV 与补齐位置为 False。
    position_mask: list[list[bool]]
    # [B]：是否至少有一个有效历史商品，供评估/推理选择热门降级。
    has_known_history: list[bool]


@dataclass(frozen=True)
class _RawSession:
    """内部原始会话，保存用户、固定窗口编号以及排序后的全部事件。"""
    user_id: str
    window: int
    events: tuple[Interaction, ...]

    @property
    def start(self) -> int:
        """原始会话首次事件时间；不能用截断后的首次事件代替时间切分依据。"""
        return self.events[0].timestamp

    @property
    def end(self) -> int:
        """原始会话最后事件时间，也是该会话的预测目标发生时间。"""
        return self.events[-1].timestamp


def load_interactions_jsonl(path: str | Path) -> list[Interaction]:
    """读取每行含 user_id、item_id、timestamp 的小型 JSONL，返回交互列表。

    保留文件行顺序，用于时间并列时打破平局；坏行异常包含行号便于定位。
    此处返回完整列表，不是面向海量原始文件的流式预处理器。
    """
    result = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                result.append(Interaction(row["user_id"], row["item_id"], row["timestamp"]))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid interaction at line {line_number}: {exc}") from exc
    return result


def prepare(
    interactions: Iterable[Interaction],
    *,
    min_user_events: int = 3,
    min_item_events: int = 5,
    max_session_length: int = 20,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    split_seed: int = 42,
) -> PreparedData:
    """小数据参考路径：全局收敛过滤、全量词表、随机会话划分。

    max_session_length只限制历史长度，目标额外保留。验证集仅用于人工调试，
    正式Amazon论文路径使用9:1和空验证集；不按训练未见删除交互。
    """
    if min_user_events<1 or min_item_events<1:raise ValueError('frequency thresholds must be at least 1')
    if max_session_length<1:raise ValueError('history limit must be positive')
    if not (0<train_ratio<1 and 0<=valid_ratio<1-train_ratio):raise ValueError('ratios must leave train and test')
    if not isinstance(split_seed,int) or split_seed<0:raise ValueError('split_seed must be nonnegative')
    records = list(interactions)
    if any(not isinstance(record, Interaction) for record in records):
        raise TypeError("interactions must contain Interaction objects")
    if not records:
        raise ValueError("at least one interaction is required")

    # 1. 保留原始行号，时间相同的交互按原行序排列，结果可追溯。
    by_user: dict[str, list[tuple[int, Interaction]]] = defaultdict(list)
    # 对交互内容和顺序计算摘要；输入改变时可发现产物不再来自同一份数据。
    digest = hashlib.sha256()
    for original_row, record in enumerate(records):
        by_user[record.user_id].append((original_row, record))
        digest.update(json.dumps(
            [record.user_id, record.item_id, record.timestamp],
            ensure_ascii=False, separators=(",", ":"),
        ).encode("utf-8") + b"\n")

    # 2. 每个用户先按时间排序，再以该用户首次事件为固定窗口原点。
    raw_sessions = []
    for user_id in sorted(by_user):
        ordered = sorted(by_user[user_id], key=lambda pair: (pair[1].timestamp, pair[0]))
        # 固定窗口不是“连续两次点击超过 24 小时才断开”的空闲超时会话。
        anchor = ordered[0][1].timestamp
        windows: dict[int, list[Interaction]] = defaultdict(list)
        for _, record in ordered:
            # 整数除法得到窗口号：同一原点下 [0,24h)、[24h,48h) 分别成组。
            windows[(record.timestamp - anchor) // DAY_MS].append(record)
        raw_sessions.extend(
            _RawSession(user_id, window, tuple(events))
            for window, events in sorted(windows.items())
        )
    # 至少两次交互才有非空前缀和一个目标。此时尚未按频次过滤或截断。
    sessions = [session for session in raw_sessions if len(session.events) >= 2]
    if not sessions:
        raise ValueError("no session has both an input and a target")
    sessions.sort(key=lambda session: (session.end, session.start, session.user_id, session.window))
    # 3. 固定窗口内反复筛除，用户/商品频次使用全体当前保留会话。
    filter_rounds=[]
    while True:
        before=sum(len(x.events) for x in sessions)
        users=Counter(x.user_id for x in sessions for _ in x.events)
        active=[x for x in sessions if users[x.user_id]>=min_user_events]
        items=Counter(e.item_id for x in active for e in x.events)
        kept=[]
        for x in active:
            events=tuple(e for e in x.events if items[e.item_id]>=min_item_events)
            if len(events)>=2:kept.append(_RawSession(x.user_id,x.window,events))
        after=sum(len(x.events) for x in kept)
        filter_rounds.append({'before':before,'after':after,'converged':before==after})
        sessions=kept
        if before==after:break
    if not sessions:raise ValueError('No sessions survive global filtering')
    # 4. 截取历史+独立目标；全部保留会话一起建立词表，随后随机分配会话。
    selected={ (x.user_id,x.window):x.events[-(max_session_length+1):] for x in sessions }
    catalog={item:i+1 for i,item in enumerate(sorted({e.item_id for events in selected.values() for e in events}))}
    def session_id(x):
        return hashlib.sha256((x.user_id+':'+str(x.window)).encode()).hexdigest()
    ranked=sorted(sessions,key=lambda x:(hashlib.sha256((str(split_seed)+':'+session_id(x)).encode()).hexdigest(),x.user_id,x.window))
    train_count=int(len(ranked)*train_ratio);valid_count=int(len(ranked)*valid_ratio)
    split_by_key={(x.user_id,x.window):'train' if i<train_count else 'valid' if i<train_count+valid_count else 'test' for i,x in enumerate(ranked)}
    prepared={name:[] for name in ['train','valid','test']}
    for x in sorted(sessions,key=lambda x:(x.end,x.start,x.user_id,x.window)):
        recent=selected[x.user_id,x.window];split=split_by_key[x.user_id,x.window]
        prepared[split].append(SessionExample(tuple(catalog[e.item_id] for e in recent[:-1]),catalog[recent[-1].item_id],x.user_id,x.start,x.end,session_id(x)))

    # 6. 热门榜只统计最终保留训练样本的完整序列，包含它们的训练标签。
    train_counts = Counter(
        item
        for example in prepared["train"]
        for item in (*example.input_item_ids, example.target_item_id)
        if item != 0
    )
    popularity = sorted(catalog.values(), key=lambda item: (-train_counts[item], item))
    # 7. 保存规则和每步数量；这份清单可用于检查过滤损失、时间边界和 OOV。
    manifest={
        'protocol':'tiny-jsonl-global-iterative-random-v2','timestamp_unit':'milliseconds',
        'interactions_sha256':digest.hexdigest(),'window_ms':DAY_MS,'anchor':'first_event_per_user',
        'duplicates':'preserved; equal timestamps retain input row order','train_ratio':train_ratio,'valid_ratio':valid_ratio,
        'split_seed':split_seed,'max_history_length':max_session_length,'max_session_length_including_target':max_session_length+1,
        'min_user_events':min_user_events,'min_item_events':min_item_events,
        'frequency_filter':'global_until_convergence_before_split','filter_rounds':filter_rounds,
        'raw_interactions':len(records),'raw_sessions':len(raw_sessions),'catalog_scope':'all_retained_products',
        'split_examples':{split:len(value) for split,value in prepared.items()},
        'oov_targets':{split:0 for split in prepared},'catalog_size':len(catalog),
        'train_item_counts':{str(item):count for item,count in sorted(train_counts.items())}}
    return PreparedData(**prepared, catalog=catalog, popularity=popularity, manifest=manifest)


def prefix_to_local_graph(prefixes: Iterable[Sequence[int]]) -> LocalGraph:
    """把若干已编号的输入前缀变成一批互不连通的局部图。

    Args:
        prefixes: 每条为非负商品 ID 序列；0 是未知交互/补齐位置。
    Returns:
        LocalGraph，含节点商品 ID、[src,dst] 边、位置映射和有效位置 mask。

    例：[1,0,2] 无有效相邻边，不补出 1→2；[1,2,1] 只有两个有效节点，
    位置映射仍含三次点击；[1,1] 的真实自转移保留。重复边去重但不压缩点击位置。
    """
    rows = [list(prefix) for prefix in prefixes]
    for row in rows:
        if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in row):
            raise ValueError("prefix item IDs must be nonnegative integers")
    # 批内统一补齐到 T；即使全为空，也预留一列虚拟位置，避免零宽输入。
    width = max(1, max((len(row) for row in rows), default=0))
    # 节点索引 0 为虚拟零节点；真实局部节点从 1 开始累加。
    node_item_ids = [0]
    edges: set[tuple[int, int]] = set()
    position_to_node = []
    position_mask = []
    has_known_history = []
    for row in rows:
        # 每个会话重新建映射，保证不同会话之间不共用局部节点、不误连边。
        session_nodes: dict[int, int] = {}
        for item in row:
            if item and item not in session_nodes:
                session_nodes[item] = len(node_item_ids)
                node_item_ids.append(item)
        # 重复商品复用节点，但每次点击的位置都保留；OOV 映射到虚拟节点。
        positions = [session_nodes.get(item, 0) for item in row]
        # 只看原序列的相邻位置，且两端都已知才连边；不跨越 OOV。
        for source, destination in zip(positions, positions[1:]):
            if source and destination:
                edges.add((source, destination))
        mask = [item != 0 for item in row]
        padding = width - len(row)
        position_to_node.append(positions + [0] * padding)
        position_mask.append(mask + [False] * padding)
        has_known_history.append(any(mask))
    # 集合去重后排序，固定边顺序；真实重复点击产生的自转移不会被排除。
    ordered_edges = sorted(edges)
    return LocalGraph(
        node_item_ids=node_item_ids,
        edge_index=([edge[0] for edge in ordered_edges], [edge[1] for edge in ordered_edges]),
        position_to_node=position_to_node,
        position_mask=position_mask,
        has_known_history=has_known_history,
    )
