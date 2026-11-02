"""收敛过滤验收：级联删除、固定窗口、空结果及独立参考实现。"""
from collections import Counter
import random
import duckdb
import pytest
from ese_agp.data.amazon import iterative_session_filter


def reference(rows,user_min,item_min):
    """纯Python参考，直接按三类频次约束反复筛除，独立于SQL执行顺序。"""
    while True:
        users=Counter(r[1] for r in rows);items=Counter(r[2] for r in rows)
        sessions=Counter((r[1],r[3]) for r in rows)
        bad_sessions={(r[1],r[3]) for r in rows if items[r[2]]<item_min}
        after=[r for r in rows if users[r[1]]>=user_min and (r[1],r[3]) not in bad_sessions and sessions[(r[1],r[3])]>=2]
        if after==rows:return rows
        rows=after


def run_filter(rows,user_min=3,item_min=2):
    """原始窗口和行ID固定，返回收敛集合及完整轮次。"""
    con=duckdb.connect()
    con.execute('CREATE TABLE events(original_row INTEGER,user_id VARCHAR,parent_asin VARCHAR,window_id INTEGER,timestamp BIGINT)')
    if rows:con.executemany('INSERT INTO events VALUES (?,?,?,?,?)',rows)
    rounds=iterative_session_filter(con,user_min,item_min)
    result=con.execute('SELECT * FROM filtered_events ORDER BY original_row').fetchall()
    assert rounds[-1]['converged'];assert rounds[-1]['removed_interactions']==0
    assert all(r['removed_interactions']>0 for r in rounds[:-1])
    con.close();return result,rounds


def test_cascade_needs_multiple_rounds_and_preserves_window_ids():
    """孤立会话使商品退化，进一步使其他会话和用户失效，必须迭代多轮。"""
    spec=[('stable1','AAA',7),('stable2','AAA',9),('bridge','ABC',3),
          ('left','B',0),('left','DD',1),('right','C',0),('right','EE',1)]
    rows=[]
    for user,items,window in spec:
        for item in items:rows.append((len(rows),user,item,window,1600000000000+len(rows)))
    result,rounds=run_filter(rows)
    assert len(rounds)==3
    assert [r['interactions'] for r in rounds]==[10,6,6]
    assert {r[1] for r in result}=={'stable1','stable2'}
    assert {(r[1],r[3]) for r in result}=={('stable1',7),('stable2',9)}
    assert result==reference(rows,3,2)
    # 同一结果再次过滤立即收敛，具有幂等性。
    again,second_rounds=run_filter(result)
    assert again==result;assert len(second_rounds)==1


@pytest.mark.parametrize('seed',range(5))
def test_matches_independent_fixed_point_reference(seed):
    """随机小图与同时应用阈值的参考不动点一致，允许最终为空。"""
    rng=random.Random(seed)
    rows=[(i,'u'+str(rng.randrange(10)),'p'+str(rng.randrange(8)),rng.randrange(3),1000+i) for i in range(80)]
    result,rounds=run_filter(rows,4,5)
    assert result==reference(rows,4,5)
    assert {r[0] for r in result}<=set(range(80))


def test_empty_result_converges_without_infinite_loop():
    """全部被删除时仍正常终止；公开预处理随后负责拒绝空训练集。"""
    result,rounds=run_filter([(0,'u','rare',5,1000)])
    assert result==[];assert rounds[-1]['interactions']==0;assert len(rounds)==2


@pytest.mark.parametrize('pattern',['XAA','AXA','AAX'])
def test_any_low_frequency_position_discards_entire_session(pattern):
    """低频商品在开头、中间、结尾时均整段删除，不构造新的A→A边。"""
    spec=[('stable1','AAA',0),('stable2','AAA',0),('affected',pattern,0),('affected','AAA',1)]
    rows=[]
    for user,items,window in spec:
        for item in items:rows.append((len(rows),user,item,window,1000+len(rows)))
    result,rounds=run_filter(rows,3,2)
    assert not any(r[1]=='affected' and r[3]==0 for r in result)
    assert sum(r[1]=='affected' and r[3]==1 for r in result)==3
    assert rounds[0]['sessions_dropped_for_low_frequency_item']==1
    assert result==reference(rows,3,2)


def test_whole_session_drop_triggers_item_and_user_cascade():
    """删掉含X的会话后，B变低频，继续删掉其会话，再检查用户阈值。"""
    spec=[('stable1','AAAAA',0),('stable2','AAAAA',0),('left','ABX',0),
          ('right','ABB',0),('right','AA',1)]
    rows=[]
    for user,items,window in spec:
        for item in items:rows.append((len(rows),user,item,window,1000+len(rows)))
    result,rounds=run_filter(rows,3,3)
    assert {r[1] for r in result}=={'stable1','stable2'}
    assert sum(r['sessions_dropped_for_low_frequency_item'] for r in rounds)==2
    assert len(rounds)>=3
