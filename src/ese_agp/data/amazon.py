"""Amazon Reviews 2023 正式预处理：磁盘SQL，统一全局收敛过滤、全量词表及随机会话协议。"""
from pathlib import Path
import hashlib
import json
import tempfile
import time

import duckdb

from ese_agp.data.artifacts import sha256_file, save_json
from ese_agp.data.features import parse_price, fit_item_features


def _literal(value):
    """只用于 SQL 字面量转义，路径/文本不拼入 shell，也不能变成 SQL 标识符。"""
    return "'" + str(value).replace("'", "''") + "'"


def _safe_details(value):
    """商品 details 仅保留描述属性，排除评分、评论数、销量排名/购买关联等反馈键。"""
    if not value:
        return '{}'
    obj = json.loads(value) if isinstance(value, str) else value
    def clean(node):
        """递归清洗嵌套属性，避免反馈键从子字典重新进入商品文本。"""
        if isinstance(node, dict):
            return {k: clean(v) for k, v in node.items()
                    if not any(term in str(k).lower() for term in ('review', 'rating', 'rank', 'bought'))}
        if isinstance(node, list):
            return [clean(x) for x in node]
        return node
    return json.dumps(clean(obj), ensure_ascii=False)


def iterative_session_filter(con, min_user_events, min_item_events, on_round=None):
    """固定原始window_id，单调删除交互直到用户/商品/会话阈值同时满足。

    低频商品所在会话整段删除，不压缩拼接剩余事件；不重置窗口、不合并会话。
    频次按交互条数计，重复交互保留。
    收敛发生在截断和随机切分之前。每轮新集合都是旧集合子集，条数不变即集合不变。
    """
    con.execute('CREATE TABLE filter_current AS SELECT * FROM events')
    rounds=[]
    while True:
        before=con.execute('SELECT count(*) FROM filter_current').fetchone()[0]
        con.execute(f"""CREATE OR REPLACE TABLE filter_user_pass AS SELECT * FROM filter_current
            WHERE user_id IN (SELECT user_id FROM filter_current GROUP BY user_id
                              HAVING count(*)>={int(min_user_events)})""")
        con.execute("""CREATE OR REPLACE TABLE filter_session_pass AS SELECT * FROM filter_user_pass
            QUALIFY count(*) OVER(PARTITION BY user_id,window_id)>=2""")
        # 任何位置出现低频商品，都丢弃整段原会话；禁止删点后把两侧拼起来。
        con.execute(f"""CREATE OR REPLACE TABLE filter_bad_sessions AS
            SELECT DISTINCT user_id,window_id FROM filter_session_pass
            WHERE parent_asin IN (SELECT parent_asin FROM filter_session_pass GROUP BY parent_asin
                                  HAVING count(*)<{int(min_item_events)})""")
        con.execute("""CREATE OR REPLACE TABLE filter_item_pass AS SELECT e.* FROM filter_session_pass e
            WHERE NOT EXISTS (SELECT 1 FROM filter_bad_sessions b
                              WHERE e.user_id=b.user_id AND e.window_id=b.window_id)""")
        con.execute("""CREATE OR REPLACE TABLE filter_next AS SELECT * FROM filter_item_pass
            QUALIFY count(*) OVER(PARTITION BY user_id,window_id)>=2""")
        after,users,items=con.execute('SELECT count(*),count(DISTINCT user_id),count(DISTINCT parent_asin) FROM filter_next').fetchone()
        sessions=con.execute('SELECT count(*) FROM (SELECT user_id,window_id FROM filter_next GROUP BY user_id,window_id)').fetchone()[0]
        row={'iteration':len(rounds)+1,'before_interactions':before,
             'after_user_filter':con.execute('SELECT count(*) FROM filter_user_pass').fetchone()[0],
             'after_short_session_filter':con.execute('SELECT count(*) FROM filter_session_pass').fetchone()[0],
             'after_item_filter':con.execute('SELECT count(*) FROM filter_item_pass').fetchone()[0],
             'sessions_dropped_for_low_frequency_item':con.execute('SELECT count(*) FROM filter_bad_sessions').fetchone()[0],
             'interactions':after,'removed_interactions':before-after,'users':users,'items':items,'sessions':sessions,
             'converged':before==after}
        assert 0<=after<=before
        rounds.append(row)
        if on_round:on_round(row)
        con.execute('DROP TABLE filter_current');con.execute('ALTER TABLE filter_next RENAME TO filter_current')
        if before==after:break
    con.execute('CREATE TABLE filtered_events AS SELECT * FROM filter_current')
    con.execute("""CREATE TABLE final_sessions AS SELECT user_id,window_id,min(timestamp) AS session_start,
        max(timestamp) AS session_end,count(*) AS n FROM filtered_events GROUP BY user_id,window_id""")
    # 在截断前独立验证收敛不变量，防止某一步漏掉用户或商品的级联退化。
    checks=[('user_id',min_user_events),('parent_asin',min_item_events)]
    for field,threshold in checks:
        bad=con.execute(f'SELECT count(*) FROM (SELECT {field} FROM filtered_events GROUP BY {field} HAVING count(*)<{int(threshold)})').fetchone()[0]
        if bad:raise RuntimeError('Converged filter invariant failed: '+field)
    if con.execute('SELECT count(*) FROM final_sessions WHERE n<2').fetchone()[0]:raise RuntimeError('Short session after convergence')
    # 各保留会话必须与最初events中的同一会话条数完全一致；本过程只能整会话删除。
    broken=con.execute("""SELECT count(*) FROM
        (SELECT user_id,window_id,count(*) n FROM events GROUP BY user_id,window_id) original
        JOIN final_sessions retained USING(user_id,window_id) WHERE original.n<>retained.n""").fetchone()[0]
    if broken:raise RuntimeError('Retained session lost original events')
    for table in ['filter_current','filter_user_pass','filter_session_pass','filter_item_pass','filter_bad_sessions']:
        con.execute('DROP TABLE '+table)
    return rounds


def prepare_amazon(reviews, metadata, output, *, min_user_events=3, min_item_events=5,
                   max_session_length=20, train_ratio=.9, valid_ratio=0,
                   user_sample_modulus=1, threads=4, protocol='paper', split_seed=42):
    """按统一论文协议处理一个类目，输出目录必须不存在。

    全局清洗、固定24小时会话、迭代过滤、随机9:1划分；全部保留商品共用词表。
    不按训练未见删除测试交互。重复事件保留，同时间按原文件顺序。
    商品文本仅来自元数据，不读取评论正文或评分作特征。
    """
    if protocol != 'paper':
        raise ValueError('Only paper protocol is supported; legacy temporal train-only vocabulary was removed')
    if not isinstance(split_seed,int) or split_seed<0:raise ValueError('split_seed must be nonnegative')
    if train_ratio != .9 or valid_ratio != 0:raise ValueError('Paper protocol requires train_ratio=0.9 and valid_ratio=0')
    reviews, metadata, output = Path(reviews), Path(metadata), Path(output)
    if min(min_user_events, min_item_events, user_sample_modulus, threads) < 1 or max_session_length < 2:
        raise ValueError('Invalid filtering or resource parameters')
    if output.exists():
        raise FileExistsError('Output already exists; choose a new directory: ' + str(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    build = Path(tempfile.mkdtemp(prefix='.' + output.name + '-building-', dir=output.parent))
    started = time.perf_counter()
    source_stats = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in (reviews, metadata)}
    source_hashes = {'reviews': sha256_file(reviews), 'metadata': sha256_file(metadata)}
    con = duckdb.connect(str(build / 'processing.duckdb'))
    con.execute("SET memory_limit='8GB'")
    con.execute(f'SET threads={int(threads)}')
    con.execute('SET preserve_insertion_order=true')
    con.execute('SET temp_directory=' + _literal(build / 'spill'))
    con.create_function('price_value', lambda value: parse_price(value), ['VARCHAR'], 'DOUBLE', null_handling='special')
    con.create_function('safe_details', _safe_details, ['VARCHAR'], 'VARCHAR', null_handling='special')
    def count(table, condition='true'):
        """统计内部已知表的行数，供每步数据质量报告使用。"""
        return con.execute(f'SELECT count(*) FROM {table} WHERE {condition}').fetchone()[0]
    def stage(name):
        """打印结构化阶段与耗时，供长时间预处理观察进度。"""
        print(json.dumps({'prepare_stage': name, 'elapsed_seconds': round(time.perf_counter()-started, 2)}), flush=True)
    try:
        # 1. 只读交互必要字段；显式 VARCHAR 后 try_cast，坏时间可统计而不静默污染会话。
        stage('read_reviews')
        con.execute(f"""CREATE TABLE raw_reviews AS SELECT row_number() OVER () AS original_row,
            trim(user_id) AS user_id, trim(parent_asin) AS parent_asin,
            try_cast(coalesce(timestamp,sort_timestamp) AS BIGINT) AS timestamp
            FROM read_json({_literal(reviews)},format='newline_delimited',compression='auto_detect',
                columns={{user_id:'VARCHAR',parent_asin:'VARCHAR',timestamp:'VARCHAR',sort_timestamp:'VARCHAR'}})""")
        valid_fields="user_id IS NOT NULL AND user_id<>'' AND parent_asin IS NOT NULL AND parent_asin<>'' AND timestamp>0"
        selected=f"CAST('0x'||substr(md5(user_id),1,8) AS UBIGINT)%{int(user_sample_modulus)}=0"
        con.execute(f'CREATE TABLE interactions AS SELECT * FROM raw_reviews WHERE {valid_fields} AND {selected}')
        # 2. 元数据显式白名单，不读取 average_rating/rating_number/bought_together。
        stage('read_metadata')
        con.execute(f"""CREATE TABLE raw_meta AS SELECT row_number() OVER () AS meta_row,*
            FROM read_json({_literal(metadata)},format='newline_delimited',compression='auto_detect',
              columns={{parent_asin:'VARCHAR',categories:'VARCHAR[]',price:'VARCHAR',title:'VARCHAR',
                        features:'VARCHAR[]',description:'VARCHAR[]',details:'JSON'}})""")
        con.execute("""CREATE TABLE first_meta AS SELECT * FROM raw_meta
            WHERE parent_asin IS NOT NULL AND trim(parent_asin)<>''
            QUALIFY row_number() OVER(PARTITION BY parent_asin ORDER BY meta_row)=1""")
        con.execute("""CREATE TABLE parsed_meta AS SELECT trim(parent_asin) AS parent_asin,
            trim(list_extract(list_filter(categories,c -> c IS NOT NULL AND length(trim(c))>0),-1)) AS category,
            price_value(price) AS price,title,features,description,details FROM first_meta""")
        con.execute("""CREATE TABLE eligible_meta AS SELECT * FROM parsed_meta
            WHERE category IS NOT NULL AND category<>'' AND price IS NOT NULL""")
        stage('paper_global_filter_and_sessionize')
        # 按论文先清洗元数据，再统计全局用户频次；删除无效交互，不保留 OOV 占位。
        con.execute("""CREATE TABLE clean_interactions AS SELECT i.* FROM interactions i
            JOIN eligible_meta m USING(parent_asin)""")
        con.execute(f"""CREATE TABLE eligible_users AS SELECT user_id FROM clean_interactions
            GROUP BY user_id HAVING count(*)>={int(min_user_events)}""")
        con.execute("""CREATE TABLE events AS SELECT i.*,
            CAST(floor((timestamp-min(timestamp) OVER(PARTITION BY user_id))/86400000.0) AS BIGINT) AS window_id
            FROM clean_interactions i JOIN eligible_users u USING(user_id)""")
        con.execute("""CREATE TABLE sessions AS SELECT user_id,window_id,min(timestamp) AS session_start,
            max(timestamp) AS session_end,count(*) AS n FROM events GROUP BY user_id,window_id HAVING count(*)>=2""")
        def report_round(row):
            """逐轮打印并固化进度，失败的building目录也能追溯过滤过程。"""
            print(json.dumps({'prepare_stage':'paper_filter_iteration',**row}),flush=True)
        filter_rounds=iterative_session_filter(con,min_user_events,min_item_events,report_round)
        save_json(build/'filter_iterations.json',{'rounds':filter_rounds,'converged':True,
            'scope':'before_truncation_and_split','window_policy':'fixed_original_window_id'})
        total = count('final_sessions')
        if total < 2:
            raise ValueError('Paper protocol needs at least two retained sessions')
        train_count = int(total * train_ratio)
        # 排序稳定的 seeded hash 置换：整段会话分配、精确 floor(0.9*N) 训练，其余测试。
        # 论文未给随机种子/随机库；明确记录本实现，保证不受输入文件行顺序影响。
        con.execute(f"""CREATE TABLE split_sessions AS SELECT * EXCLUDE(split_rank),
            CASE WHEN split_rank<={train_count} THEN 'train' ELSE 'test' END AS split FROM
            (SELECT *,row_number() OVER(ORDER BY
                sha256({ _literal(split_seed) }||':'||sha256(user_id||':'||window_id::VARCHAR)),
                user_id,window_id) AS split_rank FROM final_sessions)""")
        con.execute(f"""CREATE TABLE recent_events AS SELECT e.*,s.split,s.session_start,s.session_end
            FROM filtered_events e JOIN split_sessions s USING(user_id,window_id)
            QUALIFY row_number() OVER(PARTITION BY e.user_id,e.window_id
                ORDER BY timestamp DESC,original_row DESC)<={int(max_session_length)+1}""")
        # 为最后的预测目标额外预留1次：输入最多max_session_length次，不把目标计入输入上限。
        # 全量保留序列建立目录/类别/价格映射；不声称所有目录商品都在训练交互中出现。
        con.execute("""CREATE TABLE catalog AS SELECT row_number() OVER(ORDER BY m.parent_asin)::INTEGER AS item_id,m.*
            FROM eligible_meta m JOIN (SELECT DISTINCT parent_asin FROM recent_events) r USING(parent_asin)""")
        con.execute("""CREATE TABLE recent AS SELECT r.*,c.item_id FROM recent_events r JOIN catalog c USING(parent_asin)""")
        con.execute("""CREATE TABLE sequences AS SELECT user_id,window_id,split,session_start,session_end,
            list(item_id::INTEGER ORDER BY timestamp,original_row) AS ids
            FROM recent GROUP BY user_id,window_id,split,session_start,session_end""")
        con.execute("""CREATE TABLE examples AS SELECT *,list_slice(ids,1,len(ids)-1) AS input_item_ids,
            list_extract(ids,-1) AS target_item_id FROM sequences""")
        con.execute('CREATE TABLE retained AS SELECT * FROM examples')
        counts = {split: count('retained',f"split='{split}'") for split in ('train','valid','test')}
        if not counts['train'] or not counts['test']:
            raise ValueError('Empty paper train/test split: ' + repr(counts))
        train_end = valid_end = None
        stage('write_artifacts')
        for split in counts:
            con.execute(f"""COPY(SELECT input_item_ids,target_item_id,user_id,session_start,session_end,
                sha256(user_id||':'||window_id::VARCHAR) AS session_id FROM retained WHERE split='{split}'
                ORDER BY session_end,session_start,user_id,window_id)
                TO {_literal(build/(split+'.parquet'))} (FORMAT PARQUET,COMPRESSION ZSTD)""")
        catalog = dict(con.execute('SELECT parent_asin,item_id FROM catalog ORDER BY item_id').fetchall())
        features = fit_item_features(con.execute('SELECT item_id,category,price FROM catalog ORDER BY item_id').fetchall())
        # 热门统计来自最终保留的训练样本完整序列，包含训练标签，不含 OOV。
        con.execute("""CREATE TABLE popularity_counts AS SELECT item_id,count(*) AS n FROM
            (SELECT unnest(ids) AS item_id FROM retained WHERE split='train') t WHERE item_id>0 GROUP BY item_id""")
        popularity = [row[0] for row in con.execute('SELECT c.item_id FROM catalog c LEFT JOIN popularity_counts p USING(item_id) ORDER BY coalesce(p.n,0) DESC,c.item_id').fetchall()]
        save_json(build/'catalog.json',catalog)
        save_json(build/'item_features.json',features)
        save_json(build/'popularity.json',popularity)
        con.execute(f"""COPY(SELECT item_id,parent_asin,category,price,title,features,description,
            safe_details(details::VARCHAR) AS details FROM catalog ORDER BY item_id)
            TO {_literal(build/'products.parquet')} (FORMAT PARQUET,COMPRESSION ZSTD)""")
        stats = {'raw_reviews': count('raw_reviews'), 'invalid_review_fields': count('raw_reviews',f'NOT coalesce(({valid_fields}),false)'),
                 'selected_interactions': count('interactions'),'raw_metadata_rows':count('raw_meta'),
                 'metadata_unique_products':count('first_meta'),'eligible_metadata_products':count('eligible_meta'),
                 'metadata_missing_category':count('parsed_meta',"category IS NULL OR category=''"),
                 'metadata_invalid_price':count('parsed_meta','price IS NULL'),
                 'raw_valid_sessions':count('sessions'),'crossing_sessions_removed':count('split_sessions',"split='crossing'"),
                 'catalog_size':len(catalog),'category_count':len(features['category_mapping']),'split_examples':counts,
                 'oov_targets':{s:count('retained',f"split='{s}' AND target_item_id=0") for s in counts},
                 'no_known_history':{s:count('retained',f"split='{s}' AND len(list_filter(input_item_ids,x -> x>0))=0") for s in counts},
                 'train_unknown_target_removed':count('examples',"split='train' AND target_item_id=0"),
                 'train_no_known_history_removed':count('examples',"split='train' AND target_item_id>0 AND len(list_filter(input_item_ids,x -> x>0))=0")}
        # 进一步报告每步过滤与未知位置数量；这些统计不参与拟合或调参。
        stats['unique_users'] = con.execute('SELECT count(DISTINCT user_id) FROM interactions').fetchone()[0]
        stats['unique_raw_items'] = con.execute('SELECT count(DISTINCT parent_asin) FROM interactions').fetchone()[0]
        stats['raw_sessions_including_singletons'] = con.execute('SELECT count(*) FROM (SELECT user_id,window_id FROM events GROUP BY user_id,window_id)').fetchone()[0]
        stats['short_sessions_removed'] = stats['raw_sessions_including_singletons'] - stats['raw_valid_sessions']
        stats['interactions_with_eligible_metadata'] = con.execute('SELECT count(*) FROM interactions JOIN eligible_meta USING(parent_asin)').fetchone()[0]
        stats['raw_split_sessions'] = {split:count('split_sessions',f"split='{split}'") for split in counts}
        stats['avg_input_length'] = {split:con.execute(f"SELECT avg(len(input_item_ids)) FROM retained WHERE split='{split}'").fetchone()[0] for split in counts}
        stats['oov_input_events'] = {split:con.execute(f"SELECT sum(len(list_filter(input_item_ids,x -> x=0))) FROM retained WHERE split='{split}'").fetchone()[0] for split in counts}
        # OOV=0 与目录内但训练交互未出现是两个不同指标，论文协议也必须分别报告。
        con.execute("""CREATE TABLE seen_train_items AS SELECT DISTINCT unnest(ids) AS item_id
            FROM retained WHERE split='train'""")
        stats['train_seen_catalog_items'] = count('seen_train_items', 'item_id>0')
        stats['catalog_items_unseen_in_train'] = len(catalog) - stats['train_seen_catalog_items']
        stats['targets_unseen_in_train'] = {s: count('retained',
            f"split='{s}' AND (target_item_id=0 OR target_item_id NOT IN(SELECT item_id FROM seen_train_items))")
            for s in counts}
        parameters = {'min_user_events':min_user_events,'min_item_events':min_item_events,
                      'max_session_length':max_session_length,'train_ratio':train_ratio,'valid_ratio':valid_ratio,
                      'user_sample_modulus':user_sample_modulus,'train_end':train_end,'valid_end':valid_end,
                      'window_ms':86400000,'split_seed':split_seed if protocol=='paper' else None,
                      'filter_scope':'global_before_split',
                      'feature_fit_scope':'full_retained_catalog',
                      'filter_iterations':'until_no_interactions_removed',
                      'low_frequency_item_policy':'drop_entire_session_containing_any_low_frequency_item',
                      'split_algorithm':'seeded_sha256_session_permutation_floor_90_percent',
                      'duplicate_policy':'preserve_events;first_metadata_row;stable_source_order',
                      'invalid_metadata_policy':'delete_interactions_before_sessionize',
                      'price_formula':'paper_3_28_to_31;population_std;bins_1_to_50',
                      'max_history_length':max_session_length,
                      'max_retained_events':max_session_length+1,
                      'truncation_policy':'last_target_then_recent_history'}
        names=['train.parquet','valid.parquet','test.parquet','catalog.json','item_features.json','popularity.json','products.parquet']
        if protocol=='paper':
            names.append('filter_iterations.json')
            stats['converged_filter']=filter_rounds[-1]
            stats['filter_round_count']=len(filter_rounds)
        hashes={name:sha256_file(build/name) for name in names}
        identity={'protocol':'amazon2023-fixed24h-paper-iterative-drop-session-history20-v4','parameters':parameters,
                  'source_sha256':source_hashes,'artifact_sha256':hashes}
        manifest={'schema_version':2,**identity,'stats':stats,'synthetic_debug':False,
                  'full_category':user_sample_modulus==1,'duckdb_version':duckdb.__version__,
                  'source_files':{'reviews':reviews.name,'metadata':metadata.name},
                  'limitations':['review-derived proxy sessions, not click logs','metadata and prices are snapshots'],
                  'fingerprint':hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
                  'elapsed_seconds':time.perf_counter()-started}
        for p in (reviews,metadata):
            if (p.stat().st_size,p.stat().st_mtime_ns)!=source_stats[str(p)]:
                raise RuntimeError('Source changed during preprocessing: '+str(p))
        save_json(build/'manifest.json',manifest)
    finally:
        con.close()
    # 只有全部步骤成功才公开最终目录；失败的 building 目录保留供诊断，不覆盖已有产物。
    (build / 'processing.duckdb').unlink()  # 成功后只清理本次新建且已关闭的临时数据库。
    build.rename(output)
    stage('complete')
    return manifest
