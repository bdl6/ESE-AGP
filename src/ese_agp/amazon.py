"""Amazon Reviews 2023 正式预处理：磁盘 SQL，支持时间协议与第三章论文的全局过滤/随机会话协议。"""
from pathlib import Path
import hashlib
import json
import tempfile
import time

import duckdb

from .artifacts import sha256_file, save_json
from .features import parse_price, fit_item_features


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


def prepare_amazon(reviews, metadata, output, *, min_user_events=3, min_item_events=5,
                   max_session_length=20, train_ratio=.8, valid_ratio=.1,
                   user_sample_modulus=1, threads=4, protocol='temporal', split_seed=42):
    """完整处理一个类目，返回 manifest；output 必须为尚不存在的产物目录。

    temporal 从全部合法交互构造时间边界，由训练会话拟合频次和目录。
    paper 删除无效元数据交互，全局过滤后按会话随机 9:1；无独立验证集。
    temporal 的无效元数据商品保留 OOV 位置；paper 则删除这些交互。
    user_sample_modulus=1 表示全类目；>1 按稳定用户哈希抽取完整用户历史用于排错。
    重复事件保留，时间并列按原文件行顺序；元数据 parent_asin 重复时取首次记录。
    原始评论正文、评分和反馈字段从不读取为商品特征，文本仅来自商品元数据。
    """
    # paper: 先全局清洗/过滤，再以稳定种子随机划分会话；valid_ratio 必须为 0。
    # temporal: 保持原有训练内拟合、按时间划分和 OOV 占位协议。
    if protocol not in {'temporal', 'paper'}:
        raise ValueError('protocol must be temporal or paper')
    if not isinstance(split_seed, int) or split_seed < 0:
        raise ValueError('split_seed must be a nonnegative integer')
    if protocol == 'paper' and (train_ratio != .9 or valid_ratio != 0):
        raise ValueError('Paper protocol requires train_ratio=0.9 and valid_ratio=0')
    reviews, metadata, output = Path(reviews), Path(metadata), Path(output)
    if min(min_user_events, min_item_events, user_sample_modulus, threads) < 1 or max_session_length < 2:
        raise ValueError('Invalid filtering or resource parameters')
    if protocol == 'temporal' and not 0 < train_ratio < train_ratio + valid_ratio < 1:
        raise ValueError('Invalid split ratios')
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
        if protocol == 'temporal':
            # 3. 所有合法原始事件保留位置，缺元数据的商品不先删除，以防桥接前后交互。
            stage('sessionize_and_split')
            con.execute("""CREATE TABLE events AS SELECT *,
                CAST(floor((timestamp-min(timestamp) OVER(PARTITION BY user_id))/86400000.0) AS BIGINT) AS window_id
                FROM interactions""")
            con.execute("""CREATE TABLE sessions AS SELECT user_id,window_id,min(timestamp) AS session_start,
                max(timestamp) AS session_end,count(*) AS n FROM events GROUP BY user_id,window_id HAVING count(*)>=2""")
            if not count('sessions'):
                raise ValueError('No length-at-least-two proxy sessions')
            train_end, valid_end = con.execute(f'SELECT quantile_disc(session_end,{float(train_ratio)}),quantile_disc(session_end,{float(train_ratio+valid_ratio)}) FROM sessions').fetchone()
            con.execute(f"""CREATE TABLE split_sessions AS SELECT *,CASE
                WHEN (session_start<={train_end} AND session_end>{train_end})
                  OR (session_start<={valid_end} AND session_end>{valid_end}) THEN 'crossing'
                WHEN session_end<={train_end} THEN 'train'
                WHEN session_end<={valid_end} THEN 'valid' ELSE 'test' END AS split FROM sessions""")
            # 4. 单次 user→item 训练内频次过滤，不是迭代 k-core，也不看评估期频次。
            con.execute(f"""CREATE TABLE eligible_users AS SELECT e.user_id FROM events e JOIN split_sessions s
                USING(user_id,window_id) WHERE split='train' GROUP BY e.user_id HAVING count(*)>={int(min_user_events)}""")
            con.execute(f"""CREATE TABLE item_counts AS SELECT e.parent_asin,count(*) AS n FROM events e
                JOIN split_sessions s USING(user_id,window_id) JOIN eligible_users u USING(user_id)
                JOIN eligible_meta m USING(parent_asin) WHERE split='train'
                GROUP BY e.parent_asin HAVING count(*)>={int(min_item_events)}""")
            con.execute("""CREATE TABLE catalog AS SELECT row_number() OVER(ORDER BY m.parent_asin)::INTEGER AS item_id,m.*
                FROM eligible_meta m JOIN item_counts c USING(parent_asin)""")
            if not count('catalog'):
                raise ValueError('No training items survive metadata and frequency filtering')
            # 5. 先截取末尾含目标的最多 20 次，再编号；OOV=0 保留在原位置，不向前换目标。
            con.execute(f"""CREATE TABLE recent AS SELECT e.*,s.split,s.session_start,s.session_end,c.item_id
                FROM events e JOIN split_sessions s USING(user_id,window_id)
                LEFT JOIN catalog c USING(parent_asin) WHERE split<>'crossing'
                QUALIFY row_number() OVER(PARTITION BY e.user_id,e.window_id ORDER BY timestamp DESC,original_row DESC)<={int(max_session_length)}""")
            con.execute("""CREATE TABLE sequences AS SELECT user_id,window_id,split,session_start,session_end,
                list(coalesce(item_id,0)::INTEGER ORDER BY timestamp,original_row) AS ids
                FROM recent GROUP BY user_id,window_id,split,session_start,session_end""")
            con.execute("""CREATE TABLE examples AS SELECT *,list_slice(ids,1,len(ids)-1) AS input_item_ids,
                list_extract(ids,-1) AS target_item_id FROM sequences""")
            con.execute("""CREATE TABLE retained AS SELECT e.* FROM examples e WHERE split<>'train' OR
                (user_id IN(SELECT user_id FROM eligible_users) AND target_item_id>0
                 AND len(list_filter(input_item_ids,x -> x>0))>0)""")
            counts = {split: count('retained',f"split='{split}'") for split in ('train','valid','test')}
            if not counts['train'] or not counts['valid'] or not counts['test']:
                raise ValueError('Empty split after filtering: ' + repr(counts))
        else:
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
            # 论文列明 user -> 短会话 -> item -> 最终会话校验，未给定反复 k-core 的停止条件。
            # 因而采用该显式单轮顺序，不擅自增加迭代直至收敛规则。
            con.execute(f"""CREATE TABLE item_counts AS SELECT e.parent_asin,count(*) AS n FROM events e
                JOIN sessions s USING(user_id,window_id) GROUP BY e.parent_asin
                HAVING count(*)>={int(min_item_events)}""")
            con.execute("""CREATE TABLE filtered_events AS SELECT e.* FROM events e
                JOIN sessions s USING(user_id,window_id) JOIN item_counts c USING(parent_asin)""")
            con.execute("""CREATE TABLE final_sessions AS SELECT user_id,window_id,min(timestamp) AS session_start,
                max(timestamp) AS session_end,count(*) AS n FROM filtered_events
                GROUP BY user_id,window_id HAVING count(*)>=2""")
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
                    ORDER BY timestamp DESC,original_row DESC)<={int(max_session_length)}""")
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
                      'filter_scope':'global_before_split' if protocol=='paper' else 'train_only',
                      'feature_fit_scope':'full_retained_catalog' if protocol=='paper' else 'train_catalog',
                      'filter_iterations':'single_user_then_item_then_session_check',
                      'split_algorithm':'seeded_sha256_session_permutation_floor_90_percent' if protocol=='paper' else 'session_end_quantiles',
                      'duplicate_policy':'preserve_events;first_metadata_row;stable_source_order',
                      'invalid_metadata_policy':('delete_interactions_before_sessionize' if protocol=='paper' else 'exclude_from_catalog_but_preserve_event_positions_as_OOV'),
                      'price_formula':'paper_3_28_to_31;population_std;bins_1_to_50'}
        names=['train.parquet','valid.parquet','test.parquet','catalog.json','item_features.json','popularity.json','products.parquet']
        hashes={name:sha256_file(build/name) for name in names}
        identity={'protocol':('amazon2023-fixed24h-paper-global-random-v1' if protocol=='paper' else 'amazon2023-fixed24h-temporal-trainfit-v2'),'parameters':parameters,
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
