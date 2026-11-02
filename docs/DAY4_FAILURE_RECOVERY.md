# 实体试抽取失败定位与重试

首轮100件：成功5，失败95。错误类型为URLError 83件、TimeoutError 1件、ValueError 8件、JSONDecodeError 3件。
生成服务在sample_tokens RPC超时后EngineCore退出，18888端口停止监听；底层超时的进一步原因没有证据确定，不能直接归因于显存不足或提示词。

恢复时仅清理该引擎日志中确认的4个孤儿worker，在原GPU0–3重新加载同一Qwen3-30B-A3B-Instruct-2507；改用enforce-eager排除编译/图捕获路径的影响。
其他向量和重排序服务未重启。服务日志为runs/day04_qwen_service_recovery.log，恢复参数在runs/day04_service_recovery.json。

抽取器修复：完整Markdown围栏剥离后严格解析JSON；证据错误反馈模型重试；连接/超时连续失败后熔断批次；统计熔断后仍保留已有成功缓存。
不通过补造原文证据或忽略实体校验来提高成功率。

首轮目录完整归档到data/entities/extracted_pilot_v2_before_retry_20261030/。原成功5件继续复用；失败缓存被本轮结果逐项更新。
重试状态在runs/day04_retry_status.json；结束后生成docs/DAY4_RETRY_RESULTS.md及独立retry1语义/融合图。
服务重启不能保证根因完全消除，本轮如再次断线会暂停，不会把剩余商品批量标作模型质量失败。
