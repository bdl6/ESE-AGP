# Day04：实体抽取与语义构图

使用服务器自托管的生成模型 `Qwen3-30B-A3B-Instruct-2507`，通过本机 `http://127.0.0.1:18888/v1` 访问。
Embedding和Reranker服务未用于抽取或IDF语义图，不需要额外向量化。

## 已实现

- entities/client.py：兼容Chat Completions协议的真实请求，本机可无鉴权，远程HTTPS使用环境变量密钥。
- entities/pipeline.py：两轮抽取及遗漏检查、逐商品原子缓存、跨重启累计请求上限、最多3次重试、原始响应及token统计。
- schema和原文证据检查失败不会记为成功空实体；401等非临时服务错误立即停止。
- graphs/semantic.py：共享实体IDF边权、逐节点top20、源归一；类型参与实体身份；不添加自语义边。
- graphs/fusion.py：归一后的行为/语义图按beta相加，合并重复边；默认beta=0.5。
- CLI增加extract、build-semantic、fuse-graphs；prepare-entities支持指定数量与输出路径。
- 68项测试通过，模拟服务端到端验证；真实结果与模拟结果分别保存。

## 当前真实运行

配置：configs/day04_entities.yaml。100件商品，累计请求上限1000，单次最大输出2048token。该上限是本次pilot执行边界，不是全量调用授权。
真实缓存：data/entities/extracted_pilot_v2/；抽取日志：runs/day04_extract_pilot.log。
一次v1探测发现类型误判，保留在runs/day04_service_probe_raw.json；v2收紧提示词后启动正式pilot。
当前运行是否结束以entities.json和runs/day04_pilot_report.json为准，不预填成功率。
后续完成脚本只等待本次抽取，自动生成partial语义/融合诊断图和docs/DAY4_PILOT_RESULTS.md，不调用模型。

## 质量与边界

已有样例暴露出把标题当艺人、补充原文不存在的CD格式等问题。格式校验与证据校验无法完全保证语义角色正确，必须审核。
100件pilot不足以代表全训练目录：partial图默认不能进入正式训练。候选对超过设定预算会明确停止，避免泛化实体引发图规模失控。
没有进行真实人工30条审核，不声称实体质量已通过，也不自动扩展到全量商品。
这些模块不包含图剪枝或多视角预测。未自动提交/推送本轮新增代码。

## 复用命令

```bash
.venv/bin/python -m ese_agp extract --config configs/day04_entities.yaml
.venv/bin/python -m ese_agp build-semantic --config configs/day04_entities.yaml --allow-partial
.venv/bin/python -m ese_agp fuse-graphs --config configs/day04_entities.yaml --allow-partial
```

本轮已有任务和自动完成脚本时不要重复执行构图命令；图文件拒绝覆盖，缓存与输出目录绑定服务、文本和提示词版本。
服务提供的是内网HTTP地址时，使用同服务器回环地址；不会向公共HTTP地址发送密钥。
