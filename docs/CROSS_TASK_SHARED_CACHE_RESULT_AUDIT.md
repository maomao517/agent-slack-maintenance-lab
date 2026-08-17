# 跨任务共享视觉状态缓存：结果审计与补充测量清单

本文档对`results/d2/d2_results.json`中的120个跨任务共享缓存实验条件进行上报前审计，识别影响开题报告结论的阻塞项，并记录完整视觉状态恢复的补充验证结果。配套执行说明见[D2_SUPPLEMENTAL_EXPERIMENTS.md](D2_SUPPLEMENTAL_EXPERIMENTS.md)。

## 1. 审计方法

- **结果字段审计**：运行`scripts/audit_d2_results.py`，核对答案一致性、状态大小、CPU到GPU传输、重复编码和工作流JCT边界。
- **完整状态与输出一致性验证**：运行`scripts/verify_vlm_cache_correctness.py`，比较同一图像和提示下的冷路径与缓存命中路径，检查DeepStack结构、logits、贪心生成和视觉模块调用次数。

审计产物应保存在`results/d2/`：

|产物|路径|状态|
|---|---|---|
|结果字段审计|`results/d2/d2_result_audit.json`|已完成，`report_ready=false`|
|正确性验证（low）|`results/d2/cache_correctness_low.json`|已完成，`passed=true`|
|正确性验证（medium）|`results/d2/cache_correctness_medium.json`|已完成，`passed=true`|
|正确性验证（high）|`results/d2/cache_correctness_high.json`|因执行时间较长暂未完成|

## 2. 结果字段审计

当前结果为`3个P0、1个P1、1个P2`，因此尚不能不加限制地写入正式开题报告。

|优先级|问题|审计证据|处理要求|
|---|---|---|---|
|P0|状态大小统计不完整|`vision_state_mb`最高为0.625MB，只统计了low页面的`image_embeds`|将状态容量字段改为完整`image_embeds+3组DeepStack`|
|P0|矩阵内答案一致性未采集|2520个`answer_consistency`值全部为0|将其解释为占位字段，并使用独立正确性实验提供机制级证据|
|P0|缺少工作流边界|存在`avg_jct_ms/p95_jct_ms`，但没有工作流起止时间|将现有指标改称请求时延；补采后再报告工作流JCT|
|P1|CPU到GPU恢复未计时|`cpu_to_gpu_transfer_ms`全部为0|增加CPU查找、恢复、注入、前向和排队等待分项计时|
|P2|未形成请求合并压力|`duplicate_encoder_calls`全部为0|构造同步同页冷未命中工作负载|

### 2.1 状态大小统计问题

对D2详细记录统计得到：

```text
vision_state_mb分布：{0.0:9300,0.625:26696}
```

`stage_d2_cross_task_shared.py::get_vision_state_size_mb`只遍历`entry[0]`中的基础视觉嵌入，没有统计`entry[1]`中的3组DeepStack特征。该问题首先是**统计口径错误**，不能据此直接判断D2运行时缓存对象一定缺少DeepStack；还需要单独检查D2的保存、加载和注入代码。

早期文档曾按另一组固定视觉序列长度估算完整状态为low页面2.5MB、medium页面9.375MB。这两个数值没有对应当前验证页面的实际处理器输出，现已废弃，不能继续用于缓存容量、搬运量或可容纳页面数分析。

补充验证直接遍历缓存状态中的全部张量，并按实际`dtype`与元素数量计算，得到：

|验证输入|实测完整状态|
|---|---:|
|low页面|7.875MB|
|medium页面|30.375MB|

上述实测值是当前正确性验证页面的准确容量口径。后续结果必须同时记录：

```text
image_grid_thw
visual_token_count
image_embeds_size_mb
deepstack_size_mb
reusable_state_size_mb
逐张量shape、dtype和size_mb
```

状态容量应由实际张量结构计算，而不能只根据“low”“medium”等分辨率名称推断。

## 3. 完整状态与输出一致性验证

### 3.1 注入层级问题及修复

早期验证脚本只覆盖外层`feature_owner.get_image_features`，但Qwen3-VL前向直接调用内层`feature_owner.model.get_image_features`。外层方法只是委托转发，因此旧命中路径仍执行完整视觉编码，表现为：

```text
encoder_calls_on_hit=2
visual_module_calls_on_hit>0
命中生成时间约20s
```

修复后，缓存注入点改为内层前向真实调用的`get_image_features`。该修复同步应用于：

```text
src/slackmaint/document_service.py
scripts/verify_vlm_cache_correctness.py
```

修复后的命中路径满足：

```text
cached_deepstack_count=3
visual_module_calls_on_hit=0
encoder_calls_on_hit=0
max_abs_logit_error=0
greedy_output_equal=true
```

### 3.2 low页面验证结果

|检查项|结果|
|---|---:|
|`cached_deepstack_count`|3|
|`complete_state`|true|
|`max_abs_logit_error`|0.0|
|`greedy_output_equal`|true|
|`visual_module_calls_on_hit`|0|
|`encoder_calls_on_hit`|0|
|`passed`|true|
|命中生成时间|约0.56s|

### 3.3 medium页面验证结果

|检查项|结果|
|---|---:|
|`cached_deepstack_count`|3|
|`complete_state`|true|
|`max_abs_logit_error`|0.0|
|`greedy_output_equal`|true|
|`visual_module_calls_on_hit`|0|
|`encoder_calls_on_hit`|0|
|`passed`|true|
|命中生成时间|约0.74s|

low和medium验证说明当前`slackmaint`缓存后端可以保存并恢复完整视觉状态，在确定性生成条件下保持输出一致，并真实跳过视觉编码器。high页面因执行时间较长暂未完成，不影响low和medium路径的机制正确性结论。

### 3.4 正确性证据的范围

补充验证证明的是当前`slackmaint`缓存后端，而不是自动证明此前运行的全部120个D2条件都使用了修复后的内层注入路径。必须对D2矩阵执行代码或日志抽查，至少确认一个共享缓存命中条件满足：

```text
visual_module_calls_on_hit=0
encoder_ms=0
命中请求时延接近0.56至0.74s，而不是冷编码时延
```

如果原D2脚本使用的是旧外层注入点，现有策略时延和编码避免率需要修复后重跑；如果D2脚本具有独立且正确的Encoder-Hit实现，则无需因验证脚本的旧bug重跑整个矩阵。

## 4. 指标解释边界

### 4.1 当前可以保留的观察

在确认D2矩阵命中路径真实跳过视觉编码器后，可以保留以下结论：

1.视觉编码是请求处理的主要计算开销；
2.任务内缓存和共享缓存均减少了视觉编码调用；
3.页面重叠率提高时，多数策略的编码次数下降；
4.共享策略相对任务内缓存具有额外但有限的平均请求时延收益；
5.静态缓存策略没有稳定改善P95请求时延；
6.SharedCPU具有较高跨任务命中率，但命中率没有直接转化为最低请求时延。

### 4.2 当前不得直接使用的表述

```text
视觉状态仅需0.625MB
全部120个条件均完成答案一致性验证
SharedCPU因CPU到GPU传输而变慢
SharedCoalesced已经抑制缓存击穿
共享缓存显著降低P95工作流JCT
SharedGPU是跨GPU全局共享缓存
旧版本状态已经被物理删除
```

现有`avg_jct_ms/p95_jct_ms`应改称平均请求时延和P95请求时延。只有补充`workflow_start_ns`、`workflow_end_ns`和`workflow_jct_ms`后，才能报告完整智能体工作流JCT。

## 5. 最小补充测量清单

1.抽查D2矩阵的真实命中路径，确认视觉模块调用为0；
2.修正`vision_state_mb`，落盘完整状态及逐张量大小；
3.记录工作流起止边界，计算工作流JCT；
4.补充CPU查找、CPU到GPU恢复、状态注入、缓存后前向和排队等待；
5.增加同步同页冷未命中工作负载，测试请求合并；
6.时间允许时补充high页面正确性验证。

## 6. 对研究方案的意义

当前结果不应简单导向“提高缓存命中率”。更准确的研究问题是：

> 在并发多模态智能体工作流中，如何根据完整视觉状态的预计恢复时间、预计复用时间和资源压力，在GPU热缓存、CPU共享缓存与重新编码之间作出决策，从而降低平均请求时延和工作流JCT，并控制尾部时延与无效状态驻留？

现有D2矩阵用于提供页面复用、平均请求时延和层级权衡的动机证据；补充正确性实验用于证明完整视觉状态能够无损恢复。二者应分别报告，不能用机制级正确性验证替代矩阵内的指标采集。

## 7. 当前判定

- **缓存机制可行性**：low和medium页面已通过完整状态、输出一致性和真实Encoder-Hit验证。
- **D2矩阵可用性**：等待确认矩阵本身使用正确的内层注入路径。
- **容量指标**：low和medium正确性验证页面的完整状态已实测为7.875MB和30.375MB；D2矩阵仍需逐条记录实际张量结构，不能按分辨率名称套用固定值。
- **工作流JCT**：尚未采集，现有数据仅为请求级时延。
- **CPU层性能归因**：尚缺分项计时。
- **请求合并收益**：尚未形成有效压力。

因此，当前结果可以作为开题报告的动机和技术可行性材料；在D2矩阵命中路径与完整状态统计确认前，不应作为最终系统性能结论。
