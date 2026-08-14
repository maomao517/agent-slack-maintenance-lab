# 阶段A至D实验结果口径核验清单

## 1. 使用说明

本清单用于核验2026年8月14日更新实验脚本后生成的阶段A至D结果。请公司Agent直接读取最新实验脚本和以下结果文件，不使用旧报告中的估算数据：

```text
results/profiling/results_full.json
results/profiling/summary.csv
results/profiling/aggregated_stats.json
results/extended/extended_results.json
results/cache/cache_experiment_results.json
results/agent_trajectories/trajectory_results.json
```

核验目标不是重新运行完整实验，而是确认4个字段和计时口径。输出中不得包含公司用户名、内部绝对路径、服务器地址、原始内部文档或其他敏感信息。

## 2. 问题一：输入规模字段是否为视觉token数

请定位阶段B中以下数值的生成代码：

```text
low：491,520
medium：1,843,200
high：7,618,560
```

需要回答：

1.该字段的变量名是什么；
2.计算公式是什么；
3.它表示原图像素数、处理后像素数、`pixel_values`元素数量、`image_grid_thw`乘积，还是语言模型实际接收的视觉token数；
4.Qwen3-VL处理器是否在计算过程中进行了空间合并；
5.不同分辨率下实际视觉序列长度分别是多少。

请输出脱敏后的关键代码和结果：

```text
field_name=
formula=
low_processed_pixels=
medium_processed_pixels=
high_processed_pixels=
low_image_grid_thw=
medium_image_grid_thw=
high_image_grid_thw=
low_visual_sequence_length=
medium_visual_sequence_length=
high_visual_sequence_length=
```

判定规则：

- 如果字段由`width×height`得到，应在报告中称为“处理后像素数”；
- 如果字段由`image_grid_thw`直接相乘得到，需要说明是否已经考虑空间合并；
- 只有与语言模型实际视觉序列长度一致时，才能称为“视觉token数”。

## 3. 问题二：完整可复用状态是2.34MB还是9.375MB

请检查阶段C的`Encoder-Hit`路径，确认缓存中实际保存和恢复的对象。

需要回答：

1.`get_image_features`返回值的完整Python结构是什么；
2.缓存对象包含哪些张量；
3.每个张量的名称、shape、dtype、`numel`和大小；
4.命中时恢复的是仅`image_embeds`，还是`image_embeds+deepstack`；
5.是否做过只缓存`image_embeds`的独立消融；
6.只缓存`image_embeds`时，视觉模块调用次数、logits最大绝对误差和贪心解码结果分别是什么。

请输出：

```text
cached_object_type=
cached_tensor_count=

tensor_name,shape,dtype,numel,size_mb
...

image_embeds_only_tested=true/false
visual_calls_on_hit=
max_abs_logit_error=
greedy_output_equal=true/false
reusable_encoder_state_mb=
```

判定规则：

- 如果只缓存2.34MB的`image_embeds`即可跳过视觉编码且保持输出一致，则2.34MB可以作为最小完整可复用状态；
- 如果命中路径仍使用`deepstack`，则缓存容量必须包含所有`deepstack`张量，不能只统计`image_embeds`；
- `reusable_encoder_state_mb`必须对应实际命中路径使用的全部张量。

## 4. 问题三：Hit耗时是否包含状态加载

请检查GPU、CPU和文件后端的计时边界，提供`start`、`load`、设备传输、状态注入、模型前向、`synchronize`和`end`附近的脱敏代码。

重点回答：

1.表中的`Hit耗时≈0.29s`是否已经包含`backend.load()`；
2.CPU的235.3ms加载是否已经包含在0.29s中；
3.文件后端的248.5ms加载是否已经包含在0.29s中；
4.计时前后是否调用`torch.cuda.synchronize()`；
5.保存耗时是否被计入首次冷请求；
6.文件加载是否为操作系统页缓存命中。

请按下面格式输出：

```text
hit_timer_starts_before_load=true/false
hit_timer_ends_after_forward=true/false
cuda_synchronize_before=true/false
cuda_synchronize_after=true/false

backend,cold_wall_ms,load_wall_ms,transfer_wall_ms,cached_forward_wall_ms,end_to_end_hit_wall_ms,save_wall_ms
gpu,...
cpu,...
disk_warm,...
```

加速比统一计算为：

```text
speedup=cold_wall_ms/end_to_end_hit_wall_ms
```

如果`end_to_end_hit_wall_ms`由一个墙钟计时器完整覆盖加载、传输、注入和后续前向，则不要再次把`load_wall_ms`相加。如果0.29s只表示加载后的模型执行，则必须将加载和传输计入端到端恢复时间。

## 5. 问题四：阶段D命中率和耗时的分母

请检查阶段D四类轨迹的访问日志和汇总函数。当前报告为：

```text
Same-Image：83%
Page-Revisit：58%
Version-Change：50%
All-Distinct：33%
```

若每条4轮轨迹从空缓存开始且不同工作流缓存隔离，理论命中率应为75%、50%、50%和0%。因此需要说明实际统计范围。

请回答：

1.每类轨迹实际运行多少个工作流和多少轮；
2.命中率分子和分母分别是什么；
3.预热访问是否计入；
4.不同工作流是否共享缓存；
5.不同轨迹类型之间是否共享缓存；
6.缓存是否在每次重复前清空；
7.`13.8s`、`32.3s`、`43.4s`和`53.5s`表示单轮平均时延、单个工作流JCT，还是多次工作流运行的平均值；
8.P95 JCT的样本数量是多少。

请输出每类轨迹的汇总：

```text
trace_type=
workflow_count=
rounds_per_workflow=
total_image_accesses=
cache_hits=
cache_misses=
encoder_calls=
hit_rate=
cache_cleared_between_workflows=true/false
cache_shared_between_workflows=true/false
latency_metric_name=
latency_sample_count=
latency_median_ms=
latency_p95_ms=
```

并输出每轮脱敏记录：

```text
trace_type,workflow_id,round_id,image_id,document_version,cache_hit,encoder_called,elapsed_ms
```

## 6. 公司Agent可直接执行的总提示词

```text
请审计更新脚本后生成的阶段A至D实验结果，不重新运行完整实验，只检查以下4个口径：

1.阶段B的491520、1843200和7618560究竟是处理后像素数、image_grid_thw乘积还是实际视觉token数；
2.阶段C的Encoder-Hit实际缓存哪些张量，最小完整可复用状态是2.34MB还是包含deepstack后的9.375MB；
3.GPU、CPU和文件后端的Hit耗时是否已经包含状态加载、设备传输和后续模型前向；
4.阶段D四类轨迹的命中率分子、分母、缓存清空规则、工作流共享规则和耗时指标定义。

请直接读取最新脚本和JSON/CSV结果，提供脱敏后的关键代码、原始计数和最终判定。不要根据报告文字反推代码，不要省略不一致项，不要上传内部绝对路径、服务器地址或原始文档内容。
```

## 7. 核验后的更新规则

核验完成后，按照以下规则更新实验报告：

1.只保留最新CUDA事件实测结果，删除旧的比例估算数据；
2.输入规模字段按实际定义命名，不把像素数写成视觉token数；
3.缓存容量使用真正的最小完整可复用状态大小；
4.分层存储加速比使用单一墙钟覆盖的端到端命中时间；
5.阶段D命中率必须同时给出`hit_count/access_count`；
6.离散事件回放继续标为回放结果，不写成真实并发执行；
7.跨任务Shared Cache的收益在阶段D2完成前仍作为研究计划，不写成已验证结论。
