# 跨任务共享缓存的最小补充实验

本文档对应[跨任务共享视觉状态缓存实验：结果审计与补充测量清单](CROSS_TASK_SHARED_CACHE_RESULT_AUDIT.md)中的P0检查。两项检查均复用仓库已有的本地模型与缓存后端，不下载模型、不需要管理员权限，也不修改已有D2原始结果。

## 1. 运行前条件

在已配置`torch`、`transformers`、`Pillow`且能本地加载视觉语言模型的虚拟环境中运行。模型目录与图像均使用本地路径。

```bash
cd /path/to/agent-slack-maintenance-lab
source envs/control/bin/activate

export MODEL_DIR=/path/to/Qwen3-VL-8B-Instruct
export IMAGE_PATH=/path/to/public_docs/page_low.png
```

`IMAGE_PATH`应选择已经用于D2的页面。先用low分辨率页面完成P0核验；如时间允许，再对medium和high页面各运行一次。

## 2. P0-1：审计D2结果字段

该脚本不调用GPU，只读取现有`d2_results.json`，检查以下风险：

- `answer_consistency`是否只是未填充的占位字段；
- `vision_state_mb`是否出现0.625MB的基础视觉嵌入口径；
- CPU到GPU恢复时间是否没有记录；
- 请求合并工作负载是否真正触发了并发未命中；
- JCT字段是否具有工作流起止边界。

```bash
python scripts/audit_d2_results.py \
  --input results/d2/d2_results.json \
  --output results/d2/d2_result_audit.json
```

输出中的`report_ready=false`表示仍存在P0问题，不应直接将D2结果写入开题报告。

## 3. P0-2：验证完整状态恢复与输出一致性

该脚本使用仓库的`TransformersQwen3VLBackend`：

1.执行一次冷视觉编码；
2.将完整状态转存到CPU并恢复到GPU；
3.检查缓存对象中是否存在3组DeepStack特征；
4.比较冷路径和命中路径的logits；
5.比较确定性贪心生成的完整token序列；
6.记录命中路径的视觉编码器调用次数。

```bash
python scripts/verify_vlm_cache_correctness.py \
  --model-dir "$MODEL_DIR" \
  --image "$IMAGE_PATH" \
  --dtype float16 \
  --max-new-tokens 8 \
  --strict \
  --output results/d2/cache_correctness_low.json
```

模型使用FP16时，`--dtype float16`与D2报告一致。若已有实验明确使用BF16，可替换为`--dtype bfloat16`，但同一轮对照中的数据类型必须保持一致。

通过条件如下：

```text
passed=true
cached_deepstack_count=3
encoder_calls_on_hit=0
max_abs_logit_error<=logit_atol
greedy_output_equal=true
```

若`cached_deepstack_count`不是3，或输出不一致，必须修复D2的保存、加载或注入路径后重新运行完整D2矩阵。此时不得把`vision_state_mb=0.625MB`作为完整缓存对象的容量，也不得用当前D2数据论证无损复用。

## 4. 推荐执行顺序

1.先运行P0-1，不消耗GPU；
2.在low页面上运行P0-2；
3.确认通过后，将同一脚本分别运行在medium和high页面；
4.将三个`cache_correctness_*.json`和`d2_result_audit.json`保存至`results/d2/`；
5.只有P0通过后，才对D2的平均请求时延、跨任务复用和层级放置结果进行报告。

该最小补测不验证请求合并，也不解释SharedCPU的性能来源。前者需要专门的同步同页冷未命中压力测试，后者需要在D2执行路径中增加CPU加载、CPU到GPU恢复、状态注入和排队等待的分项计时。
