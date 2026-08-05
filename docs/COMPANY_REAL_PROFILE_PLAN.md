# 公司机器真实 VLM 前置实验方案

## 目标

公司机器阶段不实现完整的 Continuum 或 vLLM 缓存改造，只回答四个可证伪问题：

1. 多模态 Agent 是否会在后续轮次重复使用相同的视觉输入？
2. 视觉 Encoder 状态是否显著小于可复用的完整 KV 状态？
3. Encoder 和 LLM prefill 的重复计算成本是否足以影响 JCT？
4. 工具等待长尾和并发竞争是否让 `Encoder-LRU`、固定 KV TTL 出现不同的失效模式？

如果这四点不能同时得到支持，应停止复杂的 Joint-Lease 原型，转向更简单的 Encoder-LRU 成本感知淘汰或直接停止该方向。

## 环境原则

实验只使用公司机器已有的模型、推理后端和 `envs/control`，不安装 sudo 软件，不访问外网，不上传公司数据。第一轮只使用一张 V100，其他 GPU 不参与；这样可以减少变量，也不需要多卡并行框架。

先确认仓库和环境：

```bash
cd agent-slack-maintenance-lab

envs/control/bin/python -m unittest discover -s tests -v

envs/control/bin/slackmaint leasesweep \
  --config configs/lease_smoke.json \
  --output-dir artifacts/results/lease-sweep
```

如果代码已通过离线补丁导入但 CLI 入口没有更新，所有命令都可以改成：

```bash
PYTHONPATH=src envs/control/bin/python -m slackmaint.cli ...
```

## Phase 1：确认多模态模型能力

只在已有模型可接受图像输入、并且已有本地图像投影文件或等价组件时进行。先不要下载新模型。

记录以下信息：

```text
model_id / model revision
推理后端及版本
量化方式和 dtype
V100 显存容量
上下文长度
是否能单独执行视觉 Encoder
是否能读取或估算 KV Cache 状态大小
```

如果当前本地模型是纯文本模型，不能用它证明多模态 Encoder 状态的存在；可以先完成文本 KV 测量，但必须把结果标记为“非多模态替代实验”。

## Phase 2：构造受控 Agent trace

不使用真实公司文档，使用固定的本地图像和固定文本问题。建议准备三类输入：

| 输入 | 目的 |
|---|---|
| 单张中等分辨率图像 | 基准视觉状态 |
| 多张图像 | 放大 Encoder 状态和计算成本 |
| 文档页面或图表图像 | 接近真实 Agent 任务，但不引入公司数据 |

每个工作流固定 4 轮：

```text
图像输入 -> 模型回答 -> 工具等待 -> 继续追问
                 -> 工具等待 -> 继续追问
                 -> 工具等待 -> 最终回答
```

工具不需要真实联网服务，可以使用本地确定性脚本模拟等待时间。等待时间设置为：

```text
250 ms, 500 ms, 1 s, 2 s, 4 s, 8 s
```

每个条件至少重复 20 次，丢弃前 5 次 warm-up。固定模型、图像、prompt、解码参数和 batch size，避免把模型随机性混入缓存结论。

## Phase 3：测量真实 profile

每一条 trace 至少记录以下 JSONL 字段：

```json
{
  "workflow_id": "doc-01",
  "turn_id": 1,
  "image_count": 1,
  "image_pixels": 1048576,
  "input_tokens": 1200,
  "model_ms": 820,
  "tool_wait_ms": 2000,
  "prefill_ms": 230,
  "encoder_ms": 170,
  "kv_size_mb": 3200,
  "encoder_size_mb": 480,
  "peak_gpu_memory_mb": 21800
}
```

重点不是字段名字必须完全一致，而是每一项都要能回填到 `configs/lease_smoke.json` 的对应字段：

| 模拟器字段 | 真实来源 |
|---|---|
| `model_segments_ms` | 每轮模型执行的 wall-clock 时间 |
| `tool_waits_ms` | 本地工具脚本的实际等待时间 |
| `expected_tool_waits_ms` | 调度时可获得的预测值；第一轮可用固定估计 |
| `prefill_ms` | 没有复用 KV 时的 LLM prefill 时间 |
| `encoder_ms` | 没有复用视觉状态时的视觉 Encoder 时间 |
| `kv_size_mb` | 后端暴露的 KV 分配量；没有接口时按模型层数、KV head、head dim、token 数和 dtype 计算，并注明是估算 |
| `encoder_size_mb` | Encoder 输出 tensor 的元素数量乘 dtype 字节数 |

每个条件报告 median、P95 和标准差，不只报告一次运行时间。使用 `nvidia-smi` 只做峰值显存旁证，不把整张卡的占用直接当成 KV 大小。

## Phase 4：回放真实参数

把真实测量值整理成仓库配置格式，然后运行：

```bash
envs/control/bin/slackmaint leasebench \
  --config configs/real-vlm-profile.json \
  --policies no_cache,encoder_lru,fixed_kv_lease,joint_lease,oracle \
  --output artifacts/results/real-vlm-baselines.json
```

再改变以下两个变量：

```text
保留容量：真实可用显存预算的 10%、20%、30%、40%
固定 KV TTL：0.5 s、1 s、2 s、4 s、8 s
```

不要一开始做大规模参数优化。真实 profile 的作用是验证条件是否存在；只有在条件成立后，才增加更细的实验维度。

## 基线和判定标准

必须保留以下基线：

```text
NoCache
Encoder-LRU
Fixed-KV-Lease
Joint-Lease
Oracle（仅离线参考）
```

继续做系统原型的建议门槛：

1. 至少 3 个独立工作流中存在跨轮视觉复用。
2. `encoder_size_mb < kv_size_mb`，并且最好小于 KV 的 25%。
3. Encoder 重算时间占冷启动重算成本的比例不能接近 0。
4. 在至少两个容量点上，Joint 相对最佳非 Oracle 基线平均 JCT 提升至少 5%，且 P95 不恶化超过 2%。
5. 在并发增加后，Joint 不能出现明显灾难性退化；若出现退化，研究问题应改写为“并发下的联合 admission 和降级”，而不是继续调 TTL。

这些是工程决策门槛，不是论文中预先声称的结果。最终论文应报告完整失败案例。

## 三种结果对应的后续路线

### A. 真实结果支持 Joint-Lease

实现最小原型：只做跨轮状态登记、KV 到 Encoder 的降级、容量检查和命中统计，不改模型训练，不做复杂预测模型。用真实 trace 对比五个策略。

### B. Encoder-LRU 稳定优于 Joint-Lease

不应强行包装 Joint-Lease。可以把研究收窄为“多模态 Agent 的成本感知 Encoder 状态淘汰”，研究重点放在 `saved_ms / size_mb`、下一次使用时间和并发公平性，系统实现更简单。

### C. 没有稳定跨轮复用或 Encoder 状态过小

停止这条缓存租约路线，避免在开题报告中承诺不存在的系统收益；转向 Agent workflow 的调度、索引新鲜度或工具等待窗口实验。

## 预期产出

公司阶段最少应得到：

```text
real-vlm-profile.json
每个字段的 median / P95 测量表
五个策略的 JCT、P95、重算成本和命中率
容量 × TTL 二维结果
失败案例及停止/转向判断
```

这组材料足以支撑开题报告中的“研究对象可界定、问题确实存在、数据能够支撑后续分析”，即使最终不实现完整缓存系统，也能形成可信的选题决策依据。
