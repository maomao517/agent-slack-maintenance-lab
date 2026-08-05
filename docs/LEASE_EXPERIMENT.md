# 多模态 Agent 状态租约前置实验

## 目标

这个实验不复现 Continuum 论文中的完整数值，而是回答一个更基础的问题：

> 在相同的 Agent trace 和状态保留容量下，固定 KV TTL 是否会在长尾工具暂停或显存压力下失效；保留较小的多模态 Encoder 状态是否能形成有效的中间层？

实验完全离线，只使用 Python 标准库，不需要安装 vLLM、SGLang、CUDA 包或模型。

## 策略

```text
no_cache       每轮均重新执行 Encoder 和 prefill
encoder_lru    只保留 Encoder 输出，容量不足时按 LRU 淘汰
fixed_kv_lease 固定 TTL 保留完整 KV，近似公开版 Continuum
joint_lease    短暂停顿保留 KV，压力下将 KV 降级为 Encoder 状态
oracle         使用真实工具返回时间选择 KV 或 Encoder，作为 clairvoyant reference
```

`oracle` 的 admission 仍是贪心算法，不是数学最优上界，不能把它表述成完整离线最优解。

`retention_capacity_mb` 是专门分配给跨轮状态的容量预算，不包含正在执行请求的显存。这样可以保证所有策略具有相同的前台执行容量。

## 公司机器执行

仓库和 `envs/control` 已存在时，不要重新创建环境，也不要运行 `pip install`：

```bash
cd agent-slack-maintenance-lab
git pull --ff-only

envs/control/bin/python -m unittest discover -s tests -v

envs/control/bin/slackmaint leasebench \
  --config configs/lease_smoke.json \
  --policies all \
  --output artifacts/results/lease-smoke.json
```

然后扫描保留容量，不需要修改配置文件：

```bash
for capacity in 3600 7200 10800 14400; do
  envs/control/bin/slackmaint leasebench \
    --config configs/lease_smoke.json \
    --capacity-mb "$capacity" \
    --policies all \
    --output "artifacts/results/lease-capacity-${capacity}.json"
done
```

扫描固定 KV TTL：

```bash
for ttl in 500 1000 2000 4000 8000; do
  envs/control/bin/slackmaint leasebench \
    --config configs/lease_smoke.json \
    --fixed-kv-ttl-ms "$ttl" \
    --policies all \
    --output "artifacts/results/lease-ttl-${ttl}.json"
done
```

如果公司机器的可执行入口没有随 `git pull` 更新，仍然不需要联网安装，执行一次本地 editable install：

```bash
envs/control/bin/pip install --no-deps --no-build-isolation -e .
```

如果该命令提示缺少本地构建工具，直接使用模块入口，避免安装：

```bash
PYTHONPATH=src envs/control/bin/python -m slackmaint.cli leasebench \
  --config configs/lease_smoke.json \
  --policies all \
  --output artifacts/results/lease-smoke.json
```

整个流程不访问外网。配置和输出都不包含 prompt、图像或公司数据。

## 如何替换成真实测量

对目标 VLM 分别测量以下字段，并修改 `configs/lease_smoke.json`：

```text
model_segments_ms       每一轮 decode 的时间
tool_waits_ms           工具实际执行时间
expected_tool_waits_ms  调度器在当时可获得的估计
kv_size_mb              一轮结束时可复用 KV 的大小
encoder_size_mb         图像 Encoder 输出大小
prefill_ms              KV miss 后的 LLM prefill 时间
encoder_ms              Vision Encoder 计算时间
```

当前分级模型要求 `encoder_size_mb < kv_size_mb`。如果真实测量不满足该条件，Encoder 状态不是有效降级层，应停止这条设计。

在尚未接入 VLM 前，不应把示例配置产生的加速数字写入开题报告；它只用于验证模拟器和实验逻辑。

## 本机完整扫描

不需要 GPU 或外网即可运行 78 个受控场景：

```bash
envs/control/bin/slackmaint leasesweep \
  --config configs/lease_smoke.json \
  --output-dir artifacts/results/lease-sweep
```

扫描容量与 KV TTL 的二维交互，并分别改变工具等待长尾、等待时间预测误差、Encoder/KV 大小比、Encoder 计算成本和并发度。原始策略数据写入 `metrics.json`，派生比较写入 `comparisons.csv`，摘要写入 `summary.md`。

当前模拟结果不能替代真实 VLM 测量。下一阶段的最小真实实验方案见
[COMPANY_REAL_PROFILE_PLAN.md](COMPANY_REAL_PROFILE_PLAN.md)。

## 结果判断

重点比较：

```text
average_jct_ms / p95_jct_ms
total_recompute_ms / avoided_recompute_ms
kv_hits / encoder_hits / cache_misses
lease_expirations / demotions / forced_evictions
peak_retained_mb / retained_memory_time_mb_ms
```

继续研究至少需要出现以下现象：

1. `fixed_kv_lease` 在短暂停顿下优于 `no_cache`；
2. 长尾暂停使固定 KV 租约到期并产生 miss；
3. 容量压力下 `joint_lease` 能通过降级获得 Encoder hit；
4. `joint_lease` 在相同容量下改善 JCT 或 P95，同时不过度增加 memory-time。

若真实 profile 中 `encoder_ms` 很小，或多模态 prefix cache 已经稳定消除重复计算，应停止 Encoder 租约方向，转向长尾鲁棒 TTL 或多租户公平。

## 安全与清理

- 不使用 `sudo`；
- 不修改系统 Python、CUDA、驱动或共享目录；
- 不记录真实 prompt、工具输出或文件内容；
- 运行前后可用 `git status --short` 确认源码变化；
- 停止仓库内进程并保存所需结果后，可以删除整个仓库目录。

删除仓库会一并删除仓库内部的 `.venv/`、`envs/`、`.runtime/`、`artifacts/` 和 `models/`。它不会删除位于仓库外的 Hugging Face cache、uv/pip cache、独立 llama.cpp、外部模型目录或系统 CUDA。
