# 本机租约参数扫描结果

## 实验范围

本机运行了 `leasesweep`：

```text
78 个场景
390 次策略模拟
5 个策略：NoCache、Encoder-LRU、Fixed-KV-Lease、Joint-Lease、Oracle
```

扫描变量包括：

```text
保留容量：1800 / 2400 / 3600 / 5400 / 7200 / 10800 / 14400 MB
KV TTL：250 / 500 / 1000 / 2000 / 4000 / 8000 ms
容量 × TTL 二维组合
工具等待时间：0.5 / 1 / 2 / 4 倍
Encoder/KV 大小比：0.05 / 0.10 / 0.15 / 0.25 / 0.40 / 0.70
Encoder 计算成本：0.25 / 0.5 / 1 / 2 / 4 倍
并发工作流：1 / 2 / 4 倍
等待时间预测：真实等待时间的 0.25 / 0.5 / 1 / 2 倍
```

原始结果位于本地生成目录：

```text
artifacts/results/lease-sweep/metrics.json
artifacts/results/lease-sweep/comparisons.csv
artifacts/results/lease-sweep/summary.md
```

这些文件被 `.gitignore` 忽略，避免将实验产物和未来可能包含的真实 trace 提交到仓库。

在 78 个场景中，按平均 JCT 统计的非 Oracle 最优策略为：

```text
Encoder-LRU：51 个场景
Fixed-KV-Lease：6 个场景
Joint-Lease：21 个场景
```

Joint-Lease 同时改善最佳基线的平均 JCT 和 P95 的场景只有 17 个。

## 关键结果

默认配置下：

| 策略 | 平均 JCT | P95 JCT | 重算成本 |
|---|---:|---:|---:|
| NoCache | 11657.5 ms | 13240 ms | 7200 ms |
| Encoder-LRU | 10065 ms | 12100 ms | 4920 ms |
| Fixed-KV-Lease | 10702.5 ms | 13780 ms | 5240 ms |
| Joint-Lease | 10140 ms | 12700 ms | 4580 ms |
| Oracle | 9675 ms | 12090 ms | 4140 ms |

默认场景中 `Encoder-LRU` 的平均 JCT 和 P95 都优于 `Joint-Lease`。`Joint-Lease` 相比 Fixed-KV-Lease 减少了重算，但这没有转化为相对 Encoder-LRU 的端到端优势。

代表性容量结果：

| 保留容量 | Joint 相对最佳非 Oracle 基线的平均 JCT | P95 | 最佳基线 |
|---:|---:|---:|---|
| 1800 MB | +5.08% | +5.98% | Encoder-LRU |
| 2400 MB | 0.00% | 0.00% | Encoder-LRU |
| 3600 MB | -8.67% | -6.20% | Encoder-LRU |
| 7200 MB | -0.75% | -4.96% | Encoder-LRU |
| 10800 MB | +3.76% | +4.03% | Fixed-KV-Lease |
| 14400 MB | +4.67% | +2.43% | Fixed-KV-Lease |

这里的正值表示 Joint-Lease 更好，负值表示更差。

1800 MB 场景需要单独解释：所有完整 KV 都大于该容量，因此 Joint-Lease 会把 KV 全部降级为 Encoder 状态。它相对 Encoder-LRU 的收益来自保存收益密度驱动的 Encoder 淘汰，而不是 KV/Encoder 两级保留本身。真正同时产生 KV hit 和 Encoder hit 的代表性收益出现在 10800 MB 和 14400 MB 场景。

其他敏感性结果：

| 场景 | 结果 |
|---|---|
| 工具等待 0.5 倍 | Joint 平均 JCT 相对最佳基线 +2.78%，P95 +9.69% |
| 工具等待 2 倍 | Joint 平均 JCT +2.88%，P95 +0.68% |
| Encoder/KV 比为 0.05 | 平均 JCT +1.12%，但 P95 -9.50% |
| Encoder 成本 4 倍 | Joint 平均 JCT -5.43%，Encoder-LRU 更好 |
| 并发 2 倍 | Joint 平均 JCT +2.74%，P95 -0.54% |
| 并发 4 倍 | Joint 平均 JCT -16.06%，Fixed-KV-Lease 更好 |
| 等待时间预测为真实值 | Joint 平均 JCT +3.87% |
| 等待时间预测只有真实值的 0.25/0.5 倍 | Joint 没有超过 Encoder-LRU |

完整数据以 `comparisons.csv` 为准。

## 对选题的含义

这轮实验支持的是一个**条件性研究问题**，不是“Joint-Lease 在所有工作负载上都更优”：

> 在多模态 Agent 的跨轮状态保留中，如何在容量压力、工具等待长尾和并发竞争下联合选择 KV 状态与 Encoder 状态，避免固定 KV 租约和单一 Encoder-LRU 的系统性失效？

当前结果暴露了三个真实的研究缺口：

1. `Encoder-LRU` 是必须认真对比的强基线，单纯增加 KV/Encoder 两级存储并不自动带来收益。
2. 等待时间预测错误会使 Joint-Lease 选择错误层级；不能把大量手工 TTL 调参当作主要创新。
3. 并发升高后，当前基于保存收益密度的贪心淘汰会失败，说明 admission、降级和调度需要联合设计。

因此，下一步不应直接实现完整推理引擎改造，而应先确认真实 VLM 是否同时满足：

```text
Encoder 状态显著小于 KV 状态；
Encoder 重算成本不可忽略；
Agent 跨轮复用率足够高；
等待时间具有长尾且难以准确预测；
并发竞争会使单一 LRU 或固定 KV TTL 产生明显损失。
```

## 结论边界

本报告中的数字全部来自合成 profile，只能证明模拟器和实验逻辑能够产生可解释的权衡，不能直接写成真实 GPU 性能结论。公司机器阶段只需要先测量这些 profile 参数，再把真实参数回放到同一套模拟器中。
