# ContextPilot CPU 前置探针结果

## 实验边界

本实验不是端到端 Agent benchmark。它从官方 ClawTasks 的 70 个任务中提取逐轮文档引用，形成 285 个 context 操作，测量 ContextPilot 动态索引的真实函数耗时。

```text
ContextPilot commit: 1fa0a143fdeda344585666648ab2b30cb7fea77f
ClawTasks commit:    c44214abe151b73fd770757deca042a0a02566ca
Python:              3.11.15
Device:              macOS arm64, ContextPilot CPU path
```

官方 CPU 索引测试结果：`51 passed`。

## 结果

### 32 个活跃 request

| 操作 | 样本数 | 平均 | P95 | 最大值 |
|---|---:|---:|---:|---:|
| Incremental update | 277 | 0.043 ms | 0.074 ms | 0.645 ms |
| Eviction remove | 253 | 0.002 ms | 0.009 ms | 0.023 ms |
| Fresh rebuild | 17 | 1491.9 ms | 1530.4 ms | 1530.4 ms |

### 64 个活跃 request

| 操作 | 样本数 | 平均 | P95 | 最大值 |
|---|---:|---:|---:|---:|
| Incremental update | 277 | 0.041 ms | 0.061 ms | 0.094 ms |
| Eviction remove | 221 | 0.001 ms | 0.002 ms | 0.021 ms |
| Fresh rebuild | 4 | 1506.2 ms | 1561.0 ms | 1561.0 ms |

## 当前结论

### 1. 不应调度普通增量更新和删除

它们的 P95 都远低于 1 ms。即使完全隐藏，对秒级 Agent JCT 也几乎没有贡献。把“空闲窗口调度增量插入/删除”作为主创新点，现有证据不支持。

### 2. Fresh rebuild 是唯一可能成立的维护对象

Fresh rebuild 在两组实验中约为 1.5 秒，足以进入 Agent JCT。但 32 与 64 个活跃 request 的时间接近，说明当前结果很可能主要包含 CPU multiprocessing 创建和固定初始化成本，不能直接外推其规模趋势。

### 3. 必须先证明 rebuild 的必要性

“rebuild 很贵”本身不是研究动机。下一步必须比较长期 incremental index 与 fresh rebuild index，测量：

```text
prefix reuse / cache hit
reorder regret
搜索或重排延迟
树深与节点膨胀
删除后的残留结构
答案质量
```

只有长期更新导致上述指标显著退化，且 fresh rebuild 能恢复质量，才存在需要被调度的维护任务。

## 下一步判定

```text
若无索引漂移：停止 compact/rebuild 调度方向。
若有漂移但 rebuild 低于等待窗口：实现 idle-window rebuild。
若有漂移且 rebuild 长于单个窗口：研究可抢占局部重建和原子发布。
```

Linux/V100 服务器必须重新运行同一探针。Mac 结果用于筛选问题，不能作为论文最终性能数据。
