# 实验路线

## 研究问题

1. ContextPilot 的长期增量索引是否会偏离 fresh rebuild？
2. 同步更新或局部重建是否进入 Agent JCT 的关键路径？
3. Agent 工具等待窗口与真实资源余量有多少重合？
4. 工作流与资源双感知维护能否降低 JCT，并满足 freshness requirement？

## 阶段 0：框架校验

使用 `configs/smoke.json` 检查策略的基本关系：

```text
none       JCT 低，但允许 freshness violation
sync       freshness 安全，但维护进入关键路径
dual-aware 在工具等待期间完成维护，减少 blocking
```

模拟器结果只是机制校验，不能替代真实 ContextPilot 实验。

## 阶段 1：ContextPilot 静态复现

固定 ContextPilot commit，比较：

```text
NoCache
Original ContextPilot
```

记录 prefix overlap、重排时间、cache hit、TTFT 和答案质量。

## 阶段 2：索引漂移实验

对相同活跃数据比较：

```text
长期 Incremental Index
Fresh Rebuild Index
```

按 update batch 记录 reorder regret、树深、查询延迟和 prefix reuse。若差异在真实 churn 下不超过实验噪声，停止 compact 方向。

## 阶段 3：维护策略实验

所有方法保留相同的 ContextPilot reorder/dedup：

```text
No Maintenance
Synchronous Maintenance
Periodic Maintenance
Background Maintenance
Resource-Aware Maintenance
Agent-State-Aware Maintenance
Dual-Aware Maintenance
Oracle
```

共同指标：

```text
Agent JCT
P95 TTFT
Freshness lag
Stale-index query rate
Maintenance overlap ratio
Foreground interference
Maintenance backlog
```

## 阶段 4：真实 Agent 与多模态数据

先接 LoCoMo、Mem0 或 ClawTasks，再加入 PDF 页面、OCR、caption 和冻结模型 embedding。模型训练不属于本项目范围。

## Go/No-Go

只有同时观察到索引质量漂移、同步维护成本和可利用等待窗口，才继续实现完整 idle-aware maintenance。否则根据数据转向局部增量索引、多租户调度或动态向量检索。

