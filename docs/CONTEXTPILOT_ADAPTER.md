# ContextPilot Adapter 设计草案

## 固定源码审查

```text
Repository: https://github.com/EfficientContext/ContextPilot
Commit:     1fa0a143fdeda344585666648ab2b30cb7fea77f
```

该版本已经支持 OpenClaw、Hermes Agent、Mem0 和 PageIndex。与维护调度直接相关的真实入口是：

```text
ContextPilot.reorder() -> build_incremental()
SGLang/vLLM eviction callback -> remove_requests()
ContextPilot(use_gpu=False) -> build_and_schedule() / fresh rebuild
```

源码中没有现成的 `compact()`、局部重建或 copy-on-write snapshot API。因此“调度 compact”不能直接作为复现项；若 motivation 成立，需要把 fresh rebuild/局部重建实现为新增模块。

## 原则

原始 ContextPilot 必须保留为可直接运行的 baseline。新逻辑通过 adapter 和独立维护调度器接入，不修改 reorder/dedup 的语义。

## 预期映射

| 通用任务 | ContextPilot 代码位置 | 接入目标 |
|---|---|---|
| 增量插入 | `server/live_index.py` | 记录任务耗时和 generation |
| 请求移除 | `remove_requests()` | 产生局部清理任务 |
| 全量重建 | `build_and_schedule()` | 先离线测量 fresh snapshot 成本与收益 |
| 局部重建 | 不存在，待新增 | 仅在确认长期索引漂移后实现 |
| 原子发布 | 待新增 | copy-on-write snapshot 切换 |
| eviction sync | `server/http_server.py` | 保持原有行为 |

## 计划接口

```python
adapter.discover_work()
adapter.run_quantum(task_id, budget_ms)
adapter.pause(task_id)
adapter.commit(task_id, required_generation)
adapter.abort(task_id)
```

第一版 adapter 只做观测：记录 `build_incremental()`、`remove_requests()` 和 fresh rebuild 的耗时与 generation。只有确认长期索引漂移后，第二版才实现可抢占局部重建。不要同时实现 KV 搬运、多模态索引和全局调度。

本机 CPU 探针显示增量更新和删除为亚毫秒级，而 fresh rebuild 约 1.5 秒。详见 [CPU_PROBE_RESULT.md](CPU_PROBE_RESULT.md)。

## 公平比较

所有策略共享相同的：

```text
ContextPilot commit
Context Index 输入
reorder/dedup 配置
模型与 KV cache 容量
Agent trace
更新事件
```

唯一变化是维护任务何时执行、是否抢占以及何时提交。
