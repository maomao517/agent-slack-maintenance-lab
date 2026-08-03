# ContextPilot Adapter 设计草案

## 原则

原始 ContextPilot 必须保留为可直接运行的 baseline。新逻辑通过 adapter 和独立维护调度器接入，不修改 reorder/dedup 的语义。

## 预期映射

| 通用任务 | ContextPilot 代码位置 | 接入目标 |
|---|---|---|
| 增量插入 | `server/live_index.py` | 记录任务耗时和 generation |
| 请求移除 | `remove_requests()` | 产生局部清理任务 |
| 局部重建 | 待新增 | 对失衡子树重新聚类 |
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

真实 adapter 的第一版只应支持一种任务，例如局部重建。不要同时实现 KV 搬运、多模态索引和全局调度。

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

