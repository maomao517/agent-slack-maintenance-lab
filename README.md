# Agent Slack Maintenance Lab

这是一个面向硕士开题前置验证的实验框架，用于回答：

> Agent 在等待工具、模型或环境动作时产生的窗口，能否用于执行索引更新、局部重建和 compact，并在满足 freshness 约束的同时降低工作流 JCT？

当前版本提供一个不依赖 GPU 和模型的离散事件回放器。它先验证研究假设，再逐步接入 ContextPilot 和真实推理后端，避免在 motivation 尚未成立时投入大量系统改造。

仓库还包含一个独立的多模态状态租约前置实验，用于比较固定 KV TTL、Encoder LRU 和 KV/Encoder 分级租约。该实验同样只依赖 Python 标准库，详见 [docs/LEASE_EXPERIMENT.md](docs/LEASE_EXPERIMENT.md)。

## 当前能力

- 回放多工作流的模型阶段与工具等待阶段；
- 在工作流事件上释放索引维护任务；
- 比较八种维护策略；
- 模拟版本栅栏导致的 freshness blocking；
- 输出 JCT、freshness violation、维护隐藏比例、前台干扰和 backlog；
- 提供真实系统 adapter 的稳定接口。

策略包括：

```text
none
sync
periodic
background
resource_aware
agent_aware
dual_aware
oracle
```

## 快速开始

```bash
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e .

python -m unittest discover -s tests -v
slackmaint run --config configs/smoke.json --policies all --output results/smoke.json
```

运行多模态状态租约前置实验：

```bash
slackmaint leasebench \
  --config configs/lease_smoke.json \
  --policies all \
  --output results/lease-smoke.json
```

运行完整的本机参数扫描（78 个场景、390 次策略模拟）：

```bash
slackmaint leasesweep \
  --config configs/lease_smoke.json \
  --output-dir artifacts/results/lease-sweep
```

扫描结果包括 `metrics.json`、`comparisons.csv` 和 `summary.md`。合成扫描结果及公司机器真实测量方案见
[docs/LEASE_SWEEP_RESULT.md](docs/LEASE_SWEEP_RESULT.md) 和
[docs/COMPANY_REAL_PROFILE_PLAN.md](docs/COMPANY_REAL_PROFILE_PLAN.md)。

生成新的受控 trace：

```bash
slackmaint generate \
  --output configs/generated.json \
  --seed 42 \
  --workflows 8 \
  --turns 4
```

## 实验原则

所有策略必须使用相同的：

- Agent trace；
- 维护任务及其工作量；
- freshness requirement；
- 资源容量；
- ContextPilot reorder/dedup 配置；
- 模型和并发度。

第一阶段仅比较维护策略，不修改 ContextPilot 的上下文重排算法。完整实验路线见 [docs/EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md)。

## 项目结构

```text
src/slackmaint/          离散事件模拟器、策略和 adapter 接口
configs/                 可复现实验配置
tests/                   单元测试
docs/                    实验方案与 ContextPilot 接入说明
data/                    trace 和数据说明，不提交大文件
results/                 本地实验结果，不提交生成文件
scripts/                 ContextPilot 探针与真实 Agent trace 采集工具
third_party/             固定 commit 的官方 baseline submodule
```

克隆完整 baseline：

```bash
git clone --recurse-submodules \
  https://github.com/maomao517/agent-slack-maintenance-lab.git
```

8xV100 服务器上的完整复现与 trace 采集步骤见
[docs/GPU_REPRODUCTION.md](docs/GPU_REPRODUCTION.md)。
公司环境无法安装 OpenClaw 时，使用纯 Python 工具 Agent 路线：
[docs/PYTHON_AGENT_REPRODUCTION.md](docs/PYTHON_AGENT_REPRODUCTION.md)。

## 当前边界

当前模拟器只执行已经切分成时间片的可抢占维护任务。真实 ContextPilot 接入尚未实现；必须先完成 motivation 实验，再根据结果实现局部重建和原子快照提交。
