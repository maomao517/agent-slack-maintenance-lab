# 无 OpenClaw 的纯 Python Agent 实验

## 适用场景

公司机器无法安装 Node.js 或 OpenClaw 时，使用本仓库的最小工具调用 Agent：

```text
Python Agent -> Trace Proxy -> SGLang
Python Agent -> Trace Proxy -> ContextPilot -> SGLang
```

Agent 使用真实模型自主调用 `list_files` 和 `read_file`，执行 ClawTasks 的 60 个文档分析场景。相邻 LLM 调用之间的时间包含真实工具执行和 Agent 控制开销，因此仍可用于 motivation 实验。

这条路线不是 OpenClaw 官方结果的逐数值复现；它是共享 ClawTasks workload 的受控 Agent 系统实验。

## 1. 环境

```bash
cd agent-slack-maintenance-lab
source scripts/activate_experiment.sh

python3.11 -m venv envs/control
envs/control/bin/pip install -e . aiohttp openai

python3.11 -m venv envs/contextpilot
envs/contextpilot/bin/pip install \
  -e third_party/ContextPilot "sglang==0.5.9"
envs/contextpilot/bin/python -m contextpilot.install_hook
```

不需要 Node.js、npm、OpenClaw、daemon 或 `~/.openclaw`。

## 2. Direct 组

终端 A 启动 SGLang，参数与 [GPU_REPRODUCTION.md](GPU_REPRODUCTION.md) 第 4 节一致。

终端 B：

```bash
source scripts/activate_experiment.sh
envs/control/bin/python scripts/openai_trace_proxy.py \
  --upstream http://127.0.0.1:30002 \
  --port 30100 \
  --output artifacts/traces/python-direct-llm.jsonl
```

终端 C：

```bash
source scripts/activate_experiment.sh
envs/control/bin/python scripts/run_python_agent_trace.py \
  --clawtasks-root third_party/ClawTasks \
  --workspace artifacts/python-agent-workspace \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --arm Direct --trial 0 \
  --scenario s01_commercial_terms \
  --base-url http://127.0.0.1:30100/v1 \
  --store-output \
  --output artifacts/traces/python-direct-turns.jsonl
```

## 3. ContextPilot 组

停止 Direct 组进程并重启 SGLang，保证 KV cache 初始状态一致。按照 [GPU_REPRODUCTION.md](GPU_REPRODUCTION.md) 第 6 节启动 ContextPilot 和带 `CONTEXTPILOT_INDEX_URL` 的 SGLang。

终端 C 启动 CP trace proxy：

```bash
source scripts/activate_experiment.sh
envs/control/bin/python scripts/openai_trace_proxy.py \
  --upstream http://127.0.0.1:8771 \
  --port 30100 \
  --output artifacts/traces/python-cp-llm.jsonl
```

终端 D 运行相同 Agent workload：

```bash
source scripts/activate_experiment.sh
envs/control/bin/python scripts/run_python_agent_trace.py \
  --clawtasks-root third_party/ClawTasks \
  --workspace artifacts/python-agent-workspace \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --arm CP --trial 0 \
  --scenario s01_commercial_terms \
  --base-url http://127.0.0.1:30100/v1 \
  --store-output \
  --output artifacts/traces/python-cp-turns.jsonl
```

## 4. 验收

```bash
envs/control/bin/python -c \
  'import json,sys; [json.loads(line) for line in open(sys.argv[1]) if line.strip()]' \
  artifacts/traces/python-direct-turns.jsonl

wc -l \
  artifacts/traces/python-direct-llm.jsonl \
  artifacts/traces/python-direct-turns.jsonl \
  artifacts/traces/python-cp-llm.jsonl \
  artifacts/traces/python-cp-turns.jsonl
```

每个成功 turn 应满足：

```text
error == null
llm_calls >= 1
tool_calls >= 1（明确要求读文件的 turn）
output_chars > 0
```

比较 Direct/CP 输出内容，确认结论和关键数字没有明显退化。`--store-output` 只应在 ClawTasks 合成数据上使用，真实公司数据不要开启。

## 5. 转换 trace

```bash
envs/control/bin/slackmaint convert-trace \
  --input artifacts/traces/python-direct-llm.jsonl \
  --arm Direct \
  --maintenance-ms 1561 \
  --output artifacts/configs/python-agent-direct.json

envs/control/bin/slackmaint run \
  --config artifacts/configs/python-agent-direct.json \
  --policies all \
  --output artifacts/results/python-agent-direct.json
```

单场景成功后，去掉 `--scenario` 即可运行全部 60 个文档场景。首轮建议只运行 `commercial`：

```bash
--category commercial
```
