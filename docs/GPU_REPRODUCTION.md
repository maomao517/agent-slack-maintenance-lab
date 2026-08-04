# 8xV100 复现实验与真实 Agent Trace

## 目标

这一阶段分别测量两个事实，而不是直接宣称 idle-aware 调度有效：

1. ContextPilot 动态索引更新、删除和 fresh rebuild 的真实成本；
2. OpenClaw 执行 ClawTasks 时，相邻 LLM 调用之间是否存在足够长的工具或环境等待窗口。

只有二者在时间尺度上有显著重叠，才继续实现维护调度器。

## 1. 克隆固定版本

```bash
git clone --recurse-submodules \
  https://github.com/maomao517/agent-slack-maintenance-lab.git
cd agent-slack-maintenance-lab
```

固定版本：

```text
ContextPilot  1fa0a143fdeda344585666648ab2b30cb7fea77f
ClawTasks     c44214abe151b73fd770757deca042a0a02566ca
```

## 2. 纯用户态环境

所有运行时文件都放在仓库内部。不要使用 `sudo`、系统 Python 或全局 npm 安装。

```bash
source scripts/activate_experiment.sh
```

后续每个新终端都先执行 `cd agent-slack-maintenance-lab` 和 `source scripts/activate_experiment.sh`。

要求机器已经提供 `git`、`curl`、`python3.11`、NVIDIA driver 和基本编译工具。它们缺失时应联系机器管理员，不要自行修改系统环境。

创建两个可整体删除的 Python venv：

```bash
python3.11 -m venv envs/control
envs/control/bin/python -m pip install --upgrade pip
envs/control/bin/pip install -e . aiohttp

python3.11 -m venv envs/contextpilot
envs/contextpilot/bin/python -m pip install --upgrade pip
envs/contextpilot/bin/pip install \
  -e third_party/ContextPilot "sglang==0.5.9"
envs/contextpilot/bin/python -m contextpilot.install_hook
```

OpenClaw 官方要求 Node.js 22.22.3+。将 Node 和 OpenClaw 都安装到 `.runtime/`：

```bash
NODE_VERSION=v22.22.3
case "$(uname -m)" in
  x86_64) NODE_ARCH=x64 ;;
  aarch64|arm64) NODE_ARCH=arm64 ;;
  *) echo "Unsupported architecture: $(uname -m)"; exit 1 ;;
esac

curl -fL "https://nodejs.org/dist/${NODE_VERSION}/node-${NODE_VERSION}-linux-${NODE_ARCH}.tar.xz" \
  -o .runtime/node.tar.xz
tar -xJf .runtime/node.tar.xz -C .runtime
ln -sfn "node-${NODE_VERSION}-linux-${NODE_ARCH}" .runtime/node
export PATH="$EXP_ROOT/.runtime/node/bin:$PATH"

npm_config_cache="$EXP_ROOT/.runtime/npm-cache" \
  npm install --prefix .runtime/openclaw openclaw@latest
export OPENCLAW_BIN="$EXP_ROOT/.runtime/openclaw/node_modules/.bin/openclaw"
HOME="$EXP_HOME" "$OPENCLAW_BIN" onboard
```

配置独立 OpenClaw home 中的 `sglang` provider：

```bash
envs/control/bin/python scripts/configure_openclaw.py \
  --config "$EXP_HOME/.openclaw/openclaw.json" \
  --base-url http://127.0.0.1:30100/v1 \
  --context-window 65536

HOME="$EXP_HOME" "$OPENCLAW_BIN" models list
```

模型只做推理，不训练。

## 3. 先跑 CPU 索引成本探针

```bash
envs/contextpilot/bin/python scripts/contextpilot_index_probe.py \
  --clawtasks-root third_party/ClawTasks \
  --repeats 10 \
  --active-requests 128 \
  --rebuild-every 64 \
  --output artifacts/probes/contextpilot-index.json
```

重点读取：

```text
summary.incremental_update.p95_ms
summary.eviction_remove.p95_ms
summary.fresh_rebuild.p95_ms
```

## 4. 启动 SGLang

先下载模型。能直接访问 Hugging Face 时去掉 `HF_ENDPOINT`：

```bash
export HF_ENDPOINT=https://hf-mirror.com
envs/contextpilot/bin/hf download Qwen/Qwen3-4B-Instruct-2507 \
  --local-dir models/Qwen3-4B-Instruct-2507
```

V100 不支持 BF16，建议先用两张卡、FP16 和 65536 context 做小规模验证：

```bash
CUDA_VISIBLE_DEVICES=0,1 envs/contextpilot/bin/python -m sglang.launch_server \
  --model-path models/Qwen3-4B-Instruct-2507 \
  --port 30002 \
  --host 0.0.0.0 \
  --tp-size 2 \
  --dtype float16 \
  --mem-fraction-static 0.80 \
  --context-length 65536 \
  --tool-call-parser hermes \
  --attention-backend triton
```

健康检查：

```bash
curl -sS http://127.0.0.1:30002/v1/models
```

先完成一个 scenario，再扩到 131072 context。不要一开始并行占满 8 张卡，否则无法区分调度收益与跨任务资源干扰。

## 5. 采集 Direct arm

终端 A 启动透明 trace proxy：

```bash
envs/control/bin/python scripts/openai_trace_proxy.py \
  --upstream http://127.0.0.1:30002 \
  --port 30100 \
  --output artifacts/traces/direct-llm.jsonl
```

终端 B 执行一个 ClawTasks scenario：

```bash
HOME="$EXP_HOME" envs/control/bin/python scripts/run_openclaw_trace.py \
  --clawtasks-root third_party/ClawTasks \
  --openclaw-command "$OPENCLAW_BIN" \
  --config "$EXP_HOME/.openclaw/openclaw.json" \
  --workspace "$EXP_HOME/.openclaw/workspace/contracts" \
  --arm Direct \
  --scenario s01_commercial_terms \
  --base-url http://127.0.0.1:30100/v1 \
  --output artifacts/traces/direct-turns.jsonl
```

代理不保存 prompt、response 或工具内容，只保存时间、字节数、哈希和 scenario 标签。

## 6. 采集 ContextPilot arm

先停止 Direct 组的 SGLang，启动 ContextPilot：

```bash
envs/contextpilot/bin/python -m contextpilot.server.http_server \
  --port 8771 \
  --infer-api-url http://127.0.0.1:30002 \
  --model models/Qwen3-4B-Instruct-2507 \
  --log-level info
```

再用 eviction callback 环境变量重新启动 SGLang：

```bash
CONTEXTPILOT_INDEX_URL=http://127.0.0.1:8771 \
CUDA_VISIBLE_DEVICES=0,1 envs/contextpilot/bin/python -m sglang.launch_server \
  --model-path models/Qwen3-4B-Instruct-2507 \
  --port 30002 \
  --host 0.0.0.0 \
  --tp-size 2 \
  --dtype float16 \
  --mem-fraction-static 0.80 \
  --context-length 65536 \
  --tool-call-parser hermes \
  --attention-backend triton
```

将 trace proxy 的 upstream 改成 ContextPilot：

```bash
envs/control/bin/python scripts/openai_trace_proxy.py \
  --upstream http://127.0.0.1:8771 \
  --port 30100 \
  --output artifacts/traces/cp-llm.jsonl
```

再运行相同 scenario：

```bash
HOME="$EXP_HOME" envs/control/bin/python scripts/run_openclaw_trace.py \
  --clawtasks-root third_party/ClawTasks \
  --openclaw-command "$OPENCLAW_BIN" \
  --config "$EXP_HOME/.openclaw/openclaw.json" \
  --workspace "$EXP_HOME/.openclaw/workspace/contracts" \
  --arm CP \
  --scenario s01_commercial_terms \
  --base-url http://127.0.0.1:30100/v1 \
  --output artifacts/traces/cp-turns.jsonl
```

## 7. 转换并回放真实时序

先分别回放两种维护对象：

- 增量更新：使用 `incremental_update.p95_ms`；
- compact/fresh rebuild：使用 `fresh_rebuild.p95_ms`。

将相应 P95 向上取整，填入 `--maintenance-ms`：

```bash
envs/control/bin/slackmaint convert-trace \
  --input artifacts/traces/direct-llm.jsonl \
  --arm Direct \
  --maintenance-ms 1561 \
  --output artifacts/configs/clawtasks-direct.json

envs/control/bin/slackmaint run \
  --config artifacts/configs/clawtasks-direct.json \
  --policies all \
  --output artifacts/results/clawtasks-direct.json
```

## 8. Go/No-Go

继续实现真实 idle-aware maintenance，至少需要同时观察到：

- ContextPilot 维护成本稳定高于测量噪声；
- ClawTasks 多数 scenario 内存在可重复的非零调用间隔；
- 调用间隔能够覆盖有意义比例的维护工作；
- 普通后台执行造成前台干扰，而双感知策略改善 P95 JCT；
- freshness violation 保持为 0。

如果增量更新和删除始终只有亚毫秒级，fresh rebuild 也不进入关键路径，则不应继续把“索引维护调度”作为主创新点，应转向多租户 context index、跨模态 identity/version 或缓存一致性问题。

## 9. 清理

确认 `artifacts/` 已复制到其他位置，停止 SGLang、ContextPilot 和 trace proxy，然后删除整个仓库目录即可。所有新增的 Python、Node、OpenClaw 配置、模型缓存和实验文件都位于该目录中。
