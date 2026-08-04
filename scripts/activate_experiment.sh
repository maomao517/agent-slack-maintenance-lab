#!/usr/bin/env bash

_SLACKMAINT_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export EXP_ROOT="$(cd "${_SLACKMAINT_SCRIPT_DIR}/.." && pwd)"
export EXP_HOME="${EXP_ROOT}/.runtime/home"
export PIP_CACHE_DIR="${EXP_ROOT}/.runtime/pip-cache"
export HF_HOME="${EXP_ROOT}/.runtime/huggingface"
export TORCH_HOME="${EXP_ROOT}/.runtime/torch"
export XDG_CACHE_HOME="${EXP_ROOT}/.runtime/cache"
export OPENCLAW_BIN="${EXP_ROOT}/.runtime/openclaw/node_modules/.bin/openclaw"

if [[ -d "${EXP_ROOT}/.runtime/node/bin" ]]; then
  export PATH="${EXP_ROOT}/.runtime/node/bin:${PATH}"
fi

mkdir -p \
  "${EXP_HOME}" \
  "${PIP_CACHE_DIR}" \
  "${HF_HOME}" \
  "${TORCH_HOME}" \
  "${XDG_CACHE_HOME}" \
  "${EXP_ROOT}/envs" \
  "${EXP_ROOT}/models" \
  "${EXP_ROOT}/artifacts"

unset _SLACKMAINT_SCRIPT_DIR
