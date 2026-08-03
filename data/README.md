# Data Layout

大型数据、模型权重和生成缓存不提交 Git。

建议结构：

```text
data/raw/                 原始数据集
data/traces/              可提交的小型 JSON/JSONL trace
data/models/              本地模型或软链接
data/cache/               embedding、OCR 和临时缓存
```

每个可复现实验 trace 应记录数据来源、生成脚本、随机种子、许可证和校验值。

