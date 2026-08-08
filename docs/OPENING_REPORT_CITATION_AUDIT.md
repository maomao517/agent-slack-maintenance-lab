# 开题报告研究现状引用审计

> 目的：检查研究现状中的主要论断是否由对应论文的摘要、引言或方法部分直接支撑。本文档区分“论文明确内容”和“基于多篇论文的综合判断”。未完成原文核验的来源不进入正式论断。

## 审计口径

- **直接证据**：论文摘要、引言、方法或实验中明确说明的目标、机制和范围。
- **综合判断**：对多篇论文的共同覆盖范围或未覆盖交叉点进行归纳，不表述为某一篇论文作者的结论。
- **待核验**：只有题名、搜索摘要、仓库或不匹配文件时，不用于支撑具体机制、数字或 limitation。

## 核心论断审计

| 正文主张 | 引用 | 原文证据位置 | 论文真正支持的内容 | 审计结论 |
|---|---:|---|---|---|
| KV Cache 动态增长会造成显存碎片和重复浪费，PagedAttention 用分页映射管理 KV | [1] | Abstract；Introduction；§3 | KV cache 随序列动态增长，低效管理会导致 fragmentation 和 redundant duplication；PagedAttention 借鉴虚拟内存分页，并支持共享 | 保留。正文只将其定位为 KV 空间管理，不延伸为 Agent 生命周期方案 |
| 结构化 LM 程序包含多次调用、控制流和结构化输入输出，SGLang 用 RadixAttention 复用 KV | [2] | Abstract；§1；§3 | SGLang 面向 structured LM programs；运行时包含 RadixAttention；论文实验覆盖 agent control、RAG、多模态模型等任务 | 保留。多模态覆盖只写“评测涉及”，不写成多模态状态管理 |
| Prompt Cache 将重复 prompt 组织为模块并复用 attention state | [3] | Abstract；§1；§2 | PML 显式表达 prompt module；预计算和复用 attention states；讨论位置一致性与模块化复用 | 保留。GPU replacement、压缩和主机传输只能作为论文讨论的后续问题，不写成已实现功能 |
| CacheBlend 处理多个非连续知识块的 KV 融合，并选择性重算 | [4] | Abstract；§1；§3--§4 | 非 prefix 的多个文本块直接复用会忽略跨块 attention；选择性重算部分 KV，并与缓存读取流水化 | 保留。限定为文本 RAG，不外推为视觉缓存 |
| Cache-Craft 管理 RAG chunk-cache，包括识别、重算、存储和淘汰 | [17] | Abstract；§1 | 针对 arbitrary-location chunk cache reuse，讨论小比例重算、存储和 eviction | 保留。编号已补正；论文仍是文本 RAG，不支撑多模态状态结论 |
| Mooncake 采用 KV-centric 分离式架构，将 CPU/DRAM/SSD 用于 KV 池，并联合考虑复用、负载和 SLO | [5] | Abstract；§1 | prefill/decode 分离；利用 CPU、DRAM、SSD 建立分布式 KV cache；KV-centric scheduler；预测式 early rejection | 保留。作者信息已按 PDF 首页修正 |
| ContextPilot 用 context index、alignment 和 dedup 提高跨交互的 context reuse | [6] | Abstract；§1 | 识别重叠 context blocks，进行 alignment 和 de-duplication，并通过 annotation 维持质量 | 保留。不得把它写成视觉 Encoder 缓存系统 |
| Continuum 处理工具调用暂停的 Agent KV 保留，并按恢复成本和潜在排队延迟设置 TTL | [7] | Abstract；§1；TTL 相关方法章节 | 工具调用导致 pause；根据 reload cost 与 potential queueing delay 决定 TTL；考虑工具时长方差 | 已收窄。本文缺口改为多模态状态分层、预测误差和联合并发治理，不再声称首次提出 estimated-time TTL |
| HERMES 将视频 KV 作为层次化记忆，并采用 training-free 管理 | [8] | Abstract；§1；§2--§3 | streaming video token 冗余；不同层对时间/语义粒度偏好不同；hierarchical KV management；training-free | 保留。限定为 streaming video understanding |
| VisRAG 直接以文档页面图像进行检索增强生成 | [12] | Abstract；Introduction | 讨论 OCR/文本解析的信息损失，直接从页面图像检索并用于 RAG | 保留。只支撑视觉文档 RAG 动机 |
| ColPali 产生页面级多向量并用 late interaction 做视觉文档检索 | [13] | Abstract；Introduction；方法章节 | 视觉语言模型生成 patch-level embeddings，使用 late interaction 检索 | 保留。存储开销结论需结合 [14]，不单独归因给 ColPali |
| Light-ColPali 研究 patch-level embedding 的压缩以降低存储 | [14] | Abstract；Introduction；方法章节 | 研究视觉文档检索中的 patch-level embeddings，并评估减少/压缩对存储和检索效果的影响 | 保留。正文不写超出论文范围的端到端 Agent 结论 |
| MMDocRAG 提供多模态文档 RAG 的细粒度证据和检索/生成分析 | [15] | Abstract；Introduction；benchmark 章节 | 提供文本、图像、表格等文档内容及细粒度证据标注，分解 retrieval 与 generation 评价 | 保留。对本文实验指标的启发属于本文设计，不是该论文提出的缓存机制 |
| AVA 将视频分析组织为多个 Agentic 步骤 | [16] | Abstract；Introduction；系统章节 | 面向 Agentic Video Analytics，组织感知、检索、规划、验证等工作流 | 保留。不能据此声称 AVA 解决暂停状态管理 |
| Parrot 用 Semantic Variable 暴露应用数据流和共享结构 | [9] | Abstract；§1；§4--§5 | 通过应用标注的 Semantic Variable 暴露请求间依赖和共享结构，支持 workflow-aware serving 优化 | 保留。作者信息已按 OSDI PDF 首页修正 |
| Agent Workflow Memory 归纳并复用可重复 workflow | [10] | Abstract；Introduction；方法章节 | 从历史轨迹归纳可复用的 workflow，并在后续执行中选择性提供 | 保留。它关注行为程序级 memory，不写成 KV/视觉状态缓存 |
| AgenticCache 维护计划转移缓存，并用后台更新器异步验证/纠错 | [11] | Abstract；Introduction；方法章节 | 面向具身 Agent 的 plan locality；缓存 plan transition；后台 Cache Updater 进行异步更新 | 保留。它不支撑视觉 Encoder/KV 生命周期结论 |

## 已修正问题

1. 原稿用综述编号 `[18][20]` 支撑 Cache-Craft/LMCache 机制，已删除该错配，并将 Cache-Craft 编为 `[17]`；由于本地 Cache-Craft PDF 首页为 arXiv 预印本，参考文献按 arXiv 预印本登记，不写成 SIGMOD 正式论文。
2. Mooncake 的作者已按本地 PDF 首页改为 Ruoyu Qin、Zheming Li、Weiran He、Mingxing Zhang、Yongwei Wu 等。
3. Parrot 的作者已按 OSDI PDF 首页改为 Chaofan Lin、Zhenhua Han、Chengruidong Zhang、Yuqing Yang、Fan Yang 等。
4. Continuum 的研究缺口已改写为“多模态状态层联合选择、时间预测误差和并发治理”，不再声称其没有 cost/queue-aware TTL。

## 仍需谨慎的表述

- “现有研究缺少统一框架”“尚未充分测量”等是跨论文综合判断，最终正文应保留“从本文选取的代表性工作看”或“尚缺少将这些因素放在同一框架中评估的研究”等限定语。
- Continuum 当前本地文件是 arXiv 预印本，参考文献中按预印本写，不应写成已经核验的正式会议版本。
- 2026 年论文的正式出版信息、页码和最终作者列表应在学校提交前再查一次官方 proceedings；本审计只确认本地 PDF 首页和现有登记信息。
- 中文文献尚未纳入本稿，原因是已有候选只有搜索摘要，未完成题名、摘要、出处和正文相关性的核验。
