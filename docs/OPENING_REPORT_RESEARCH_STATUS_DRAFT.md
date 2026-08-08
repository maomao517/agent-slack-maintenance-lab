# 第二章 国内外研究现状（初稿）

> 题目工作假设：面向多模态 Agent 工作流的跨轮推理状态管理与缓存优化研究
>
> 使用说明：本稿按照开题报告模板的 2.1--2.4 结构组织。正文中的参考文献编号为本稿临时编号，最终提交前需要根据学校要求统一核对作者、会议名称、出版信息、页码和 DOI。

## 2.1 LLM/VLM 推理缓存与前缀复用研究

随着大语言模型逐渐应用于长上下文问答、检索增强生成和多轮 Agent 工作流，输入处理阶段的计算开销和显存占用成为影响系统性能的重要因素。对于一个包含较长历史上下文的请求，模型需要首先执行 prefill，生成与输入 token 对应的 KV Cache；在后续解码过程中，这些 KV 状态还会持续占用 GPU 显存。因此，如何减少 KV Cache 的内存浪费、提高跨请求和跨轮次的状态复用率，已经成为 LLM 推理系统的重要研究方向。

在基础内存管理方面，PagedAttention 将 KV Cache 划分为固定大小的物理块，并借鉴操作系统分页机制建立逻辑 token 与物理存储块之间的映射，从而缓解动态序列带来的显存碎片和重复复制问题[1]。该工作构建的 vLLM 系统表明，KV Cache 不应被视为随请求临时分配的一块连续内存，而应被作为一种需要统一分配、共享和回收的系统资源。PagedAttention 主要解决的是 KV Cache 的空间管理问题，对于暂停任务的生命周期、状态优先级和跨轮恢复策略没有进行专门讨论。

在应用级复用方面，Prompt Cache 将系统提示词、模板和文档等可重复输入组织为具有明确边界的 prompt module，并缓存其 attention state，以减少重复 prefill[3]。SGLang 则进一步面向结构化语言模型程序设计了 RadixAttention，通过 Radix Tree 管理共享前缀，并将缓存复用与程序中的生成、并行和控制流结合起来[2]。这些研究说明，LLM 推理中的复用对象可以从单个请求扩展到模块、前缀和结构化程序，但其基本假设仍然是：可复用上下文的生命周期较为稳定，缓存对象主要是文本 token 产生的 KV 状态。

针对检索增强生成中的非连续上下文，CacheBlend 研究了来自不同知识块的 KV Cache 融合问题，通过选择性重算部分 token 来修正不同缓存片段直接拼接造成的位置和依赖误差[4]。Cache-Craft 进一步讨论了 RAG 场景中知识块缓存的识别、部分重算、存储和淘汰问题[17]。Mooncake 则将 KV Cache 作为分布式推理系统中的核心资源，研究了 KV 状态在计算节点和存储池之间的组织与调度[5]。这些工作推动了推理状态从单机显存对象向跨层级、跨请求的系统资源演进。LMCache 是与上述方向相关的开源系统实现，但代码仓库不能替代论文引用，本文不将其作为独立论文编号。

近期研究开始关注 Agent 和长上下文应用中的上下文复用。ContextPilot 通过上下文索引、上下文对齐和去重，识别不同 LLM 交互之间的重叠上下文，以提高 KV Cache 复用率[6]。Continuum 则更加关注 Agent 因工具调用而暂停时的状态保留和恢复问题[7]。这类研究直接说明，Agent 工作流中的一次模型调用并不一定意味着任务结束，暂停任务此前产生的推理状态可能在工具返回后再次使用。

总体来看，现有推理缓存研究已经在分页管理、前缀复用、缓存融合和异构状态存储方面形成了较为成熟的技术基础。然而，这些工作通常以单一 KV Cache 作为主要状态对象，并采用 LRU、固定容量或固定生命周期等管理方式。对于多模态 Agent 而言，视觉 Encoder 输出、视觉 token 对应的 KV 状态和文本 prefill 状态具有不同的大小、重算时间和复用价值；如何联合管理这些异构状态，仍缺少统一的研究框架。

## 2.2 多模态推理状态与多模态数据存储研究

多模态大模型在处理图像、文档和视频时，通常需要经过视觉 Encoder、模态投影和语言模型 prefill 等阶段。与纯文本模型相比，多模态输入不仅增加了视觉编码计算，还可能产生大量视觉 token 和相应的 KV Cache。当同一图像在多个 Agent 轮次中被重复引用时，系统既可以重新执行视觉 Encoder，也可以保留视觉 Encoder 输出，或者直接保留完整的 KV 状态。不同状态层级在显存占用、恢复延迟和可复用范围方面存在明显差异。

在多模态文档检索方面，VisRAG 直接将文档页面作为视觉对象进行编码和检索，避免 OCR、文本切分和版面解析造成的图表、布局等信息损失[12]。ColPali 使用视觉语言模型生成页面级多向量表示，并通过 late interaction 完成视觉文档检索[13]；Light-ColPali 进一步研究了多向量表示的压缩，以降低视觉文档索引的存储开销[14]。这些工作证明了直接保留视觉信息的价值，同时也暴露出一个系统问题：视觉表示越丰富，存储、索引、回表和推理输入的成本越高。

在多模态文档问答评测方面，MMDocRAG 将检索质量与生成质量解耦，提供文本、图片、表格和细粒度证据标注，用于分析多模态检索系统究竟能否找到支撑答案的原始证据[15]。这类 benchmark 对本文具有两方面启发：一方面，实验需要保留原始视觉输入和可追溯的证据标识；另一方面，不能只报告答案准确率，还应记录编码、检索、状态恢复和显存占用等系统指标。

对于连续视频理解，HERMES 将 KV Cache 视为一种具有不同时间和语义粒度的层次化记忆，并在不训练额外模型的条件下压缩和复用视频相关状态[8]。AVA 则从 Agentic Video Analytics 的角度，将视频分析组织为感知、检索、规划和验证等多个步骤，研究视觉语言模型驱动的长视频分析工作流[16]。M3-Agent 等工作也说明，多模态 Agent 可能需要在多个视频片段、多个时间点和多个任务轮次之间复用语义记忆。

上述研究主要面向视觉检索、流式视频或长视频分析，与本文所关注的工具驱动 Agent 暂停场景存在差异。它们重点解决的是视觉表示质量、长视频压缩、检索粒度和分析准确率，而不是工具等待期间状态应该保留多久、在显存压力下是否需要降级，以及多个暂停工作流之间如何进行公平的状态 admission。因此，已有多模态研究为本文提供了状态分层和视觉表示成本的基础，但还没有完全解决多模态 Agent 推理状态的生命周期管理问题。

## 2.3 Agent 工作流调度与状态管理研究

传统 LLM Serving 通常以单次请求为执行单位，主要优化 TTFT、ITL、吞吐和显存利用率。然而，Agent 工作流往往由多次模型调用、工具调用、环境交互和动态分支组成。一次模型调用结束后，Agent 可能进入 OCR、检索、数据库查询或网页交互阶段；工具返回后，工作流再次进入模型推理。因此，Agent 的关键性能指标不仅是单次模型延迟，还包括从工作流开始到最终完成的 JCT、工具等待期间的资源占用以及暂停任务恢复时的重复计算。

在 Agent 的程序化执行方面，Parrot 通过 Semantic Variable 暴露应用中的数据依赖和共享结构，使 Serving 系统能够识别不同 LLM 调用之间的复用关系[9]。SGLang 通过结构化语言模型程序和运行时优化，将生成、并行和缓存管理结合起来[2]。这类工作表明，Agent 工作流本身可以成为系统调度和缓存优化的执行单位，而不是将所有模型调用视为相互独立的请求。

在工作流记忆方面，Agent Workflow Memory 从历史轨迹中归纳可重复的任务流程，并在后续 Agent 执行时选择性提供这些流程[10]。该工作关注的是行为程序和任务经验的复用，说明 Agent 的复用对象已经从文本前缀扩展到更高层次的工作流结构。AgenticCache 则在具身 Agent 中缓存常见的计划转移，并通过后台 Cache Updater 异步验证和更新缓存计划[11]。这些研究体现了 Agent 系统利用后台执行和异步更新降低前台延迟的趋势，但其缓存对象主要是计划或动作，不是视觉 Encoder 和模型 KV 状态。

在长上下文和工具驱动 Agent 方面，ContextPilot 通过上下文索引、对齐和去重优化输入内容，Continuum 则研究暂停 Agent 的推理状态保存和恢复，并根据恢复成本与潜在排队延迟决定 KV 状态的保留 TTL[6][7]。这些工作已经涉及 Agent 工作流中的上下文复用和暂停状态，但仍存在进一步研究空间：第一，ContextPilot 主要围绕上下文块的索引、对齐、去重和 KV 复用展开，Continuum 主要围绕文本 Agent 暂停时的 KV TTL 展开，二者都没有在本文设定的多模态状态层面联合比较完整 KV、视觉 Encoder 输出和完全释放三种恢复路径；第二，Continuum 已经考虑了工具调用持续时间的方差，因此本文不能把“按预计时间设置 TTL”表述为首次提出的机制，较稳妥的切入点是研究多模态状态分层下的预测误差、降级策略和并发影响；第三，单个工作流的状态保留策略与多租户并发调度通常被分开处理，二者在本文中的联合效果仍需实验验证。

因此，Agent 工作流研究已经从“如何让模型完成任务”逐步发展到“如何让系统高效执行任务程序”。但对于多模态 Agent，工作流结构、工具等待时间和异构推理状态之间还缺少统一的资源管理接口。本文拟将工具等待期间的暂停状态作为系统研究对象，重点分析恢复时间、状态大小、重算成本和并发资源竞争之间的关系。

## 2.4 当前研究存在的问题

综上，国内外研究已经分别在 LLM 推理缓存、多模态表示与检索、Agent 工作流和异步执行等方面取得了较多成果，但针对“工具驱动多模态 Agent 的跨轮推理状态管理”仍存在以下不足。

**第一，现有缓存研究通常采用单一状态对象，缺少多模态状态的分层描述。** 现有 KV Cache 系统主要把 token 对应的 KV 作为缓存单元，视觉研究则更多关注视觉表示压缩或检索索引。对于多模态 Agent 的一次暂停状态，至少可以区分完整 KV Cache、视觉 Encoder 输出和完全释放三种状态。它们具有不同的空间开销和重算路径，直接使用单一 LRU 或单一 TTL 可能无法获得最优结果。

**第二，现有时间感知保留机制尚未覆盖多模态状态的分层选择。** OCR、数据库查询和网页检索具有不同的运行时间分布，同一工具也可能受到网络、负载和数据规模影响。Continuum 已经根据恢复成本和潜在排队延迟研究 KV 状态 TTL，并讨论工具调用时长方差；本文拟进一步考察：当完整 KV、视觉 Encoder 输出和完全释放具有不同的恢复成本与占用空间时，如何在预计恢复时间存在误差的情况下选择保留层级。该问题不能由固定 TTL 的单一基线直接解决，需要用真实 profile 和误差敏感性实验验证是否存在可测的收益。

**第三，状态管理和并发调度之间缺少联合目标。** 暂停任务不消耗当前 GPU 计算，但其 KV 或视觉状态可能持续占用显存；如果系统只追求单个任务的恢复速度，可能挤压活跃任务并导致整体 P95 延迟恶化。因此，状态 admission 不仅应考虑再次访问的可能性和重算成本，还应考虑状态大小、当前显存压力、工作流优先级以及多租户公平性。

**第四，已有多模态工作流系统的端到端系统指标仍不完整。** 多模态检索工作通常重点报告 Recall、nDCG 或答案准确率，Agent 工作流工作通常重点报告成功率、调用次数或总时延，而推理缓存工作重点报告 TTFT、吞吐和命中率。跨轮视觉复用率、Encoder 重算时间、KV/Encoder 状态大小、工具等待分布和工作流 JCT 尚未在同一实验框架中得到充分测量。

基于以上问题，本文拟以多轮文档图像分析 Agent 为受控应用场景，首先通过真实 VLM profile 验证视觉 Encoder 状态的大小和重算成本，再比较 NoCache、Encoder-LRU、Fixed-KV-Lease 和 Oracle 等策略。在此基础上，研究一种面向恢复时间的状态管理机制：根据工具预计返回时间、状态重算收益和当前显存压力，在完整 KV、视觉 Encoder 和完全释放之间进行选择，并分析其在工具等待长尾和多 Agent 并发条件下的适用范围。

需要说明的是，本文的研究假设仍需通过真实 Qwen3-VL 实验验证。如果视觉 Encoder 状态相对于 KV Cache 没有足够的空间优势，或跨轮视觉复用率很低，研究将收窄为成本感知的 KV/Encoder 缓存，不能预先宣称分层租约一定有效。

## 本稿候选参考文献

[1] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, et al. Efficient Memory Management for Large Language Model Serving with PagedAttention. In: Proceedings of the 29th ACM Symposium on Operating Systems Principles. 2023

[2] Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Sun, Jeff Huang, Cody Hao Yu, et al. SGLang: Efficient Execution of Structured Language Model Programs. In: Advances in Neural Information Processing Systems. 2024

[3] In Gim, Guojun Chen, Seung-seob Lee, Nikhil Sarda, Anurag Khandelwal, Lin Zhong. Prompt Cache: Modular Attention Reuse for Low-Latency Inference. In: Proceedings of the 7th Conference on Machine Learning and Systems. 2024

[4] Jiayi Yao, Hanchen Li, Yuhan Liu, Siddhant Ray, Yihua Cheng, Qizheng Zhang, et al. CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion. In: Proceedings of the 20th European Conference on Computer Systems. 2025

[5] Ruoyu Qin, Zheming Li, Weiran He, Mingxing Zhang, Yongwei Wu, et al. Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving. In: Proceedings of the 23rd USENIX Conference on File and Storage Technologies. 2025

[6] Yinsicheng Jiang, Yeqi Huang, Liang Cheng, Cheng Deng, Xuan Sun, Luo Mai. ContextPilot: Fast Long-Context Inference via Context Reuse. In: Proceedings of the 9th Conference on Machine Learning and Systems. 2026

[7] Hanchen Li, Runyuan He, Qiuyang Mang, Qizheng Zhang, Huanzhi Mao, Xiaokun Chen, et al. Continuum: Efficient and Robust Multi-Turn LLM Agent Scheduling with KV Cache Time-to-Live. arXiv preprint arXiv:2511.02230, 2026

[8] Haowei Zhang, Shudong Yang, Jinlan Fu, See-Kiong Ng, Xipeng Qiu. HERMES: KV Cache as Hierarchical Memory for Efficient Streaming Video Understanding. In: Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics. 2026

[9] Chaofan Lin, Zhenhua Han, Chengruidong Zhang, Yuqing Yang, Fan Yang, et al. Parrot: Efficient Serving of LLM-based Applications with Semantic Variable. In: Proceedings of the 18th USENIX Symposium on Operating Systems Design and Implementation. 2024

[10] Zora Zhiruo Wang, Jiayuan Mao, Daniel Fried, Graham Neubig. Agent Workflow Memory. In: Proceedings of the 42nd International Conference on Machine Learning. 2025

[11] Hojoon Kim, Yuheng Wu, Thierry Tambe. AgenticCache: Cache-Driven Asynchronous Planning for Embodied AI Agents. In: Proceedings of the 9th Conference on Machine Learning and Systems. 2026

[12] Shi Yu, Chaoyue Tang, Bokai Xu, Junbo Cui, Junhao Ran, Yukun Yan, et al. VisRAG: Vision-Based Retrieval-Augmented Generation on Multimodal Documents. In: International Conference on Learning Representations. 2025

[13] Manuel Faysse, Hugues Sibille, Tony Wu, Bilel Omrani, Gautier Viaud, Céline Hudelot, et al. ColPali: Efficient Document Retrieval with Vision Language Models. In: International Conference on Learning Representations. 2025

[14] Yubo Ma, Jinsong Li, Yuhang Zang, Xiaobao Wu, Xiaoyi Dong, Pan Zhang, et al. Towards Storage-Efficient Visual Document Retrieval: An Empirical Study on Reducing Patch-Level Embeddings. In: Findings of the Association for Computational Linguistics. 2025

[15] Kuicai Dong, Yujing Chang, Shijie Huang, Yasheng Wang, Ruiming Tang, et al. Benchmarking Retrieval-Augmented Multimodal Generation for Document Question Answering. In: Advances in Neural Information Processing Systems: Datasets and Benchmarks Track. 2025

[16] Yuxuan Yan, Shiqi Jiang, Ting Cao, Yifan Yang, Qianqian Yang, Yuanchao Shu, et al. Ava: Towards Agentic Video Analytics with Vision Language Models. In: Proceedings of the 23rd USENIX Symposium on Networked Systems Design and Implementation. 2026

[17] Shubham Agarwal, Sai Sundaresan, Subrata Mitra, Debabrata Mahapatra, Archit Gupta, Rounak Sharma, et al. Cache-Craft: Managing Chunk-Caches for Efficient Retrieval-Augmented Generation. arXiv preprint arXiv:2502.15734, 2025

> 说明：前一版中有 6 篇检索得到的综述候选，但当前没有纳入正文，也没有完成本地全文核验，因此从正式参考文献表移出。它们的待核验状态见 `OPENING_REPORT_CHINESE_SOURCES.md` 及论文普查目录。
