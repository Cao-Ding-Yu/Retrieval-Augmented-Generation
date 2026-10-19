# 学习指南：从零读懂这个 RAG 项目

> 这份文档不是 API 手册，而是**阅读路线**。目标：让你从"代码能跑但我看不懂"变成"我能改、能解释、能回答面试官"。
>
> 预计投入：跑通约 1 小时 → 读懂 1~2 天 → 能讲清楚 3~5 天。

---

## 0. 先看这一节（2 分钟）

学习方法的核心只有一句话：

> **不要按文件顺序读代码，要跟着一条数据走完全程。**

这个项目只有两个方向的数据流：

```
入库方向：  PDF/TXT ──► 分块 ──► 向量 ──► Milvus
检索方向：  问题 ──► 向量 ──► 双分支召回 ──► RRF 融合 ──► 交叉编码器重排 ──► 提示词 ──► 流式回答
```

把这两条链路各走通一遍，整个项目就通了。**其余文件都是这两条链路上的零件**，
不在链路上的（如 `logging_setup.py`）可以先跳过。

---

## 1. 一张图：一次提问到底发生了什么

这是全文最重要的一张表。每读一个文件，都回到这张表上找它的位置。

| # | 做什么 | 代码位置 |
|---|---|---|
| 1 | 浏览器 POST `/chat`，读取流式响应 | [web/index.html](<F:\PyCharm 2025.1.1.1\projects\Retrieval-Augmented Generation\web\index.html>) 的 `ask()` |
| 2 | FastAPI 进入路由，启动 SSE 生成器 | [routes_chat.py:136](<F:\PyCharm 2025.1.1.1\projects\Retrieval-Augmented Generation\src\rag_app\api\routes_chat.py>) `event_stream()` |
| 3 | 检索入口（异步） | [hybrid.py:86](<F:\PyCharm 2025.1.1.1\projects\Retrieval-Augmented Generation\src\rag_app\retrieval\hybrid.py>) `retrieve()` |
| 3.1 | 查询向量化 | `hybrid.py:107` `_retrieve_sync()` |
| 3.2 | 双分支召回（稠密 + BM25） | [milvus_store.py:509](<F:\PyCharm 2025.1.1.1\projects\Retrieval-Augmented Generation\src\rag_app\retrieval\milvus_store.py>) `hybrid_search()` |
| 3.3 | RRF 融合 | [milvus_store.py:119](<F:\PyCharm 2025.1.1.1\projects\Retrieval-Augmented Generation\src\rag_app\retrieval\milvus_store.py>) `reciprocal_rank_fusion()` |
| 3.4 | 交叉编码器重排 | [rerank.py:395](<F:\PyCharm 2025.1.1.1\projects\Retrieval-Augmented Generation\src\rag_app\retrieval\rerank.py>) `CrossEncoderReranker.rerank()` |
| 4 | 组装提示词 | [prompts.py:121](<F:\PyCharm 2025.1.1.1\projects\Retrieval-Augmented Generation\src\rag_app\llm\prompts.py>) `build_messages()` |
| 5 | 流式生成 | [chat.py:228](<F:\PyCharm 2025.1.1.1\projects\Retrieval-Augmented Generation\src\rag_app\llm\chat.py>) `RAGChat.astream_answer()` |
| 5.1 | **剥离推理模型的思考块** | [chat.py:66](<F:\PyCharm 2025.1.1.1\projects\Retrieval-Augmented Generation\src\rag_app\llm\chat.py>) `ThinkingStripper`（流式状态机） |
| 6 | 编码 SSE 帧 | [sse.py:48](<F:\PyCharm 2025.1.1.1\projects\Retrieval-Augmented Generation\src\rag_app\api\sse.py>) `sse_frame()` |
| 7 | 浏览器解析并渲染 | `web/index.html` 的 `parseFrame()` / `renderSources()` |

**注意第 3 步到第 3.4 步是同步阻塞的**（Ollama 与 pymilvus 都是同步库），
所以用 `asyncio.to_thread` 丢到线程池。这是本项目第一个值得理解的设计点。

---

## 2. 第一步：先跑起来，用眼睛看（约 1 小时）

**不要先读代码。** 先让它动起来，建立"它到底在干什么"的直觉。读代码时你才有参照物。

### 2.1 环境自检

```powershell
# 1) 虚拟环境
.\.venv\Scripts\Activate.ps1

# 2) 确认依赖装好（配置项全部在 .env 里，见 README 的「配置说明」一节）
python -c "import rag_app, pymilvus, sentence_transformers; print('imports ok')"

# 3) Milvus 与 Ollama 是否活着
rag-app-cli stats
```

若 `rag-app-cli` 命令不存在（未 `pip install -e .`），用模块方式调用：

```powershell
$env:PYTHONPATH="src"; python -m rag_app.cli stats
```

**预期**：打印出集合名、Milvus 版本、BM25 模式、实体总数、当前检索参数。
**这一步的收获**：你会知道自己的系统当前跑在**服务端 BM25 + 服务端融合**模式。
（原因见 §5.1 与 §8）

### 2.2 建库

```powershell
rag-app-cli ingest ./data
```

**预期**：输出 `扫描文件 / 成功入库 / 写入分块` 三个数字。

**观察点**：注意日志里的 `chunk_size` 与 `overlap`。
分块是检索质量的天花板，后面你会反复回来调它。

### 2.3 关键一步：用 `/search` 看检索（不经过 LLM）

这是本指南**最有价值的一步**。先确认检索对不对，再去看生成。

```powershell
rag-app-cli search "十五五时期的指导思想有哪些"
```

**预期输出**每条包含：来源文件、分块序号、命中分支、分数、前 160 字正文。

**这一步的收获**：

- 看到 `命中分支` 是 `稠密#N`、`稀疏#N` 还是两者都有；
- 看到 `RRF=...` 或 `重排=...` 分数；
- 若显示「★ 稠密与稀疏同时命中」，说明该结果同时满足语义与关键词，是最可信的。

> 如果这里的结果就已经不相关，**不要去看 LLM 的问题**。
> 检索错了，模型再强也救不回来。先调 `CHUNK_SIZE` 与 `TOP_K`。

### 2.4 再看流式生成

```powershell
rag-app
# 浏览器打开 http://127.0.0.1:8000/
```

或者用命令行直接看 SSE 原始帧（**强烈推荐至少做一次**）：

```powershell
curl.exe -N -X POST http://127.0.0.1:8000/chat `
  -H "Content-Type: application/json" `
  -d "{\"message\":\"RRF 是怎么工作的\"}"
```

**预期**：先出现一个 `event: sources` 帧（整批引用），然后是多个 `event: token` 帧（逐字），
最后 `event: done`。

**这一步的收获**：你会亲眼看到"流式"不是抽象的 —— 它就是一个个 `data:` 文本块。

### 2.5 三个观察实验（做完再往下读）

| 实验 | 怎么做 | 观察什么 | 说明什么 |
|---|---|---|---|
| A | 问一个知识库里**没有**的东西 | 直接返回兜底文案，**没有调用模型** | 检索为空时短路，避免模型硬编答案 |
| B | 看 `/health` | `ready`、`bm25_mode`、`fusion_mode`、`reranker` | 系统当前实际生效的配置 |
| C | 停掉 Milvus 再访问 `/chat` | 返回 503 且带具体原因 | 优雅降级：服务不崩，原因可见 |

---

## 3. 第二步：跟着一条数据走完全程（核心方法）

### 3.1 入库链路：文件 → Milvus

只有 4 个函数，按顺序读：

| 顺序 | 位置 | 输入 → 输出 | 关键决策 |
|---|---|---|---|
| 1 | [loader.py:31](<F:\PyCharm 2025.1.1.1\projects\Retrieval-Augmented Generation\src\rag_app\ingest\loader.py>) `compute_doc_id()` | 路径 → sha1 | 为什么不用 uuid？→ 幂等更新（见 §6） |
| 2 | `loader.py:156` `load_directory()` | 目录 → `list[Document]` | PDF 按**页**拆，便于引用定位到"第 7 页" |
| 3 | [splitter.py:67](<F:\PyCharm 2025.1.1.1\projects\Retrieval-Augmented Generation\src\rag_app\ingest\splitter.py>) `split_documents()` | 文档 → 分块 | 分隔符里有**中文标点**（见 §8 陷阱 1） |
| 4 | [service.py:124](<F:\PyCharm 2025.1.1.1\projects\Retrieval-Augmented Generation\src\rag_app\ingest\service.py>) `_ingest_one_document()` | 分块 → 写入 | **先删后插** + 分批向量化 |

**数据形状变化**（这条线要能背下来）：

```
Path
  └─► Document(page_content="...", metadata={doc_id, source, page})
        └─► Document  (切分后增加 metadata.chunk_index)
              └─► dict{text, doc_id, source, chunk_index, dense:[1024 floats]}
                    └─► Milvus 行（服务端自动补 sparse 字段）
```

**注意第 4 步的 `dense` 字段**：稀疏向量 `sparse` **不在这里出现**。
服务端 BM25 模式下它由 Milvus 的 Function 自动生成 —— 这是本项目"少一个模型"的关键。

### 3.2 检索链路：问题 → 上下文

| 顺序 | 位置 | 说明 |
|---|---|---|
| 1 | `hybrid.py:86` `retrieve()` | 异步入口，空查询直接返回 |
| 2 | `hybrid.py:107` `_retrieve_sync()` | 真正的编排；**整条链路在线程池里跑** |
| 3 | `milvus_store.py:509` `hybrid_search()` | 一次调用交给 Milvus：稠密 + BM25 + 服务端 RRF |
| 4 | `hybrid.py:233` `_apply_rerank()` | 交叉编码器精排，取 `final_top_k` |

**数据形状变化**：

```
str "十五五..."
  └─► list[float] × 1024            (embed_query)
        └─► list[RawHit]             (Milvus 融合后的命中，含 text/score)
              └─► list[RetrievedChunk]  (补上溯源与打分字段)
                    └─► 按重排分数排序后取前 final_top_k 条
```

**`RetrievedChunk` 值得单独看**（[models.py:16](<F:\PyCharm 2025.1.1.1\projects\Retrieval-Augmented Generation\src\rag_app\models.py>)）：
它同时携带 `dense_rank` / `sparse_rank` / `rrf_score` / `rerank_score`。
**这不是冗余，而是"可解释检索"的实现** —— 前端能据此告诉用户"这条为什么排在前面"。

### 3.3 生成链路：上下文 → 用户看到的话

| 顺序 | 位置 | 说明 |
|---|---|---|
| 1 | `routes_chat.py:136` `event_stream()` | 先发 `sources`，再进模型 |
| 2 | `prompts.py:71` `format_context()` | 给上下文编号 `[1] [2]`，超长从**末尾**截断 |
| 3 | `prompts.py:121` `build_messages()` | 拼成 `[System, Human]` 两条消息 |
| 4 | `chat.py:228` `astream_answer()` | 逐 token 产出 |
| 5 | `chat.py:66` `ThinkingStripper` | 边流边剥离思考块（推理模型必需） |
| 5 | `sse.py:48` `sse_frame()` | 编码成 SSE 帧 |

**为什么上下文放 Human 而不是 System**：本地 7B 模型对 system 的遵循度偏弱，
把"资料 + 问题"放一起关联更紧。这类小决策在 [prompts.py 顶部注释](<F:\PyCharm 2025.1.1.1\projects\Retrieval-Augmented Generation\src\rag_app\llm\prompts.py>) 里有完整理由。

---

## 4. 第三步：按依赖顺序精读（6 层）

层与层之间是**依赖关系**：读第 N 层时，它依赖的都在前面读过了。

### Layer 0 — 词汇表（10 分钟）

- [models.py](<F:\PyCharm 2025.1.1.1\projects\Retrieval-Augmented Generation\src\rag_app\models.py>)：`RetrievedChunk`、`IngestReport`
- [config.py:19](<F:\PyCharm 2025.1.1.1\projects\Retrieval-Augmented Generation\src\rag_app\config.py>)：`Settings`

**读完应能回答**：一个分块被检索到时，随它一起流动的信息有哪些？

### Layer 1 — Milvus 结构（重要，30 分钟）

- [schema.py:120](<F:\PyCharm 2025.1.1.1\projects\Retrieval-Augmented Generation\src\rag_app\schema.py>) `build_schema()`
- `schema.py:108` `supports_server_bm25()`（版本要求）、`schema.py:227` `sparse_index_params()`

**这是整个混合检索的地基。** 重点看三处：

1. 字段 `text` 上有 `enable_analyzer=True` —— BM25 的前提；
2. Schema 上注册了 BM25 `Function`，`text → sparse` 自动映射；
3. 稀疏索引的 `metric_type` 固定为 `BM25` —— **写错会直接报错**。

### Layer 2 — 入库（40 分钟）

`ingest/loader.py` → `ingest/splitter.py` → `ingest/service.py`（顺序同 §3.1）

**读完应能回答**：同一个文件入库两次会发生什么？为什么？

### Layer 3 — RRF 融合（30 分钟，性价比最高）

[milvus_store.py:119](<F:\PyCharm 2025.1.1.1\projects\Retrieval-Augmented Generation\src\rag_app\retrieval\milvus_store.py>) `reciprocal_rank_fusion()`

**建议从这一段开始读，而不是从文件开头。** 理由：

- 它是**纯函数**，没有外部依赖，最容易懂；
- 它是检索质量的核心（决定哪些内容进 LLM）；
- 它被完整单测覆盖，配合 [test_rrf.py](<F:\PyCharm 2025.1.1.1\projects\Retrieval-Augmented Generation\tests\test_rrf.py>) 读，测试就是最好的规格说明；
- 它就在文件顶部，前面那一大段 RRF 注释把"为什么不用分数加权"讲透了。

**读完应能回答**：为什么 `2/(k+2) > 1/(k+1)` 这个不等式是 RRF 的理论基础？

### Layer 4 — Milvus 存储封装（1 小时）

[milvus_store.py](<F:\PyCharm 2025.1.1.1\projects\Retrieval-Augmented Generation\src\rag_app\retrieval\milvus_store.py>) 是最大的文件，**按功能分块读**，不要顺序读：

| 块 | 行号 | 内容 |
|---|---|---|
| A RRF 融合 | 102–196 | `reciprocal_rank_fusion` / `fuse_two_branch`（纯函数，建议先读） |
| B 生命周期 | 212–362 | `connect`（含版本校验）/ `ensure_collection` / `_validate_dimension` |
| C 写入 | 375–448 | `insert` / `delete_by_doc_id` |
| D 检索 | 450–589 | `dense_search` / `sparse_search` / `hybrid_search` / `search_and_fuse_branches` |
| E 结果解析 | 600+ | `_flatten_results` |

**读完应能回答**：为什么 `insert()` 里要 `flush()`？代价是什么？

### Layer 5 — 重排与编排（1 小时）

- [rerank.py](<F:\PyCharm 2025.1.1.1\projects\Retrieval-Augmented Generation\src\rag_app\retrieval\rerank.py>)：先看 `Reranker` Protocol（L47），再看两个实现；`load()`（L274）的三级离线加载值得细读
- [hybrid.py](<F:\PyCharm 2025.1.1.1\projects\Retrieval-Augmented Generation\src\rag_app\retrieval\hybrid.py>)：`_retrieve_sync()`（L107）串起了整条链路，是理解编排的最佳入口

**读完应能回答**：为什么重排器加载要有"本地目录 → 离线 → 下载"三级降级？

### Layer 6 — 服务与前端（1 小时）

- [deps.py:121](<F:\PyCharm 2025.1.1.1\projects\Retrieval-Augmented Generation\src\rag_app\deps.py>) `build_context()` —— 唯一知道"所有零件怎么拼"的地方
- `deps.py:93` `_apply_hf_env()` —— 必须在任何 HF 相关 import **之前**执行（见 §8 陷阱 8）
- [app.py:66](<F:\PyCharm 2025.1.1.1\projects\Retrieval-Augmented Generation\src\rag_app\api\app.py>) `create_app()` + `lifespan`
- [routes_chat.py](<F:\PyCharm 2025.1.1.1\projects\Retrieval-Augmented Generation\src\rag_app\api\routes_chat.py>)：三个接口
- [web/index.html](<F:\PyCharm 2025.1.1.1\projects\Retrieval-Augmented Generation\web\index.html>)：只读 `ask()` 与 `parseFrame()` 两个函数

**读完应能回答**：为什么重排模型必须在 `lifespan` 里加载，而不是每个请求加载？

---

## 5. 五个核心概念（不补这些，读了也白读）

### 5.1 稀疏检索 / BM25

- **一句话**：按词频与逆文档频率给文档打分，靠**精确词匹配**。
- **本项目体现**：`schema.py` 的 BM25 `Function`（L120）+ `milvus_store.sparse_search()`（L473）
- **为什么需要**：稠密向量漏精确标识符（错误码、版本号、专有名词）。
- **关键点**：稀疏向量由服务端 Function 在**写入时**自动生成，查询时把原始文本
  连同 `metric_type="BM25"` 交给 Milvus，分词与打分全在服务端完成。
  所以**应用层完全不碰稀疏编码**，也不需要第二个模型。
- **自测**：为什么 `TEXT_MATCH` 不能当词法分支？（答：它只做布尔匹配，没有排名，
  而 RRF 的输入必须是排名）

### 5.2 RRF（倒数排名融合）

- **一句话**：`RRF(d) = Σ 1/(k + rank)`，只用**名次**，不用分数。
- **本项目体现**：两条路径都提供 ——
  `milvus_store.reciprocal_rank_fusion()`（本地实现，L119）
  与 Milvus 内置 `RRFRanker`（服务端实现，`hybrid_search` 内）。
  由 `FUSION_MODE` 决定用哪个，**两者公式相同、排序结果一致**。
- **为什么需要**：余弦相似度（-1~1）与 BM25 分数（无上界）量纲不同，加权前必须归一化，
  而归一化不稳定。RRF 绕开了分数校准。
- **自测**：`k` 调大会发生什么？为什么"两路都排第二"能胜过"单路排第一"？

### 5.3 双编码器 vs 交叉编码器

- **双编码器**（稠密检索）：查询与文档**分别**编码，可预建索引 → 快，但两者无交互。
- **交叉编码器**（重排）：`(查询, 文档)` **拼在一起**编码 → 准，但无法预计算，N 个候选跑 N 次。
- **本项目体现**：`OllamaEmbeddings`（双）+ `sentence_transformers.CrossEncoder`（交叉）
- **所以**：两阶段 —— 先便宜地召回 top-20~30，再对这些候选精排。
- **自测**：为什么不能直接用交叉编码器检索全库？

### 5.4 ANN / HNSW

- **一句话**：近似最近邻检索，用图结构换取"不遍历全库"的速度。
- **本项目体现**：`schema.dense_index_params()` 里的 `HNSW` + `M=16` + `efConstruction=200`
- **参数直觉**：`M` 越大召回越高、内存越多；`efConstruction` 越大索引质量越好、构建越慢；
  检索期的 `ef`（`milvus_store.HNSW_SEARCH_EF`）越大召回越高、延迟越高。
- **自测**：为什么稠密索引用 COSINE 而不是 L2？

### 5.5 SSE 流式

- **一句话**：基于普通 HTTP 的单向服务端推送，一帧 = `event:` 行 + `data:` 行 + 空行。
- **本项目体现**：`api/sse.py`（编码）+ `web/index.html` 的 `parseFrame()`（解码）
- **为什么不用 WebSocket**：本场景是单向的，SSE 更简单且天然穿过代理。
- **为什么不用 `EventSource`**：它只支持 GET、不能带请求体，长问题会被 URL 长度截断。
  所以前端用 `fetch` + `ReadableStream` **手写** SSE 解析。
- **自测**：为什么响应头要有 `X-Accel-Buffering: no`？

### 5.6 推理模型的思考块

- **一句话**：`deepseek-r1` 这类推理模型会在正文前输出一段被标签包裹的推理链，
  必须剥离后再给用户。
- **本项目体现**：`llm/chat.py` 的 `ThinkingStripper`（L66）与 `strip_thinking()`（L196）
- **为什么不能用正则**：流式的 token 边界与标签边界无关，结束标签可能被切成两半到达。
  对单个 chunk 做正则必然漏掉被切断的标签。
- **正确做法**：状态机 —— 只输出确定安全的部分，把可能构成"半个标签"的尾巴留在缓冲区。
- **自测**：为什么缓冲区最多积压 `len(标签)-1` 个字符而不是清空？

---

## 6. 关键设计决策速查（面试就考这些）

| 决策 | 备选方案 | 为什么选它 | 代价 |
|---|---|---|---|
| 混合检索（稠密+BM25） | 只用稠密 | 互补：语义 vs 精确词 | 多一次检索 |
| RRF 融合 | 分数加权 | 免归一化、鲁棒 | 丢弃了分数幅度信息 |
| 两阶段重排 | 只召回 | 精度显著提升 | 延迟随候选数线性增长 |
| BM25 用 Milvus 服务端 Function | 独立稀疏模型（SPLADE 等）／客户端 BM25 编码 | 少一个模型、索引同库、应用层不碰稀疏编码 | **硬性要求 Milvus 2.5+** |
| doc_id = 路径 sha1 | uuid / 自增 | **幂等更新** | 文件移动后视为新文档 |
| 入库先删后插 | 直接追加 | 文件更新后不留旧版本 | 每次多一次删除 |
| `build_context` 不抛异常 | 启动即崩 | `/health` 能报告原因，可排障 | 需处理 `ready=False` |
| 重排器 lifespan 单例 | 每请求加载 | 避免数秒冷启动 | 启动变慢、常驻 2GB |
| 分块重叠 120 字符 | 无重叠 | 避免语义被切断 | 存储与计算冗余 |
| `score` 与 `rrf_score` 分开 | 只存一个 | 可做"融合 vs 精排"消融对比 | 字段变多 |
| 重排模型**离线优先**加载 | 默认联网校验 | 缓存命中时零网络，规避 etag 校验超时 | 缓存缺失时需显式允许下载 |
| 思考块剥离用**状态机** | 正则替换 | 流式下标签会被切断，正则必然漏 | 实现比正则复杂、需缓冲几个字符 |
| 思考标签用 `chr()` 拼装 | 直接写字面量 | 防止尖括号在文本往返中被改写成 Unicode 近似字符 | 可读性略降 |
| 只支持 Milvus 2.5+，不做 2.4 回退 | 兼容 2.4 客户端 BM25 | 少一套编码路径、少 2 个配置项、代码更易读 | 版本不足时启动即失败 |
| 融合策略改为**可自由配置** | 由 BM25 能力自动修正 | 两条路径都成立，行为可预期、无隐藏逻辑 | 用户需自己选（默认可解释性更好的 client） |

---

## 7. 动手实验（改一行，看变化）

> 建议每个实验前记下当前输出，方便对比。改配置后需重启服务。

| # | 改什么 | 预期现象 | 学到什么 |
|---|---|---|---|
| 1 | `RERANK_ENABLED=false` | `/search` 里分数从 `重排=...` 变 `RRF=...`，**排序可能变化** | 重排到底改变了什么 |
| 2 | `RRF_K=1`（从 60） | 头部结果被显著强化 | `k` 的语义 |
| 3 | `RERANK_TOP_N=3`（从 30） | 响应更快，但可能漏掉好结果 | 重排的延迟-质量权衡 |
| 4 | `FUSION_MODE` 在 `client` / `server` 之间切换 | `/search` 结果的 `dense_rank` / `sparse_rank` 从"有"变"无"，排序基本一致 | 两种融合算的是同一个 RRF 公式 |
| 5 | `EMBEDDING_DIM=768` | 启动即报清晰的维度错误并给出修复建议 | 防御性设计 |
| 6 | 停掉 Milvus 后访问 `/chat` | 503 + 具体原因 | 优雅降级 |
| 7 | 在 `_apply_rerank` 里 print 融合顺序 | 看到融合顺序 ≠ 重排顺序 | 两阶段真实生效 |
| 8 | `CHUNK_SIZE=200` 并**重新入库** | 检索粒度变细，引用更零碎 | 分块是质量天花板 |
| 9 | 问一个库里没有的问题 | 直接返回兜底文案，日志无模型调用 | 检索为空的短路 |
| 10 | `pytest -v` | 全部通过，且**不需要 Milvus/Ollama** | 测试替身（test double）的价值 |
| 11 | 看启动日志里重排模型那几行 | 出现"已从本地缓存加载（无网络请求）" | 离线优先真的生效了 |
| 12 | `RERANKER_OFFLINE_FIRST=false` 后重启 | 加载明显变慢（要联网校验 etag） | 联网校验的代价 |
| 13 | `STRIP_THINKING=false` 后提问 | 回答前出现大段推理过程，且可能夹着 `[1]` | 思考块为什么必须剥离 |
| 14 | `pytest tests/test_chat.py -v` | 逐字符流式的用例也全部通过 | 状态机的必要性（正则过不了这个用例） |

**关于实验 10**：`tests/` 里所有单元测试都用替身（`FakeStore` / `FakeEmbedder` / `FakeReranker`），
这是本项目最值得学的**工程习惯之一** —— 让测试快、稳定、离线可跑。见 [conftest.py](<F:\PyCharm 2025.1.1.1\projects\Retrieval-Augmented Generation\tests\conftest.py>)。

---

## 8. 最容易读错的地方（陷阱清单）

**1. 分隔符里的中文标点**
`splitter.py` 的 `DEFAULT_SEPARATORS` 包含 `。！？；，`。
如果你的语料是中文而删掉它们，整段中文会被当成一个"词"，
因为默认分隔符只有空格 —— 这是中文 RAG 最常见的坑。

**2. 融合策略现在是自由配置，但要理解代价**
`FUSION_MODE=client` / `server` 两种都能跑，行为完全由配置决定（不再有隐藏的自动修正）。
区别只在：`client` 多一次往返、但保留各分支名次；`server` 一次往返、但只有融合后的分数。
**两者算出的排序一致** —— Milvus 的 `RRFRanker` 用的就是同一个公式。
如果哪天你发现"搜索结果里 `dense_rank` 全是 null"，先看 `FUSION_MODE` 是不是 `server`。

**3. `sparse` 字段不在 insert 的数据里**
它由 Milvus 的 BM25 `Function` 在服务端生成。若你手动塞 `sparse`，
会出现"函数输出字段不可重复赋值"类错误（`insert()` 里已做防御性剔除）。

**4. 重排器返回的是原始下标**
`rerank()` 返回 `[(原始下标, 分数)]`，不是排序后的位置。
若误当位置用，会取错文本且**不报错** —— 静默错误。所以有专门的测试钉住它。

**5. `NoOpReranker` 返回 0 分**
关闭重排时它返回全 0，`_apply_rerank` 特意**不覆盖** `score`，
否则最终分数会全变 0。这也是有测试保护的。

**6. 检索链路内部是同步的**
`retrieve()` 是 async，但内部用 `asyncio.to_thread` 包住同步调用。
读的时候别以为里面有 await 就真的是异步 IO。

**7. 中文语料缺分析器**
若 `.env` 未设 `BM25_ANALYZER_TYPE=chinese`，文本字段用 Milvus 默认分析器，
中文分词效果差，**BM25 分支基本失效**。改它必须重建集合并重新入库（只在建集合时生效）。

**8. `HF_HOME` 必须在导入 HF 库之前设置**
`huggingface_hub` 在 **import 时**就读取 `HF_HOME` 并算出缓存路径常量。
一旦该模块被导入过，之后再改环境变量就不生效了，表现为"配了却没用"。
所以 `deps.py:97` 的 `_apply_hf_env()` 刻意放在 `build_context` 最前面、
且在 `from langchain_ollama import ...` 之前 —— 那些库会间接拉起 HF 依赖。

**9. 思考标签必须是纯 ASCII**
`ThinkingStripper` 的 `OPEN_TAG` / `CLOSE_TAG` 用 `chr(60)` / `chr(62)` 拼出尖括号，
而不是写成完整字面量。原因很实在：这几个字符在文本往返（工具写入、编码转换、
复制粘贴）中容易被改写成 Unicode 近似字符 —— **显示效果一模一样，
但 `find()` 永远匹配不上**，剥离会静默失效。
`tests/test_chat.py` 第一组测试专门断言标签是纯 ASCII，就是为了守住这个坑。

**10. 重排模型"能加载"不等于"缓存命中"**
即使模型已下载，HuggingFace 默认仍会向远端校验 etag。
判断是否真的走了本地，看日志里有没有"已从本地缓存加载（无网络请求）"，
或读 `CrossEncoderReranker.loaded_offline`。

---

## 9. 自测题（能答上 = 真读懂了）

### 基础档

1. 一次提问从浏览器到答案，依次经过哪些文件？
2. `RetrievedChunk` 里为什么要有 `dense_rank` 和 `sparse_rank`？
3. 分块为什么要重叠？
4. 同一文件入库两次，库里有几份数据？

### 进阶档

5. 为什么 RRF 不需要分数归一化？写出它的公式。
6. 为什么重排只对 top-N 而不是全部召回结果？
7. `NoOpReranker` 为什么返回 0 分？上层怎么处理？
8. 为什么入库时要 `flush()`？代价是什么？
9. 为什么 `build_context()` 不抛异常？
10. 思考块剥离为什么不能用正则？缓冲区为什么要留最多 `len(标签)-1` 个字符？

### 架构档

11. `FUSION_MODE=client` 与 `server` 的差别是什么？为什么结果排序一致？
12. 为什么本项目直接要求 Milvus 2.5+，而不兼容 2.4 的客户端 BM25？
12. 为什么重排模型要在 `lifespan` 里加载？换到每请求加载会怎样？
13. 为什么 `HF_HOME` 必须在导入 HF 库之前设置？配晚了会怎样？
14. 为什么思考标签要用 `chr()` 拼装而不是写字面量？
15. 为什么前端不用 `EventSource`？
16. 如果要把 Milvus 换成 Qdrant，需要改哪些文件？（这题最能检验你是否真的理解了分层）
17. 如果要加"查询改写"（多路召回再融合），你会插在哪一层？为什么？

---

## 10. 学完之后往哪走

**按性价比排序：**

1. **补评估** —— 项目已内置：先跑 `rag-app-cli eval data/golden_qa.jsonl` 拿到检索基线，
   再改 `RERANK_ENABLED` / `RRF_K` / `CHUNK_SIZE` 各跑一次对比。
   有了数字，你才能把"我觉得更好了"变成"提升了 X%"。
   评估模块本身也是很好的阅读材料：`eval/retrieval_metrics.py` 是纯函数，比主链路更好懂。
2. **补查询改写** —— 在 `hybrid.py` 的 `_retrieve_sync` 前插入多查询生成，
   多路召回后用 `milvus_store.reciprocal_rank_fusion` 融合。**你已经有了现成的融合函数。**
3. **父子分块** —— 用小块检索、大块送上下文。改 `splitter.py` + `RetrievedChunk`。
4. **加可观测性** —— 每次检索打印各阶段耗时与候选数，找性能瓶颈。
5. **RAGAS 进阶** —— 用它的 testset generation 自动生成评估集，
   替代现在手工标注 `data/golden_qa.jsonl` 的做法。

**学习顺序建议**：先做 1（因为它告诉你其他改动是否真的有效），再做 2 或 3。

---

## 附：一页速查

```
入库：  loader.compute_doc_id → loader.load_directory → splitter.split_documents
        → service._ingest_one_document → milvus_store.insert

检索：  hybrid.retrieve → _retrieve_sync
        ├─ FUSION_MODE=server → milvus_store.hybrid_search   (一次往返，服务端 RRF)
        └─ FUSION_MODE=client → milvus_store.search_and_fuse_branches
                                + milvus_store.reciprocal_rank_fusion  (本地 RRF)
        → rerank.rerank                (交叉编码器精排)

生成：  prompts.build_messages → llm.chat.astream_answer
        → ThinkingStripper             (剥离思考块，流式安全)
        → sse.sse_frame → StreamingResponse → 浏览器 parseFrame

装配：  deps._apply_hf_env   (必须最先，早于任何 HF import)
        deps.build_context   (启动时一次)  ← 唯一知道零件怎么拼的地方
        app.lifespan         (创建/释放)
        routes_*.py          (对外接口)
```

**遇到不明白的代码，先问三个问题**：

1. 它在**哪条链路**上？（入库 / 检索 / 生成 / 装配）
2. 它处理的数据**长什么样**？（对照 §3 的形状变化）
3. 删掉它会**坏掉什么**？—— 答案往往就是那段注释里写的"为什么"
