# RAG 知识库问答系统

本地部署的检索增强生成（Retrieval-Augmented Generation）服务：基于 Milvus 的稠密 + BM25 混合检索，
经 RRF 融合与交叉编码器重排后，交由本地大模型生成带引用的答案，并通过 SSE 以流式方式返回。

后端由 8 个源文件构成，不依赖任何外部 API。

---

## 目录

- [1. 项目简介](#1-项目简介)
- [2. 核心特性](#2-核心特性)
- [3. 系统架构](#3-系统架构)
- [4. 环境要求](#4-环境要求)
- [5. 快速开始](#5-快速开始)
- [6. 配置](#6-配置)
- [7. HTTP 接口](#7-http-接口)
- [8. 命令行工具](#8-命令行工具)
- [9. 项目结构](#9-项目结构)
- [10. 关键设计说明](#10-关键设计说明)
- [11. 测试与代码质量](#11-测试与代码质量)
- [12. 故障排查](#12-故障排查)
- [13. 已知限制](#13-已知限制)
- [14. 许可证](#14-许可证)

---

## 1. 项目简介

系统的目标是在完全离线的环境下提供可溯源的文档问答：给定一个自然语言问题，从知识库中
检索相关片段，并将答案与来源一并返回。

主要设计取向：

- **检索质量优先于链路复杂度。** 双分支召回 + 融合 + 重排均保留完整实现，便于观察每一步的影响。
- **可解释性。** 每个引用片段都返回其分数、在两个召回分支中的名次，以及 PDF 页码。
- **本地优先。** 模型推理、向量检索、重排全部在本机完成，无外发请求。
- **失败可见。** 依赖不可用或能力降级时，服务仍可启动，原因通过 `/health` 与前端告警条呈现。

---

## 2. 核心特性

| 特性 | 实现方式 |
|---|---|
| 混合检索 | 稠密向量（`qwen3-embedding:8b`，4096 维）+ Milvus 服务端 BM25，两路召回互补 |
| RRF 融合 | 仅使用排名而非分数，规避不同量纲不可比的问题；实现为纯函数，可精确单测 |
| 交叉编码器重排 | `BAAI/bge-reranker-v2-m3` 对融合后的候选精排，支持 CPU / CUDA / Apple MPS |
| 重排离线加载 | 命中本地缓存时零网络请求，规避 HuggingFace etag 校验超时 |
| 思考过程分离 | 推理模型的思考与答案作为两个独立的 SSE 事件下发，网页端独立渲染、生成完毕后自动收起 |
| 可解释检索 | 每条结果返回 `score`、`score_kind`、`dense_rank`、`sparse_rank`、PDF 页码 |
| 流式输出 | SSE 逐 token 推送，前端使用 `fetch` + `ReadableStream` 解析 |
| 幂等入库 | 以文件路径派生的 `doc_id` 先删后插，重复入库不产生冗余数据 |
| 优雅降级 | Milvus 或重排模型不可用时服务仍可启动，并通过 `/health` 报告具体原因 |
| 驱动层 | 使用 `MilvusClient` 而非已弃用的 ORM API |

---

## 3. 系统架构

```
                         ┌─────────────────────────────────────────┐
   浏览器  ── POST ────► │ api.py                                  │
                         │   /chat (SSE)  /chat/sync  /search       │
                         │   /ingest  /ingest/upload  /stats  /health│
                         └───────────────────┬─────────────────────┘
                                             │ RAG.aretrieve() / RAG.astream()
                         ┌───────────────────▼─────────────────────┐
                         │ rag.py                                  │
                         │   1. embed_query        查询向量化       │
                         │   2. dense_search       稠密召回         │
                         │   3. sparse_search      BM25 召回        │
                         │   4. fuse               RRF 融合         │
                         │   5. _rerank            交叉编码器精排   │
                         │   6. llm.astream        流式生成         │
                         │   7. ThinkingStripper   思考/答案分离    │
                         └───────────────────┬─────────────────────┘
                                             │
                         ┌───────────────────▼─────────────────────┐
                         │ store.py     Milvus 唯一出口             │
                         │   建表 / 写入 / 两路检索 / RRF 融合      │
                         └─────────────────────────────────────────┘

   旁路：ingest.py   文件 → 分块 → 向量化 → 写入（递归遍历目录）
         config.py   配置与日志
         main.py     服务入口        cli.py   命令行工具
```

检索流程与耗时特征：

| 阶段 | 实现 | 说明 |
|---|---|---|
| 查询向量化 | Ollama `/api/embed` | 单次网络往返 |
| 稠密召回 | HNSW，`ef = max(128, k)` | 余弦相似度 |
| 词法召回 | 服务端 BM25 Function | 传入原始文本，由 Milvus 分词并打分 |
| 融合 | `fuse()`，RRF `k=60` | 纯函数，输入输出均为 `list[Hit]` |
| 重排 | 交叉编码器，逐对前向 | 仅对 `RERANK_TOP_N` 条候选计算 |
| 生成 | Ollama `/api/chat` | 流式返回 |

依赖方向为单向一层：`api → rag → store`，不存在循环依赖或回调注册。

---

## 4. 环境要求

| 组件 | 版本要求 | 说明 |
|---|---|---|
| Python | >= 3.10, < 4.0 | 代码使用 `X \| None` 等新语法 |
| Milvus | >= 2.5（实测 3.0.2） | 词法分支依赖服务端 BM25 `Function` |
| Ollama | 支持 `thinking` 字段的版本 | 用于对话与嵌入 |
| 对话模型 | `deepseek-r1:14b` | 推理模型；换用非推理模型需设 `SHOW_THINKING=false` |
| 嵌入模型 | `qwen3-embedding:8b` | 输出 4096 维；须与 `EMBEDDING_DIM` 一致 |
| 内存 | 建议 16 GB 以上 | 重排模型约 2 GB；对话模型由 Ollama 管理 |

Milvus 需以 Docker 方式部署（etcd + MinIO + milvus 三组件），默认 gRPC 端口 `19530`。

---

## 5. 快速开始

### 5.1 拉取模型

```bash
ollama pull qwen3-embedding:8b
ollama pull deepseek-r1:14b
ollama list
```

### 5.2 安装依赖

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

pip install -e ".[dev]"
```

### 5.3 创建配置

在项目根目录创建 `.env`。所有配置项均有默认值，最小可用配置如下：

```dotenv
CHAT_MODEL=deepseek-r1:14b
EMBEDDING_MODEL=qwen3-embedding:8b
EMBEDDING_DIM=4096
MILVUS_URI=http://localhost:19530
MILVUS_COLLECTION=rag_chunks
BM25_ANALYZER_TYPE=chinese
RERANKER_DEVICE=auto
```

完整配置项见[第 6 节](#6-配置)。

### 5.4 导入文档

```bash
rag-app-cli ingest ./data
rag-app-cli stats
```

目录会被递归遍历，支持 `.txt`、`.md`、`.markdown`、`.rst`、`.pdf`；
无文本层的扫描版 PDF 会被跳过，并在结果中列出。

### 5.5 启动服务

```bash
rag-app
```

访问 `http://127.0.0.1:8000/` 使用网页界面，`http://127.0.0.1:8000/docs` 查看 OpenAPI 文档。

### 5.6 验证检索

在排查"答案不准"之前，应先确认检索本身是否合理：

```bash
rag-app-cli search "某个问题"
```

输出包含每条结果的命中分支（如 `稠密#2 + 稀疏#1`）、分数与 PDF 页码。若此步结果已不相关，
问题通常位于分块策略或分析器配置，而非生成模型。

---

## 6. 配置

配置通过环境变量或项目根目录的 `.env` 文件提供（`.env` 已在 `.gitignore` 中排除）。
变量名不区分大小写。

### 6.1 服务与模型

| 变量 | 默认值 | 说明 |
|---|---|---|
| `HOST` | `127.0.0.1` | 监听地址 |
| `PORT` | `8000` | 监听端口 |
| `LOG_LEVEL` | `info` | 日志级别 |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama 服务地址 |
| `CHAT_MODEL` | `deepseek-r1:14b` | 对话模型 |
| `EMBEDDING_MODEL` | `qwen3-embedding:8b` | 嵌入模型 |
| `EMBEDDING_DIM` | `4096` | 嵌入维度，必须与模型实际输出一致 |

### 6.2 Milvus

| 变量 | 默认值 | 说明 |
|---|---|---|
| `MILVUS_URI` | `http://localhost:19530` | Milvus 服务地址 |
| `MILVUS_TOKEN` | 空 | 鉴权信息；留空表示无鉴权 |
| `MILVUS_COLLECTION` | `rag_chunks` | 集合名 |
| `BM25_ANALYZER_TYPE` | `chinese` | 文本分析器；**仅在创建集合时生效** |

### 6.3 检索与重排

| 变量 | 默认值 | 说明 |
|---|---|---|
| `DENSE_TOP_K` | `20` | 稠密分支召回条数 |
| `SPARSE_TOP_K` | `20` | BM25 分支召回条数 |
| `RRF_K` | `60` | RRF 平滑常数；越小越看重头部排名 |
| `RERANK_ENABLED` | `true` | 是否启用重排 |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | 交叉编码器模型 |
| `RERANKER_DEVICE` | `auto` | `auto` / `cpu` / `cuda` / `mps` |
| `RERANK_TOP_N` | `30` | 送入重排的候选数 |
| `FINAL_TOP_K` | `5` | 最终交给模型的上下文条数 |
| `HF_HOME` | 空 | HuggingFace 缓存根目录；留空使用默认位置 |
| `RERANKER_OFFLINE_FIRST` | `true` | 优先以本地缓存加载，避免联网校验 |
| `RERANKER_ALLOW_DOWNLOAD` | `true` | 缓存缺失时是否允许下载 |

### 6.4 入库与生成

| 变量 | 默认值 | 说明 |
|---|---|---|
| `CHUNK_SIZE` | `800` | 分块最大字符数 |
| `CHUNK_OVERLAP` | `120` | 相邻分块重叠字符数，须小于 `CHUNK_SIZE` |
| `DATA_DIR` | `./data` | 上传文件的落盘目录 |
| `TEMPERATURE` | `0.2` | 采样温度 |
| `MAX_CONTEXT_CHARS` | `10000` | 单次请求的上下文长度上限 |
| `SHOW_THINKING` | `true` | 是否将思考过程下发给前端 |

### 6.5 关于 GPU

重排是本机唯一运行神经网络的环节，默认由 `RERANKER_DEVICE=auto` 自动选择设备。
若需使用 NVIDIA 显卡，须安装 CUDA 版 PyTorch —— 纯 CPU 轮子（版本号形如 `2.14.1+cpu`）
中 `torch.cuda.is_available()` 恒为 `False`，此时即使配置为 `cuda` 也会报错而非静默降级。

```bash
# 确认当前构建
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# RTX 50 系列（Blackwell, sm_120）需要 cu128；更早的显卡可用 cu126
pip install --index-url https://download.pytorch.org/whl/cu128 torch
```

安装成功后，`/health` 的 `reranker.device` 将显示 `cuda`。

---

## 7. HTTP 接口

### 7.1 接口一览

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/chat` | 流式对话（SSE） |
| POST | `/chat/sync` | 非流式对话，便于脚本调用 |
| POST | `/search` | 仅检索，返回排序明细 |
| POST | `/ingest` | 按服务端路径导入（目录递归） |
| POST | `/ingest/upload` | 上传单个文件并导入 |
| GET | `/stats` | 知识库规模与当前配置 |
| GET | `/health` | 健康检查与降级原因 |
| GET | `/` | 网页界面（静态资源） |
| GET | `/docs` | OpenAPI 文档 |

### 7.2 SSE 协议

`POST /chat` 的响应为 `text/event-stream`，事件序列如下：

| 事件 | 方向 | 负载 |
|---|---|---|
| `sources` | 服务端 → 客户端 | `{"count": int, "sources": [ ... ]}` |
| `thinking` | 服务端 → 客户端 | `{"text": "增量文本"}`，可出现多次 |
| `token` | 服务端 → 客户端 | `{"text": "增量文本"}`，可出现多次 |
| `done` | 服务端 → 客户端 | `{"answer_chars": int}` |
| `error` | 服务端 → 客户端 | `{"message": "错误描述"}` |

`sources` 先于生成下发，客户端可在文本到达前渲染引用列表。思考与答案使用不同事件，
使前端能够将二者渲染为独立区块，并避免思考内容中的 `[1]` 被误认为引用标记。

响应头包含 `X-Accel-Buffering: no`，用于禁用反向代理的响应缓冲。

### 7.3 请求与响应示例

**流式对话**

```bash
curl -N -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "混合检索的原理是什么", "include_sources": true}'
```

```
event: sources
data: {"count":2,"sources":[{"doc_id":"…","source":"data/intro.pdf","chunk_index":13,
      "page":10,"score":0.989556,"score_kind":"rerank","dense_rank":0,"sparse_rank":0,
      "is_hybrid_hit":true,"snippet":"…","text":"…"}]}

event: thinking
data: {"text":"\n分析问题：用户问的是混合检索的原理…"}

event: token
data: {"text":"混合检索同时使用"}

event: done
data: {"answer_chars":458}
```

**非流式对话**

```bash
curl -X POST http://127.0.0.1:8000/chat/sync \
  -H "Content-Type: application/json" \
  -d '{"message": "混合检索的原理是什么"}'
```

```json
{ "answer": "……", "sources": [ { "source": "data/intro.pdf", "page": 10, "score": 0.9896 } ] }
```

**仅检索**

```bash
curl -X POST http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "混合检索"}'
```

```json
{ "query": "混合检索", "count": 5, "results": [ { "source": "data/intro.pdf", "page": 10,
  "chunk_index": 13, "score": 0.9896, "score_kind": "rerank",
  "dense_rank": 0, "sparse_rank": 0, "is_hybrid_hit": true } ] }
```

**入库**

```bash
curl -X POST http://127.0.0.1:8000/ingest \
  -H "Content-Type: application/json" -d '{"path": "./data"}'

curl -X POST http://127.0.0.1:8000/ingest/upload -F "file=@./manual.pdf"
```

```json
{ "files_loaded": 2, "chunks_written": 371, "skipped": [] }
```

**健康检查**

```bash
curl http://127.0.0.1:8000/health
```

```json
{
  "status": "ok", "version": "0.1.0", "ready": true, "error": null, "warnings": [],
  "milvus": { "collection": "rag_chunks", "server_version": "3.0.2", "chunks": 371 },
  "retrieval": { "embedding_model": "qwen3-embedding:8b", "dense_top_k": 20,
                  "sparse_top_k": 20, "rrf_k": 60, "final_top_k": 5 },
  "reranker": { "enabled": true, "model": "BAAI/bge-reranker-v2-m3",
                 "device": "cpu", "device_configured": "auto" },
  "llm": { "chat_model": "deepseek-r1:14b", "show_thinking": true }
}
```

依赖不可用时 `status` 为 `degraded`、`ready` 为 `false`，HTTP 状态码仍为 `200`，
具体原因见 `error` 与 `warnings` 字段。

---

## 8. 命令行工具

```bash
rag-app-cli ingest <path>      # 导入文件或目录（递归）
rag-app-cli search <query>     # 仅检索，打印命中分支与分数
rag-app-cli stats              # 知识库规模与当前配置
rag-app-cli -v <command>       # 输出 DEBUG 级日志
```

`search` 支持 `--json` 输出完整结构，便于脚本处理。CLI 与服务端共用同一套组装逻辑，
因此 CLI 中的检索行为与 HTTP 接口一致。

---

## 9. 项目结构

```
.
├── src/rag_app/
│   ├── __init__.py      版本号
│   ├── config.py        配置模型与日志初始化
│   ├── store.py         Milvus 访问层：建表、写入、两路检索、RRF 融合
│   ├── ingest.py        文件加载、切分、向量化、写入
│   ├── rag.py           检索编排、重排、提示词、流式生成、思考分离
│   ├── api.py           FastAPI 应用与路由、SSE 编码
│   ├── main.py          服务启动入口
│   └── cli.py           命令行工具
├── tests/
│   ├── conftest.py      公共夹具与测试替身
│   ├── test_store.py    RRF 融合、页码传递
│   ├── test_rag.py      提示词、思考分离、设备解析
│   └── test_api.py      SSE 协议、状态码、降级行为
├── web/index.html       单页界面（无构建步骤）
├── data/                语料与上传文件（不入库）
├── pyproject.toml       依赖、脚本入口、ruff 与 pytest 配置
└── .env                 本地配置（不入库）
```

| 文件 | 职责 |
|---|---|
| `config.py` | 全部可调参数的唯一定义处，含跨字段校验 |
| `store.py` | 唯一直接访问 Milvus 的模块，上层无需了解存储细节 |
| `ingest.py` | 将文件转换为可检索的分块，并保证重复导入的幂等性 |
| `rag.py` | 检索与生成的主流程 |
| `api.py` | HTTP 适配层：请求校验、SSE 编码、降级处理 |
| `main.py` / `cli.py` | 两个入口，共用同一组装逻辑 |

---

## 10. 关键设计说明

### 10.1 为何采用 RRF 而非加权分数

稠密检索输出余弦相似度（约 -1~1），BM25 输出无上界的词频加权分。二者量纲不同，
直接加权需先归一化，而归一化对分数分布尾部极敏感。RRF 仅使用排名：

```
RRF(d) = Σ 1 / (k + rank_i(d))
```

其性质 `2/(k+2) > 1/(k+1)`（任意 `k > 0`）意味着"两路均排第 2"稳定优于"单路排第 1"，
即在形式上表达了"多路证据一致优于单路极端高分"。

融合在客户端完成（两次检索 + 本地 `fuse`），而非调用 Milvus 的 `hybrid_search`。
多一次往返换取各分支排名，前端据此标注"双分支命中"并解释排序依据。

### 10.2 中文语料必须配置中文分析器

`BM25_ANALYZER_TYPE` 留空时，Milvus 使用默认分析器，会将整句中文视为一个词元。
后果是仅当查询串与文档中出现过的片段完全一致时才命中 —— 实测完整问句的 BM25 召回为 0 条，
混合检索退化为纯向量检索，前端表现为所有引用均标注"语义命中"。

该参数仅在创建集合时生效。修改后须以新的集合名重建并重新导入。

### 10.3 重排模型的离线加载

加载采用三级降级：先按 HuggingFace 缓存布局定位 `snapshots/<commit>/` 并以本地目录加载；
失败则带 `local_files_only=True` 按仓库名加载；仍失败才联网下载。

第一级之所以必要，是因为默认加载流程每次都会向远端校验 etag，在国内网络下常超时数秒。
其报错为 `couldn't connect to huggingface.co ... couldn't find them in the cached files`，
但文件实际存在，失败发生在仓库解析而非文件缺失环节，具有误导性。

### 10.4 思考内容的两个来源

推理模型的思考内容有两种到达方式，实现中同时处理：

1. 独立字段 `additional_kwargs["reasoning_content"]` —— 新版 Ollama 将思考置于
   `message.thinking`，`content` 在思考阶段为空。`langchain_ollama` 仅在
   `reasoning=True` 时转存该字段，否则**静默丢弃**。
2. 正文内标签 `<think>…</think>` —— 旧版本或其它推理模型的内联形式。

流式 token 的边界与标签边界无关，结束标签可能被拆分到多个分片，因此剥离逻辑实现为
状态机而非正则，并将可能构成"半个标签"的尾部保留在缓冲区。

### 10.5 分块计数为何不用 `get_collection_stats()`

入库采用"先删后插"，而 `get_collection_stats().row_count` 统计的是物理行数，
包含已删除但尚未 compaction 的墓碑行。重新导入后该值不降反升（实测 371 → 742），
易被误判为数据重复。因此 `/stats` 与 `/health` 使用 `count(*)` 聚合查询，
仅统计未删除行。

### 10.6 PDF 页码的传递

`chunk_index` 表示"文档内第几个分块"，与 PDF 页码无关：单页可能切分出多个分块，
且分块序号跨页连续。因此解析阶段将页码写入元数据，随分块一并入库到 `page` 字段
（非 PDF 来源写 0，读出时映射为 `None`），用于引用定位与提示词标注。

旧集合缺少该字段时，服务启动阶段通过 `add_collection_field` 自动补加 ——
服务端要求新增字段必须为 nullable。

---

## 11. 测试与代码质量

```bash
pytest            # 87 个用例，无需 Milvus / Ollama / 任何模型
ruff check .      # 静态检查
```

测试全部使用替身，可在数秒内离线完成：

| 文件 | 覆盖范围 |
|---|---|
| `test_store.py` | RRF 打分公式、双分支优势性质、分支内去重、`k` 的影响、排序确定性、页码往返 |
| `test_rag.py` | 上下文拼装与截断、提示词约束、思考拆分（跨分片状态机、两个来源）、设备解析 |
| `test_api.py` | SSE 帧格式与事件顺序、`503` 降级、空检索兜底、字段契约 |

静态检查配置于 `pyproject.toml`：`line-length = 110`（ruff 按显示宽度计算，中文字符占 2 列），
并关闭 `E701` 以允许紧凑的单行写法。

---

## 12. 故障排查

| 现象 | 处理 |
|---|---|
| 启动报"Milvus 版本过低" | 服务端需 >= 2.5；升级或部署更高版本实例 |
| `/health` 显示 `degraded` | 查看 `error` 字段；确认 `MILVUS_URI` 与端口 |
| 前端顶部出现黄色告警条 | 该处即为降级原因（如重排模型加载失败），修复后重启 |
| 启动缓慢且访问 HuggingFace | 确认 `RERANKER_OFFLINE_FIRST=true` |
| 报"couldn't find them in the cached files"但缓存存在 | 失败位于仓库解析环节；查看启动日志完整堆栈 |
| 所有引用均标注"语义命中" | `BM25_ANALYZER_TYPE` 未设为 `chinese`，需重建集合并重新导入 |
| 检索结果不相关 | 先调整 `CHUNK_SIZE`；用 `rag-app-cli search` 检查命中分支 |
| 答案中出现 `<think>` 标签 | 该模型未使用标准标签格式，当前实现按 `<think>` / `</think>` 拆分 |
| `RERANKER_DEVICE=cuda` 报错 | 当前 PyTorch 为 CPU 构建，见[第 6.5 节](#65-关于-gpu) |
| 跨域调用失败 | CORS 未开启；如需跨域访问，应在 `api.py` 中显式配置允许来源 |

---

## 13. 已知限制

- **单机单进程。** 重排模型在进程内加载，无法通过增加进程数横向扩展；多副本部署会重复占用内存。
- **无鉴权。** 接口未提供任何认证机制，仅应部署于受信网络。
- **无对话历史。** 每次请求独立检索，不维护多轮上下文。
- **不支持增量索引。** 文档更新依赖整文件重新导入；重命名文件会被视为新文档，旧路径下的分块需手动清理。
- **扫描版 PDF 不支持。** 无文本层的页面会被跳过，未集成 OCR。
- **思考过程不参与检索。** 思考内容仅用于展示，不写入知识库。
- **未实现请求取消。** 生成过程中无法中止，仅能在客户端断开后由服务端感知并停止。

---

## 14. 许可证

本项目为学习与演示用途。第三方组件遵循各自许可证，其中：

- `BAAI/bge-reranker-v2-m3`：MIT
- Milvus：Apache-2.0
- LangChain：MIT
