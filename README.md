# 本地 RAG 知识库问答系统

> Milvus 混合检索（稠密 + BM25）+ RRF 融合 + 交叉编码器重排 + Ollama 本地生成，
> 配套 FastAPI SSE 流式接口与 Web 聊天界面。**全栈自托管，零外部 API 开销。**

整个后端只有 **8 个源文件**，调用链是一条直线 —— 想读懂代码请看 [LEARNING.md](./LEARNING.md)。

---

## 核心特性

| 能力 | 实现要点 |
|---|---|
| **混合检索** | 稠密向量（qwen3-embedding:8b）+ 服务端 BM25，两路召回互补 |
| **RRF 融合** | 只用名次不用分数，规避量纲不可比；纯函数可直接单测 |
| **交叉编码器重排** | `bge-reranker-v2-m3` 对 top-30 候选精排 |
| **重排离线加载** | 命中本地缓存时零网络请求，规避 HuggingFace etag 校验超时 |
| **思考块剥离** | 自动剥离 `deepseek-r1` 的推理过程，且做到流式跨 chunk 安全 |
| **可解释检索** | 每条结果带"在两条分支各排第几"，前端能标出双分支命中 |
| **流式输出** | SSE 逐 token 推送 |
| **幂等入库** | 按 doc_id 先删后插，重复入库不产生重复数据 |
| **优雅降级** | Milvus 不可用时服务仍启动，`/health` 报告原因 |
| **驱动层已现代化** | 用 `MilvusClient` 而非已弃用的 ORM API（3.1 将移除） |

---

## 文件职责（按调用顺序）

```
                     ┌──────────────────────────────────────┐
  浏览器 ──POST───►  │ api.py                               │
                     │  /chat(SSE) /chat/sync /search        │
                     │  /ingest /stats /health              │
                     └───────────────┬──────────────────────┘
                                     │ rag.aretrieve() / rag.astream()
                     ┌───────────────▼──────────────────────┐
                     │ rag.py     ← 全项目的主流程           │
                     │  1. embed_query                       │
                     │  2. store.dense_search / sparse_search│
                     │  3. store.fuse (RRF)                  │
                     │  4. _rerank (交叉编码器)              │
                     │  5. llm.astream → ThinkingStripper    │
                     └───────────────┬──────────────────────┘
                                     │
                     ┌───────────────▼──────────────────────┐
                     │ store.py   Milvus 唯一入口            │
                     │  建表 / 写入 / 两路检索 / RRF 融合     │
                     └──────────────────────────────────────┘

  旁路：ingest.py  文件 → 分块 → 向量 → store.insert
        config.py  所有配置 + 日志
        main.py    启动入口        cli.py  命令行
```

| 文件 | 职责 | 约行数 |
|---|---|---|
| `config.py` | 配置（`Settings`）与日志设置 | 120 |
| `store.py` | Milvus：建表、写入、稠密/BM25 检索、RRF 融合 | 330 |
| `ingest.py` | 文件加载 → 切分 → 向量化 → 写入 | 240 |
| `rag.py` | 检索编排、重排、提示词、流式生成、思考块剥离 | 400 |
| `api.py` | FastAPI 应用与所有路由、SSE 编码 | 290 |
| `main.py` | uvicorn 启动 | 40 |
| `cli.py` | 命令行：入库 / 检索 / 概况 | 170 |

依赖关系只有单向一层：`api → rag → store`。没有依赖注入容器、
没有跨模块回调、没有为"将来可能复用"而预先拆分的层。

---

## 快速开始

### 1. 前置条件

```powershell
ollama pull qwen3-embedding:8b    # 嵌入模型，输出 4096 维
ollama pull deepseek-r1:14b       # 对话模型
ollama list

# Milvus 需 >= 2.5（词法分支依赖服务端 BM25 Function）
# 用 Docker 自行部署一个实例即可（etcd + minio + milvus 三件套），
# 默认 gRPC 端口 19530，与 .env 里的 MILVUS_URI 对应。
```

### 2. 安装与配置

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

配置全部在 `.env` 里（已被 gitignore）。所有项都有代码默认值，不填也能跑。

### 3. 建库

```powershell
rag-app-cli ingest ./data
rag-app-cli stats
```

### 4. 启动

```powershell
rag-app
# 打开 http://127.0.0.1:8000/
```

### 5. 先验证检索（推荐）

```powershell
rag-app-cli search "你的问题"
```

会打印每条结果的**命中分支**（如 `稠密#2 + 稀疏#1`）与分数。
如果这里就不相关，先别怀疑模型 —— 调 `CHUNK_SIZE` / `TOP_K`。

---

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/chat` | **流式对话（SSE）** |
| POST | `/chat/sync` | 非流式，便于脚本 |
| POST | `/search` | **纯检索**，排查问题时先看这个 |
| POST | `/ingest` | 按路径入库 |
| POST | `/ingest/upload` | 上传文件入库 |
| GET | `/stats` | 知识库规模与配置 |
| GET | `/health` | 健康检查（含降级原因） |
| GET | `/docs` | OpenAPI 文档 |

SSE 事件序列：`sources`（引用列表）→ `token`（文本增量，多次）→ `done`。
出错时发 `error` —— 此时响应头已发出，只能靠事件类型告知前端。

```powershell
curl.exe -N -X POST http://127.0.0.1:8000/chat `
  -H "Content-Type: application/json" -d "{\"message\":\"RRF 是什么\"}"
```

---

## 配置要点

| 参数 | 默认 | 说明 |
|---|---|---|
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 800 / 120 | 分块大小；检索质量的天花板 |
| `DENSE_TOP_K` / `SPARSE_TOP_K` | 20 / 20 | 两个分支各召回多少条 |
| `RRF_K` | 60 | 融合常数：越大越平均，越小越看重头部 |
| `RERANK_TOP_N` | 30 | 送去重排的候选数（逐对前向，不能太多） |
| `FINAL_TOP_K` | 5 | 最终交给模型的上下文条数 |
| `BM25_ANALYZER_TYPE` | `chinese` | **中文语料必须设 `chinese`**。留空时默认分析器把整句中文当成一个词，正常问句的 BM25 召回是 0 条，混合检索退化成纯向量检索（症状：所有引用都标"语义命中"）。仅建集合时生效，改了要重建集合 |
| `RERANKER_OFFLINE_FIRST` | true | 优先读本地缓存，避免每次启动联网校验 |
| `STRIP_THINKING` | true | 剥离推理模型的思考过程 |
| `EMBEDDING_DIM` | 4096 | 必须与嵌入模型一致，否则写库报错 |

---

## 测试

```powershell
pytest            # 62 个测试，不需要 Milvus / Ollama / 任何模型
ruff check .
```

| 文件 | 覆盖内容 |
|---|---|
| `test_store.py` | **RRF 融合**：打分公式、双分支优势、去重、k 的影响、确定性 |
| `test_rag.py` | **提示词拼接**、上下文截断、**思考块剥离的流式状态机** |
| `test_api.py` | **SSE 帧格式与事件顺序**、503 降级、检索为空走兜底 |

全部使用测试替身，因此跑得快、离线可跑、结果稳定。

---

## 故障排查

| 现象 | 处理 |
|---|---|
| 启动报"Milvus 版本过低" | 服务端需 >= 2.5（BM25 Function）；升级或用 Docker 起一个 2.5+ 实例 |
| `/health` 显示 degraded | 看 `error` 字段；确认 19530 端口与 `.env` 里的地址 |
| **前端顶部出现黄色告警条** | 那里就是降级原因（如重排加载失败），修好后重启即可 |
| **每次启动都联网、很慢** | 确认 `RERANKER_OFFLINE_FIRST=true` |
| 报 "couldn't find them in the cached files" 但缓存里有 | 项目会先自己定位 `snapshots/<commit>/` 并以本地目录加载；看启动日志里的完整堆栈 |
| 检索结果不相关 | 先调 `CHUNK_SIZE`；用 `rag-app-cli search` 看命中分支；考虑设中文分析器 |
| 回答里出现大段思考内容 | 确认 `STRIP_THINKING=true` |
| 换嵌入模型后报维度不匹配 | 改 `MILVUS_COLLECTION` 换新集合名（不要自动删库） |

---

## 许可

个人作品集项目，供学习与演示使用。
