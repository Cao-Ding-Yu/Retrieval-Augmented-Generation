# 学习指南：读懂这个 RAG 项目

> 后端只有 8 个文件、约 1300 行。不要按文件顺序读，按下面的**调用顺序**读。

---

## 0. 方法只有一句话

**跟着一次提问，从上到下读完 5 个函数，你就懂了这个项目。**

其余文件（入库、配置、入口）都是旁路，最后看。

---

## 1. 一次提问发生了什么

| 步 | 做什么 | 位置 |
|---|---|---|
| 1 | 浏览器 POST `/chat` 并读取流式响应 | `web/index.html` 的 `ask()` |
| 2 | FastAPI 收到请求，进入 SSE 生成器 | `api.py` → `chat()` 里的 `stream()` |
| 3 | **检索**：向量化 → 两路召回 → RRF 融合 → 重排 | `rag.py` → `RAG.retrieve()` |
| 4 | 两路召回与融合的具体实现 | `store.py` → `dense_search` / `sparse_search` / `fuse` |
| 5 | 组装提示词 → 流式生成 → 剥离思考块 | `rag.py` → `astream()` / `build_messages()` |
| 6 | 编码成 SSE 帧发回浏览器 | `api.py` → `_sse()` |

**第 3 步是全项目的心脏**，`RAG.retrieve()` 一个方法里就能看完检索的全部逻辑：

```python
vector = self.embedder.embed_query(question)          # ① 查询变向量
dense  = self.store.dense_search(vector, ...)         # ② 语义召回
sparse = self.store.sparse_search(question, ...)      # ③ 关键词召回
candidates = fuse(dense, sparse, k=...)               # ④ 按名次融合
candidates = candidates[:rerank_top_n]                # ⑤ 截断
return self._rerank(question, candidates)             # ⑥ 精排取前 k
```

---

## 2. 阅读顺序（约 3 小时）

### 第 1 轮：先跑起来，用眼睛看（1 小时）

不要先读代码。

```powershell
rag-app-cli stats                      # 看当前配置
rag-app-cli ingest ./data              # 建库
rag-app-cli search "你的问题"           # ★ 最重要：看检索结果与命中分支
rag-app                               # 启动服务，浏览器里问一句
```

`search` 的输出里每条都标了 `命中分支: 稠密#2 + 稀疏#1`。
**先确认这一步的结果合理，再去读代码。** 检索错了，模型再强也救不回来。

### 第 2 轮：读主流程（1.5 小时）

按这个顺序，每个文件只读指定的方法：

| 顺序 | 文件 | 只读这些 | 读完应能回答 |
|---|---|---|---|
| 1 | `rag.py` | `RAG.retrieve()` | 检索分几步？每步的输入输出是什么？ |
| 2 | `store.py` | `fuse()` | 为什么用名次而不是分数？结果怎么排序？ |
| 3 | `store.py` | `dense_search` / `sparse_search` | 两个分支的调用差别在哪？ |
| 4 | `rag.py` | `_rerank()` | 为什么要第二阶段重排？只重排多少条？ |
| 5 | `rag.py` | `build_messages()` / `format_context()` | 上下文怎么编号？超长丢哪一头？ |
| 6 | `rag.py` | `astream()` | 流式怎么产出？思考块在哪被剥离？ |
| 7 | `api.py` | `chat()` 的 `stream()` | 事件顺序为什么是 sources 在前？ |

### 第 3 轮：旁路与边界（0.5 小时）

- `ingest.py` 的 `ingest()`：为什么"先删后插"？为什么 doc_id 用路径哈希？
- `config.py`：有哪些旋钮，改哪个影响什么
- `api.py` 的 `lifespan`：为什么重排模型要在这里加载而不是每请求加载

---

## 3. 每个文件的核心问题

读完一个文件，你应该能用一句话说出它存在的原因。

| 文件 | 一句话 |
|---|---|
| `config.py` | 所有可调参数与日志，一处修改全局生效 |
| `store.py` | 唯一碰 Milvus 的地方，上层不需要知道 Milvus 的任何细节 |
| `ingest.py` | 把文件变成"能检索的分块"，且重复入库不出问题 |
| `rag.py` | 检索→重排→提示词→流式，主流程都在这里 |
| `api.py` | 把 RAG 包成 HTTP，负责 SSE 与降级 |
| `main.py` | 打日志、启动 uvicorn |
| `cli.py` | 不启服务也能入库与调试检索 |

---

## 4. 三个最容易卡的坑

**1. 中文语料必须设中文分析器**

`.env` 里 `BM25_ANALYZER_TYPE` 留空时，Milvus 用默认分析器按空格切词。
中文没有空格，整句会被当成一个词 —— **BM25 分支基本失效**，混合检索退化成纯向量检索。
设成 `chinese` 后必须重建集合（分析器只在建表时生效）。

**2. 检索链路内部是同步的**

`rag.retrieve()` 是同步方法，`aretrieve()` 只是用 `asyncio.to_thread` 包了一层。
读的时候别以为里面有 `await` 就是异步 IO —— 它是把阻塞调用丢到线程池，
避免卡住事件循环（否则流式输出期间其他请求会全部卡住）。

**3. 思考块剥离不能用正则**

流式 token 的边界和标签边界无关：结束标签可能被切成两半两次到达。
对单个 chunk 做正则必然漏掉被切断的标签，碎片会泄漏到界面上。
所以 `ThinkingStripper` 是个状态机，会把可能构成"半个标签"的尾巴留在缓冲区。

---

## 5. 自测题

能答上这 8 题，说明你真读懂了。

1. 一次提问依次经过哪几个文件的哪几个方法？
2. 为什么检索要跑两个分支？各自擅长什么、漏什么？
3. RRF 为什么不需要分数归一化？写出公式。
4. 为什么重排只对前几十条做，而不是全部召回结果？
5. 两个分支都排第 2 的结果，为什么能赢过只在单分支排第 1 的？
6. 为什么思考块剥离需要缓冲区？
7. 检索为空时为什么直接返回兜底文案、不调用模型？
8. 把 Milvus 换成别的向量库，需要改哪些文件？（只需改 `store.py` 就说明你读懂了分层）

---

## 6. 想继续深入时

按性价比排序：

1. **改一个参数看变化** —— `RRF_K` 从 60 改成 1，或关掉重排再问同一个问题。
   这比读十遍代码更能建立直觉。
2. **加一个接口** —— 比如 `/stats` 里再加几个字段。改 `api.py` 一处即可。
3. **换分块策略** —— 改 `ingest.py` 的 `SEPARATORS` 或 `chunk_size`，重新入库对比。
4. **加可观测性** —— 在 `RAG.retrieve()` 里打印各步耗时，找性能瓶颈。

---

## 附：一页速查

```
提问：  api.chat() → rag.aretrieve()
                      ├─ embedder.embed_query
                      ├─ store.dense_search   (向量)
                      ├─ store.sparse_search  (BM25)
                      ├─ store.fuse           (RRF)
                      └─ rag._rerank          (交叉编码器)
                    → api._sse("sources")
                    → rag.astream() → ThinkingStripper → api._sse("token")
                    → api._sse("done")

入库：  ingest.ingest() → _load → _split → _embed → store.delete_doc + store.insert

启动：  main.main() → api.create_app() → lifespan: RAG.start()
                          ├─ store.connect()（建表/校验）
                          └─ _Reranker.load()（离线优先）
```
