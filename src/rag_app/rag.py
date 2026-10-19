"""RAG 主流程：检索 → 重排 → 组装提示词 → 流式生成。

整条链路集中在这一个文件里，``RAG.retrieve()`` 就能从上到下读完。
调用关系只有两层：

    api.py  →  RAG.aretrieve()  →  store.dense_search / sparse_search / fuse
            →  RAG.astream()    →  llm.astream

## 为什么检索要分两个分支再做融合

- **稠密检索**（向量）擅长语义相近但措辞不同的情况，但漏精确标识符
  （错误码、版本号、专有名词）。
- **BM25 检索**（关键词）正好相反：精确词命中极强，但不认识同义词。

两者互补，所以各跑一次，再用 RRF 按名次融合（见 store.fuse 的说明）。
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings

from rag_app.config import Settings
from rag_app.store import Hit, MilvusStore, fuse

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是一个严谨的知识库问答助手。你的回答必须完全基于用户提供的【参考资料】。

规则：
1. 只使用【参考资料】中的信息作答，不要用你自己的先验知识补充或推测。
2. 每个结论后标注来源编号，格式为 [1]、[2]，可同时标注多个如 [1][3]。
3. 如果资料不足以回答，直接说明"根据已有资料无法回答该问题"，禁止编造。
4. 如果资料之间矛盾，指出矛盾之处并说明你的倾向及理由。
5. 使用与提问相同的语言，需要列举时用有序列表。
6. 不要复述这些规则，直接给出答案。"""

# 检索为空时的兜底文案（此时不调用模型，省一次推理）
NO_CONTEXT_REPLY = (
    "我在当前知识库中没有检索到与该问题相关的内容。"
    "请换个说法，或先向知识库补充相关文档。"
)


# ===========================================================================
# 主类
# ===========================================================================
class RAG:
    """把检索、重排、生成串起来。整个应用只创建一个实例。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.store = MilvusStore(settings)

        # 嵌入模型：用于把查询和分块变成向量
        self.embedder = OllamaEmbeddings(
            base_url=settings.ollama_base_url,
            model=settings.embedding_model,
        )
        # 对话模型：用于生成答案
        self.llm = ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.chat_model,
            temperature=settings.temperature,
        )

        self._reranker: _Reranker | None = None
        self.rerank_warning: str | None = None
        """重排加载失败时的原因，会显示在健康检查与前端告警条里。"""

    # ------------------------------------------------------------------
    # 启动 / 关闭
    # ------------------------------------------------------------------
    def start(self) -> None:
        """连接 Milvus 并加载重排模型。由应用的 lifespan 调用一次。

        Milvus 连不上会抛异常（检索是整个服务的基础，无法降级）；
        重排加载失败只降级为不重排，并把原因记到 ``rerank_warning``。
        """
        _apply_hf_env(self.settings)
        self.store.connect()

        if not self.settings.rerank_enabled:
            logger.info("重排已关闭（RERANK_ENABLED=false）")
            return

        try:
            self._reranker = _Reranker(self.settings)
            self._reranker.load()
        except Exception as exc:
            # 不阻断服务：降级为不重排，但把原因暴露出来（不静默）
            self.rerank_warning = f"重排模型加载失败，已降级为不重排：{type(exc).__name__}: {exc}"
            logger.exception("重排模型加载失败")
            self._reranker = None

    def close(self) -> None:
        self.store.close()

    @property
    def rerank_enabled(self) -> bool:
        return self._reranker is not None

    # ------------------------------------------------------------------
    # 检索（检索 + 重排，不调用生成模型）
    # ------------------------------------------------------------------
    async def aretrieve(self, question: str) -> list[Hit]:
        """检索入口。同步链路放到线程池，避免阻塞事件循环。"""
        return await asyncio.to_thread(self.retrieve, question)

    def retrieve(self, question: str) -> list[Hit]:
        """完整检索：向量化 → 双分支召回 → RRF 融合 → 交叉编码器重排。

        这是全项目最核心的方法，从上到下就是检索的全部逻辑。
        """
        question = (question or "").strip()
        if not question:
            return []

        # 1) 查询向量化
        vector = self.embedder.embed_query(question)

        # 2) 两个分支各召回一次
        dense = self.store.dense_search(vector, self.settings.dense_top_k)
        sparse = self.store.sparse_search(question, self.settings.sparse_top_k)

        # 3) 按名次融合（RRF）
        candidates = fuse(dense, sparse, k=self.settings.rrf_k)
        if not candidates:
            return []

        # 4) 截断到重排预算：重排是逐对前向计算，候选太多会很慢
        candidates = candidates[: self.settings.rerank_top_n]

        # 5) 精排后取最终条数
        return self._rerank(question, candidates)

    def _rerank(self, question: str, hits: list[Hit]) -> list[Hit]:
        """用交叉编码器重排，返回前 final_top_k 条。

        未启用重排时原样截断 —— 调用方不需要关心重排是否可用。
        """
        limit = self.settings.final_top_k
        if self._reranker is None:
            return hits[:limit]

        scores = self._reranker.score(question, [h.text for h in hits])
        # 按重排分数降序；分数相同时保持原顺序，保证结果可复现
        order = sorted(range(len(hits)), key=lambda i: (-scores[i], i))
        result = []
        for index in order[:limit]:
            hit = hits[index]
            hit.score = scores[index]
            result.append(hit)
        return result

    # ------------------------------------------------------------------
    # 生成
    # ------------------------------------------------------------------
    async def astream(self, question: str, hits: list[Hit]) -> AsyncIterator[str]:
        """流式生成答案，逐段产出文本。

        推理模型（deepseek-r1 等）会把思考过程混在正文里，
        这里边流边剥离，用户只会看到最终答案。
        """
        messages = build_messages(question, hits, self.settings.max_context_chars)
        stripper = ThinkingStripper(enabled=self.settings.strip_thinking)

        async for chunk in self.llm.astream(messages):
            text = _chunk_text(chunk.content)
            if not text:
                continue
            safe = stripper.feed(text)
            if safe:
                yield safe

        tail = stripper.flush()
        if tail:
            yield tail

    async def answer(self, question: str, hits: list[Hit]) -> str:
        """一次性生成完整答案（脚本与测试用；面向用户应走 SSE）。"""
        messages = build_messages(question, hits, self.settings.max_context_chars)
        response = await self.llm.ainvoke(messages)
        return strip_thinking(_chunk_text(response.content),
                              enabled=self.settings.strip_thinking)


# ===========================================================================
# 提示词
# ===========================================================================
def build_messages(question: str, hits: list[Hit], max_chars: int) -> list[BaseMessage]:
    """拼成 [System, Human] 两条消息。

    参考资料放在 Human 而不是 System：本地小模型对 system 的遵循度偏弱，
    把"资料 + 问题"放一起关联更紧。
    """
    context = format_context(hits, max_chars)

    if context:
        body = (
            f"【参考资料】\n{context}\n\n"
            f"【问题】\n{question}\n\n"
            "请依据上述参考资料回答问题，并在结论处标注来源编号。"
        )
    else:
        # 没有资料时明确告知模型，避免它用先验知识硬编答案
        body = (
            "【参考资料】\n（本次检索未返回任何相关内容）\n\n"
            f"【问题】\n{question}\n\n"
            "由于没有检索到参考资料，请直接说明无法回答，不要使用你自己的知识作答。"
        )

    return [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=body)]


def format_context(hits: list[Hit], max_chars: int) -> str:
    """把检索结果拼成带编号的参考资料。

    超出长度上限时从**末尾**丢弃 —— 列表本身按相关性降序，末尾最不重要。
    """
    if not hits:
        return ""

    blocks: list[str] = []
    used = 0
    for index, hit in enumerate(hits, start=1):
        name = hit.source.replace("\\", "/").rsplit("/", 1)[-1]
        block = f"[{index}] 来源: {name} · 第 {hit.chunk_index + 1} 块\n{hit.text.strip()}"

        if used + len(block) + 2 > max_chars:
            if not blocks:  # 第一条就超长，截断保留开头
                blocks.append(block[: max(0, max_chars - used)])
            break
        blocks.append(block)
        used += len(block) + 2

    return "\n\n".join(blocks)


# ===========================================================================
# 重排器
# ===========================================================================
class _Reranker:
    """交叉编码器重排器。

    ## 为什么还需要第二阶段重排

    稠密检索用的双编码器把查询和文档**分别**编码，为了能预建索引而牺牲了交互 ——
    编码文档时根本不知道查询是什么。

    交叉编码器把 (查询, 文档) 拼成一个序列一起送进模型，注意力可直接在两者间交互，
    判断准确得多；代价是无法预计算，N 个候选要跑 N 次前向。
    所以只对前几十条候选做精排。
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: Any = None

    def load(self) -> None:
        """加载模型（离线优先）。

        三级降级，从最稳到最兜底：
          1. 自己按 HF 缓存布局定位 ``snapshots/<commit>/``，以**本地目录**加载 ——
             完全跳过仓库解析与 etag 校验；
          2. 按仓库名加载但带 ``local_files_only=True``；
          3. 前两级都失败才联网下载。

        ## 为什么需要第 1 级

        HuggingFace 默认每次加载都会向远端校验 etag，即使文件都在本地。
        在国内网络下这一步常超时数秒甚至失败，报错是
        "couldn't connect ... couldn't find them in the cached files" ——
        但文件其实就在磁盘上，失败发生在**仓库解析**环节而非文件缺失，极具误导性。
        直接传本地目录则不走这套流程。
        """
        from sentence_transformers import CrossEncoder

        name = self._settings.reranker_model
        kwargs = {"device": "cpu", "max_length": 512}

        if self._settings.reranker_offline_first:
            snapshot = _resolve_snapshot_dir(name)
            if snapshot is not None:
                try:
                    logger.info("从本地快照加载重排模型：%s", snapshot)
                    self._model = CrossEncoder(str(snapshot), **kwargs)
                    logger.info("重排模型已加载（无网络请求）")
                    return
                except Exception:
                    logger.warning("本地快照加载失败，改试离线模式", exc_info=True)
            try:
                self._model = CrossEncoder(name, local_files_only=True, **kwargs)
                logger.info("重排模型已加载（无网络请求）")
                return
            except Exception:
                if not self._settings.reranker_allow_download:
                    raise
                logger.warning("本地缓存不可用，联网下载重排模型 %s（仅此一次）", name)

        self._model = CrossEncoder(name, local_files_only=False, **kwargs)
        logger.info("重排模型已加载（经由网络/默认路径）")

    def score(self, question: str, documents: list[str]) -> list[float]:
        """给每个候选打相关性分数（越大越相关）。"""
        if not documents:
            return []
        pairs = [(question, doc) for doc in documents]
        return [float(s) for s in self._model.predict(pairs, batch_size=16,
                                                      show_progress_bar=False)]


def _apply_hf_env(settings: Settings) -> None:
    """在任何 HF 相关 import 之前把缓存位置写进环境变量。

    ``huggingface_hub`` 在 **import 时**就读取 HF_HOME 并算出缓存路径常量，
    之后再改环境变量就不生效了（表现为"配了却没用"）。所以必须最早执行。
    """
    if settings.hf_home:
        os.environ["HF_HOME"] = str(Path(settings.hf_home).expanduser())


def _resolve_snapshot_dir(model_name: str) -> Path | None:
    """把 ``作者/模型名`` 解析成 HF 缓存里的快照目录。

    缓存布局::

        <hub>/models--<作者>--<模型名>/
            refs/main           # 内容为 commit 哈希
            snapshots/<commit>/ # 真正的模型文件

    以本地目录加载时 HuggingFace 会当作"本地模型"，完全跳过仓库解析与网络校验。
    """
    if "/" not in model_name:
        return None

    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        root = Path(HF_HUB_CACHE)
    except Exception:  # pragma: no cover
        return None

    org, _, name = model_name.partition("/")
    repo = root / f"models--{org}--{name}"
    snapshots = repo / "snapshots"
    if not snapshots.is_dir():
        return None

    ref = repo / "refs" / "main"
    if ref.is_file():
        commit = ref.read_text(encoding="utf-8").strip()
        candidate = snapshots / commit
        if commit and candidate.is_dir():
            return candidate

    # refs 缺失或不匹配时退化为取最新快照（缓存被手工修补过时会出现）
    dirs = [d for d in snapshots.iterdir() if d.is_dir()]
    return max(dirs, key=lambda d: d.stat().st_mtime) if dirs else None


# ===========================================================================
# 思考块剥离
# ===========================================================================
class ThinkingStripper:
    """剥离推理模型的思考块，**支持流式**输入。

    ## 为什么不能用正则

    ``deepseek-r1`` 会把推理链包在一对标签之间混在正文里输出。直接转发的话，
    用户会先看到一大段"让我想想…"，而且思考里的 ``[1]`` 会被前端误当成引用标记。

    但流式 token 的边界与标签边界无关：结束标签可能被切成两半两次到达。
    对单个 chunk 做正则必然漏掉被切断的标签，碎片会泄漏到界面上。

    所以这里用状态机：只输出确定安全的部分，把可能构成"半个标签"的尾巴
    留在缓冲区等下一个 chunk。代价是缓冲区最多积压 ``len(标签)-1`` 个字符
    （结束标签 8 字符 → 最多 7 个），流式体验上无感知。

    Note:
        标签用 ``chr()`` 拼出尖括号，而不是写成完整字面量 ——
        这几个字符在文本往返（写入、编码转换、复制粘贴）中容易被改写成
        Unicode 近似字符，那样 ``find()`` 就永远匹配不上，剥离会**静默失效**，
        而且从代码上完全看不出来。用字符码构造保证一定是 ASCII。
    """

    OPEN_TAG = chr(60) + "think" + chr(62)
    CLOSE_TAG = chr(60) + "/think" + chr(62)

    def __init__(self, enabled: bool = True) -> None:
        self._enabled = enabled
        self._buffer = ""       # 尚未确定能否输出的尾部
        self._inside = False    # 当前是否在思考块内

    def feed(self, text: str) -> str:
        """送入一段增量，返回可以安全输出的部分。"""
        if not self._enabled:
            return text

        self._buffer += text
        out: list[str] = []

        while True:
            if self._inside:
                index = self._buffer.find(self.CLOSE_TAG)
                if index == -1:
                    self._buffer = _keep_tail(self._buffer, self.CLOSE_TAG)
                    return "".join(out)
                self._buffer = self._buffer[index + len(self.CLOSE_TAG):]
                self._inside = False
            else:
                index = self._buffer.find(self.OPEN_TAG)
                if index == -1:
                    keep = len(self.OPEN_TAG) - 1
                    if len(self._buffer) > keep:
                        out.append(self._buffer[: len(self._buffer) - keep])
                        self._buffer = self._buffer[len(self._buffer) - keep:]
                    return "".join(out)
                out.append(self._buffer[:index])
                self._buffer = self._buffer[index + len(self.OPEN_TAG):]
                self._inside = True

    def flush(self) -> str:
        """流结束时调用，返回剩余可输出内容（未闭合的思考块直接丢弃）。"""
        if not self._enabled:
            return ""
        remainder = "" if self._inside else self._buffer
        self._buffer = ""
        self._inside = False
        return remainder


def strip_thinking(text: str, enabled: bool = True) -> str:
    """一次性剥离完整文本里的思考块（非流式场景）。"""
    if not enabled:
        return text
    stripper = ThinkingStripper()
    return stripper.feed(text) + stripper.flush()


def _keep_tail(text: str, tag: str) -> str:
    """保留末尾可能与标签前缀重合的部分（长度 len(tag)-1）。"""
    keep = len(tag) - 1
    return text[-keep:] if keep > 0 else ""


def _chunk_text(content: Any) -> str:
    """从消息内容里抽取纯文本（兼容 str 与内容块列表两种形态）。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part if isinstance(part, str) else (part.get("text") or "")
            for part in content
            if isinstance(part, (str, dict))
        )
    return ""
