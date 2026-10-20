"""
主流程：检索 → 重排 → 组装提示词 → 流式生成

    api.py  →  RAG.aretrieve()  →  store.dense_search / sparse_search / fuse
            →  RAG.astream()    →  llm.astream
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

SYSTEM_PROMPT = """你是一个严谨的知识库问答助手。你的回答必须完全基于用户提供的【参考资料】。规则：
1. 只使用【参考资料】中的信息作答，不要用你自己的先验知识补充或推测。
2. 每个结论后标注来源编号，格式为 [1]、[2]，可同时标注多个如 [1][3]。
3. 如果资料不足以回答，直接说明"根据已有资料无法回答该问题"，禁止编造。
4. 如果资料之间矛盾，指出矛盾之处并说明你的倾向及理由。
5. 使用与提问相同的语言，需要列举时用有序列表。
6. 不要复述这些规则，直接给出答案。"""

# 检索为空时的兜底文案
NO_CONTEXT_REPLY = (
    "我在当前知识库中没有检索到与该问题相关的内容。"
    "请换个说法，或先向知识库补充相关文档。"
)


# ===========================================================================
# 主类
# ===========================================================================
class RAG:
    """检索、重排、生成"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.store = MilvusStore(settings)

        # 嵌入模型：用于把查询和分块变成向量
        self.embedder = OllamaEmbeddings(
            base_url=settings.ollama_base_url,
            model=settings.embedding_model,
        )
        # 对话模型：用于生成答案
        #
        # reasoning=True 不是"让模型开始思考"，而是让**库把思考内容交出来**：
        # langchain_ollama 默认（reasoning=None）不会把 Ollama 返回的
        # message.thinking 写进 additional_kwargs，思考会被它直接丢弃 ——
        # 表现为"模型明明思考了，但界面上完全看不到"，而且没有任何报错。
        # 只有需要展示时才打开，避免给不支持思考的模型传这个参数。
        self.llm = ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.chat_model,
            temperature=settings.temperature,
            reasoning=settings.show_thinking,
        )

        self._reranker: _Reranker | None = None
        self.rerank_warning: str | None = None
        """重排加载失败时的原因，会显示在健康检查与前端告警条里"""
        self.rerank_device: str | None = None
        """重排实际使用的设备（cpu / cuda / mps），加载成功后才有值"""

    # ------------------------------------------------------------------
    # 启动 / 关闭
    # ------------------------------------------------------------------
    def start(self) -> None:
        """连接 Milvus 并加载重排模型"""
        _apply_hf_env(self.settings)
        self.store.connect()

        if not self.settings.rerank_enabled:
            logger.info("重排已关闭（RERANK_ENABLED=false）")
            return

        try:
            self._reranker = _Reranker(self.settings)
            self._reranker.load()
            self.rerank_device = self._reranker.device
        except Exception as exc:
            self.rerank_warning = f"重排模型加载失败，已降级为不重排：{type(exc).__name__}: {exc}"
            logger.exception("重排模型加载失败")
            self._reranker = None
            self.rerank_device = None

    def close(self) -> None:
        self.store.close()

    @property
    def rerank_enabled(self) -> bool:
        return self._reranker is not None

    # ------------------------------------------------------------------
    # 检索 + 重排
    # ------------------------------------------------------------------
    async def aretrieve(self, question: str) -> list[Hit]:
        """检索入口"""
        return await asyncio.to_thread(self.retrieve, question)

    def retrieve(self, question: str) -> list[Hit]:
        """
        完整检索：向量化 → 双分支召回 → RRF 融合 → 交叉编码器重排
        """
        question = (question or "").strip()
        if not question: return list()

        # 1) 查询向量化
        vector = self.embedder.embed_query(question)

        # 2) 两个分支各召回一次
        dense = self.store.dense_search(vector, self.settings.dense_top_k)
        sparse = self.store.sparse_search(question, self.settings.sparse_top_k)

        # 3) 按名次融合（RRF）
        candidates = fuse(dense, sparse, k=self.settings.rrf_k)
        if not candidates: return list()

        # 4) 截断到重排预算
        candidates = candidates[: self.settings.rerank_top_n]

        # 5) 精排后取最终条数
        return self._rerank(question, candidates)

    def _rerank(self, question: str, hits: list[Hit]) -> list[Hit]:
        """用交叉编码器重排，返回前 final_top_k 条"""
        limit = self.settings.final_top_k
        if self._reranker is None: return hits[:limit]

        scores = self._reranker.score(question, [h.text for h in hits])
        # 按重排分数降序；分数相同时保持原顺序，保证结果可复现
        order = sorted(range(len(hits)), key=lambda i: (-scores[i], i))
        result = list()
        for index in order[:limit]:
            hit = hits[index]
            hit.score = scores[index]
            hit.score_kind = "rerank"
            result.append(hit)
        return result

    # ------------------------------------------------------------------
    # 生成
    # ------------------------------------------------------------------
    async def astream(
        self, question: str, hits: list[Hit]
    ) -> AsyncIterator[tuple[str, str]]:
        """流式生成答案，逐段产出 ``(类型, 文本)``。

        推理模型（deepseek-r1 等）会产生思考内容，这里始终把它和答案**拆开**：
          * ``("answer", 文本)``   —— 最终答案；
          * ``("thinking", 文本)`` —— 思考过程。

        拆分与"是否展示"是两件事：思考内容无论如何都不能混进答案，
        但要不要给用户看由 ``SHOW_THINKING`` 决定（见 ``settings.show_thinking``）。

        ## 思考内容有两个来源，都要处理

        1. **独立字段** ``additional_kwargs["reasoning_content"]`` ——
           新版 Ollama 对支持思考的模型（如 ``deepseek-r1``）把思考放在
           ``message.thinking`` 里，``content`` 在思考阶段是空的。
           ``langchain_ollama`` 会把它转存到 ``reasoning_content``。
        2. **正文里的标签** ``<think>…</think>`` ——
           老版本 Ollama 或其它推理模型会直接混在 ``content`` 里。

        只处理其中一种都会漏：只认标签时，字段型思考会被**静默丢弃**
        （界面上"看不到思考过程"且没有任何报错）。
        """
        messages = build_messages(question, hits, self.settings.max_context_chars)
        stripper = ThinkingStripper()
        show = self.settings.show_thinking

        async for chunk in self.llm.astream(messages):
            # 来源 1：独立字段
            field_thinking = str(chunk.additional_kwargs.get("reasoning_content") or "")

            # 来源 2：正文标签
            raw = _chunk_text(chunk.content)
            answer, tag_thinking = stripper.feed(raw) if raw else ("", "")

            thinking = field_thinking + tag_thinking
            if thinking and show:
                yield "thinking", thinking
            if answer:
                yield "answer", answer

        answer, tag_thinking = stripper.flush()
        if tag_thinking and show:
            yield "thinking", tag_thinking
        if answer:
            yield "answer", answer

    async def answer(self, question: str, hits: list[Hit]) -> str:
        """一次性生成完整答案（脚本与测试用；面向用户应走 SSE）。"""
        messages = build_messages(question, hits, self.settings.max_context_chars)
        response = await self.llm.ainvoke(messages)
        return strip_thinking(_chunk_text(response.content))


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
        # PDF 标页码（模型可以据此在答案里写"第几页"），其它标分块序号
        where = f"第 {hit.page} 页" if hit.page else f"第 {hit.chunk_index + 1} 块"
        block = f"[{index}] 来源: {name} · {where}\n{hit.text.strip()}"

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
    """交叉编码器重排器"""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: Any = None
        self.device = "cpu"
        """实际加载到的设备，供 /health 展示"""

    def load(self) -> None:
        """
        加载模型

        三级加载顺序：
          1. 按 HF 缓存定位，跳过仓库解析与 etag 校验
          2. 按仓库名加载但带 ``local_files_only=True``
          3. 联网下载
        """
        from sentence_transformers import CrossEncoder

        name = self._settings.reranker_model
        self.device = resolve_device(self._settings.reranker_device)
        logger.info("重排模型将加载到设备：%s", self.device)
        kwargs = {"device": self.device, "max_length": 512}

        if self._settings.reranker_offline_first:
            snapshot = _resolve_snapshot_dir(name)
            if snapshot is not None:
                try:
                    logger.info("从本地快照加载重排模型：%s", snapshot)
                    self._model = CrossEncoder(str(snapshot), **kwargs)
                    logger.info("重排模型已加载")
                    return
                except Exception:
                    logger.warning("本地快照加载失败，改试离线模式", exc_info=True)
            try:
                self._model = CrossEncoder(name, local_files_only=True, **kwargs)
                logger.info("重排模型已加载")
                return
            except Exception:
                if not self._settings.reranker_allow_download: raise
                logger.warning("本地缓存不可用，联网下载重排模型 %s", name)

        self._model = CrossEncoder(name, local_files_only=False, **kwargs)
        logger.info("重排模型已加载（经由网络/默认路径）")

    def score(self, question: str, documents: list[str]) -> list[float]:
        """给候选文档打相关性分数"""
        if not documents: return list()
        pairs = [(question, doc) for doc in documents]
        return [float(s) for s in self._model.predict(pairs, batch_size=16, show_progress_bar=False)]


def resolve_device(preference: str) -> str:
    """把 ``RERANKER_DEVICE`` 解析成实际可用的设备名。

    ``auto`` 的优先级：CUDA → Apple MPS → CPU。

    ## 为什么不能只看 torch.cuda.is_available()

    装了**纯 CPU 版** torch 时（版本号带 ``+cpu``），``torch.cuda`` 模块存在
    但 ``is_available()`` 恒为 False —— 哪怕机器上有能用的 NVIDIA 显卡。
    这种情况下必须换 CUDA 版的 torch 轮子，改配置是没用的。
    所以这里显式检查，并在用户**手写** cuda 但不可用时抛出可读的错误，
    而不是悄悄退回 CPU 让人以为在用 GPU。

    Args:
        preference: ``auto`` / ``cpu`` / ``cuda`` / ``mps``。

    Returns:
        可直接传给 sentence-transformers 的设备字符串。

    Raises:
        RuntimeError: 显式指定了不可用的设备。
    """
    preference = (preference or "auto").strip().lower()

    try:
        import torch
    except Exception:  # pragma: no cover - 没装 torch 时不该拦住建库
        if preference == "auto":
            return "cpu"
        raise

    def has_cuda() -> bool:
        return bool(torch.cuda.is_available())

    def has_mps() -> bool:
        backend = getattr(torch.backends, "mps", None)
        return bool(backend and backend.is_available())

    if preference == "auto":
        if has_cuda():
            return "cuda"
        if has_mps():
            return "mps"
        return "cpu"

    if preference == "cuda" and not has_cuda():
        raise RuntimeError(
            f"RERANKER_DEVICE=cuda，但当前 torch 无法使用 CUDA（版本 {torch.__version__}）。\n"
            "原因：装的是纯 CPU 版 torch，或驱动/CUDA 版本不匹配。\n"
            "处理：改装 CUDA 版 torch（见 README 的『用 GPU 跑重排』一节），"
            "或把 RERANKER_DEVICE 改成 auto/cpu。"
        )
    if preference == "mps" and not has_mps():
        raise RuntimeError("RERANKER_DEVICE=mps，但当前环境不支持 Apple MPS。")

    return preference


def _apply_hf_env(settings: Settings) -> None:
    """在 HF 相关 import 之前把缓存位置写进环境变量"""
    if settings.hf_home: os.environ["HF_HOME"] = str(Path(settings.hf_home).expanduser())


def _resolve_snapshot_dir(model_name: str) -> Path | None:
    """
    把目标模型解析向 HF 缓存里的快照目录

    缓存布局::
        <hub>/models--<作者>--<模型名>/
            refs/main           # 内容为 commit 哈希
            snapshots/<commit>/ # 真正的模型文件
    """
    if "/" not in model_name: return None
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        root = Path(HF_HUB_CACHE)
    except Exception: return None

    org, _, name = model_name.partition("/")
    repo = root / f"models--{org}--{name}"
    snapshots = repo / "snapshots"
    if not snapshots.is_dir(): return None

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

    def __init__(self) -> None:
        self._buffer = ""       # 尚未确定能否输出的尾部
        self._inside = False    # 当前是否在思考块内

    def feed(self, text: str) -> tuple[str, str]:
        """送入一段增量，返回 ``(正文, 思考)`` 两部分的增量。

        两者互斥：一段文本要么属于正文，要么属于思考块。
        返回空串表示这一段没有对应内容。
        """
        self._buffer += text
        answer: list[str] = []
        thinking: list[str] = []

        while True:
            tag = self.CLOSE_TAG if self._inside else self.OPEN_TAG
            sink = thinking if self._inside else answer
            index = self._buffer.find(tag)

            if index == -1:
                # 找不到完整标签：可以安全输出到"可能构成半个标签"的位置之前
                keep = len(tag) - 1
                if len(self._buffer) > keep:
                    sink.append(self._buffer[: len(self._buffer) - keep])
                    self._buffer = self._buffer[len(self._buffer) - keep:]
                break

            sink.append(self._buffer[:index])
            self._buffer = self._buffer[index + len(tag):]
            self._inside = not self._inside

        return "".join(answer), "".join(thinking)

    def flush(self) -> tuple[str, str]:
        """流结束时调用，返回剩余内容 ``(正文, 思考)``。

        未闭合的思考块（模型被截断）仍归入思考部分 —— 它是推理过程，
        不能混进答案，但拿来展示是合适的。
        """
        answer = "" if self._inside else self._buffer
        thinking = self._buffer if self._inside else ""
        self._buffer = ""
        self._inside = False
        return answer, thinking


def strip_thinking(text: str) -> str:
    """一次性剥离完整文本里的思考块，只返回正文（非流式场景）。"""
    stripper = ThinkingStripper()
    answer, _thinking = stripper.feed(text)
    tail, _ = stripper.flush()
    return answer + tail


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
