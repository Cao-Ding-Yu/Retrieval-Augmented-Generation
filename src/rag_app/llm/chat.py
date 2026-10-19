"""对话模型封装：非流式与流式两种调用方式。

## 为什么流式是必须的

7B 模型在 CPU/中端 GPU 上生成一段 300 字的回答大约需要十几秒到几十秒。
如果等生成完再一次性返回，用户面对的是长时间空白页面，体验接近"卡死"。
流式输出把首个 token 的等待时间（TTFT）压缩到 1~2 秒内，
用户能立刻看到响应在增长，主观等待感大幅下降 —— 这是产品体验的关键，
也是本项目的核心工程亮点之一。

## 关于 token 提取

LangChain 的 ``astream()`` 产出 ``AIMessageChunk``，其 ``content`` 字段在
不同模型/版本下可能是字符串，也可能是内容块列表（多模态场景）。
这里统一做了归一化，只抽取文本，避免把非文本块直接吐给前端。

## 关于推理模型的思考块

``deepseek-r1`` 这类推理模型会把推理链混在正文里返回，因此本模块提供
:class:`ThinkingStripper` 做**流式安全**的剥离。原因与实现见该类的文档。
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Sequence
from typing import Any

from langchain_core.messages import BaseMessage

from rag_app.config import Settings
from rag_app.llm.prompts import build_messages
from rag_app.models import RetrievedChunk

logger = logging.getLogger(__name__)


def _extract_text(content: Any) -> str:
    """从消息内容中抽取纯文本。

    Args:
        content: ``AIMessageChunk.content``，可能是 str，
            也可能是 ``[{"type": "text", "text": "..."}, ...]`` 形式的块列表。

    Returns:
        抽取出的文本；无法识别时返回空串。
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                # 只取文本块，跳过工具调用、图片等非文本内容
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)

    return ""


class ThinkingStripper:
    """剥离推理模型的思考块，且**支持流式输入**。

    ## 为什么需要它

    ``deepseek-r1`` 这类推理模型会把完整推理链包在一对思考标签之间
    （开始标签是 ``think`` 前面加一个左尖括号，结束标签是 ``/think`` 前面加一个左尖括号）
    混在正文里输出。在 RAG 场景下这有两个问题：

      1. 用户会先看到一大段"让我想想…"的思考内容，界面像坏了；
      2. 与本项目的引用编号机制冲突 —— 思考里出现的 ``[1]`` 会被前端
         误当成引用标记。

    ## 为什么不能用简单的正则替换

    流式输出的 token 边界与标签边界无关。结束标签可能被切成
    "前半截" + "后半截" 两次到达。对单个 chunk 直接做正则，
    会漏掉被切断的标签，导致标签碎片泄漏到界面上。

    因此这里用一个小的状态机：**只输出确定安全的部分，
    把可能是"半个标签"的尾巴留在缓冲区里等下一个 chunk**。

    ## 已知代价

    为了保证不误吐半个标签，缓冲区最多会积压 ``len(标签)-1`` 个字符
    （结束标签 8 字符，即最多 7 个字符）。这在流式体验上无感知，
    但意味着极短的输出会等到 ``flush()`` 才吐出。

    Args:
        enabled: 为 False 时完全透传，不做任何处理。
    """

    # 思考标签：开始 = 左尖括号 + "think" + 右尖括号；结束 = 左尖括号 + "/think" + 右尖括号。
    #
    # 刻意用 chr() 拼出尖括号，而不是写成完整字面量：这几个字符在文本往返
    # （工具写入、编码转换、复制粘贴）中容易被改写成 Unicode 近似字符，
    # 那样 find() 就永远匹配不上模型真实输出 —— 剥离会**静默失效**，
    # 而且从代码上完全看不出来（显示效果一模一样）。
    # 用字符码构造可以保证落到磁盘上的一定是纯 ASCII。
    OPEN_TAG = chr(60) + "think" + chr(62)
    CLOSE_TAG = chr(60) + "/think" + chr(62)

    def __init__(self, enabled: bool = True) -> None:
        self._enabled = enabled
        self._buffer = ""
        """尚未确定能否输出的尾部片段。"""

        self._inside = False
        """当前是否处于思考块内部。"""

    def feed(self, text: str) -> str:
        """送入一段新的文本增量，返回可以安全输出的部分。

        Args:
            text: 模型新产出的文本片段。

        Returns:
            可以立即发给用户的文本（可能为空字符串）。
        """
        if not self._enabled:
            return text

        self._buffer += text
        output: list[str] = []

        while True:
            if self._inside:
                # 在思考块内：找结束标签，找到它之前的内容全部丢弃
                index = self._buffer.find(self.CLOSE_TAG)
                if index == -1:
                    # 没找到。保留可能是半个结束标签的尾巴，其余丢弃
                    self._buffer = self._keep_tail(self._buffer, self.CLOSE_TAG)
                    return "".join(output)

                self._buffer = self._buffer[index + len(self.CLOSE_TAG) :]
                self._inside = False
            else:
                # 在正文中：找开始标签，它之前的内容都是正文
                index = self._buffer.find(self.OPEN_TAG)
                if index == -1:
                    # 没找到开始标签。输出除"可能是半个开始标签"之外的全部内容
                    keep = len(self.OPEN_TAG) - 1
                    if len(self._buffer) > keep:
                        split_at = len(self._buffer) - keep
                        output.append(self._buffer[:split_at])
                        self._buffer = self._buffer[split_at:]
                    return "".join(output)

                output.append(self._buffer[:index])
                self._buffer = self._buffer[index + len(self.OPEN_TAG) :]
                self._inside = True

    def flush(self) -> str:
        """流结束时调用，返回缓冲区里剩余的可输出内容。

        若结束时仍处于思考块内部（模型被截断），剩余内容属于推理过程，直接丢弃。

        Returns:
            可安全输出的尾部文本。
        """
        if not self._enabled:
            return ""

        # 结束时缓冲区里只可能是"看起来像标签开头的正文"，
        # 属于正常内容，应当输出而不是丢弃
        remainder = "" if self._inside else self._buffer
        self._buffer = ""
        self._inside = False
        return remainder

    @staticmethod
    def _keep_tail(text: str, tag: str) -> str:
        """保留 ``text`` 末尾可能与 ``tag`` 前缀重合的部分。

        例如 ``tag`` 长度为 8 时，末尾 7 个字符都可能是半个标签，
        必须留着等后续 chunk 拼上再判断。

        Args:
            text: 当前缓冲内容。
            tag: 目标标签。

        Returns:
            需要保留的尾部子串。
        """
        keep = len(tag) - 1
        if keep <= 0:
            return ""
        return text[-keep:]


def strip_thinking(text: str, enabled: bool = True) -> str:
    """一次性剥离完整文本中的思考块（非流式场景使用）。

    Args:
        text: 完整回答文本。
        enabled: 是否启用剥离。

    Returns:
        剥离思考块后的文本。
    """
    if not enabled:
        return text

    stripper = ThinkingStripper(enabled=True)
    return stripper.feed(text) + stripper.flush()


class RAGChat:
    """RAG 对话服务：把检索结果与问题组装成提示词并调用本地模型。

    Args:
        llm: 已构造的 ChatOllama 实例（或其他兼容的聊天对象）。
        settings: 全局配置。
    """

    def __init__(self, llm: Any, settings: Settings) -> None:
        self._llm = llm
        self._settings = settings

    @property
    def llm(self) -> Any:
        """底层对话模型对象。

        对外暴露是为了让评估模块（RAGAS）能复用同一个模型实例当裁判，
        而不必再构造一个 —— 本地模型重复加载既慢又占内存。
        """
        return self._llm

    @property
    def settings(self) -> Settings:
        """当前配置，便于调用方读取生成参数。"""
        return self._settings

    # ------------------------------------------------------------------
    # 流式
    # ------------------------------------------------------------------
    async def astream_answer(
        self,
        question: str,
        chunks: Sequence[RetrievedChunk],
    ) -> AsyncIterator[str]:
        """流式生成回答，逐段产出文本增量。

        Args:
            question: 用户问题。
            chunks: 检索到的上下文分块。

        Yields:
            文本增量片段。调用方直接拼接即可得到完整回答。
            检索为空时由上层短路处理，这里不会产出内容。
        """
        messages = build_messages(
            question=question,
            chunks=chunks,
            max_context_chars=self._settings.max_context_chars,
        )

        # 推理模型会把思考过程混在正文里，必须边流边剥离（见 ThinkingStripper）
        stripper = ThinkingStripper(enabled=self._settings.strip_thinking)

        try:
            async for chunk in self._llm.astream(messages):
                text = _extract_text(chunk.content)
                if not text:
                    continue

                safe = stripper.feed(text)
                if safe:
                    yield safe

            # 收尾：把缓冲区里剩余的正文吐出去，否则末尾几个字符会丢
            tail = stripper.flush()
            if tail:
                yield tail

        except Exception:
            # 流式过程中出错时，异常无法再通过 HTTP 状态码表达
            # （响应头早已发出），只能记录日志并让连接自然结束。
            # 前端通过"没有收到 done 事件"来识别异常。
            logger.exception("流式生成失败")
            raise

    # ------------------------------------------------------------------
    # 非流式
    # ------------------------------------------------------------------
    async def answer(
        self,
        question: str,
        chunks: Sequence[RetrievedChunk],
    ) -> str:
        """一次性生成完整回答。

        适用于脚本、测试与批量评测场景；面向用户的 HTTP 接口应使用
        :meth:`astream_answer` 以获得更好的交互体验。

        Args:
            question: 用户问题。
            chunks: 检索到的上下文分块。

        Returns:
            完整回答文本（已剥离思考块）。
        """
        messages: list[BaseMessage] = build_messages(
            question=question,
            chunks=chunks,
            max_context_chars=self._settings.max_context_chars,
        )
        response = await self._llm.ainvoke(messages)
        # 非流式路径同样要剥离，否则脚本与接口拿到的答案不一致
        return strip_thinking(
            _extract_text(response.content),
            enabled=self._settings.strip_thinking,
        )
