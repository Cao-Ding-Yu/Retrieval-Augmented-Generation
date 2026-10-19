"""提示词构造。

## RAG 提示词的三个要点

1. **约束在上下文内作答**：明确要求"仅依据参考资料回答"。没有这条约束，
   模型会用预训练知识补全，答案看起来更流畅，但无法溯源、也无法通过
   更新知识库来纠正 —— 那就退化成了一个普通聊天机器人。

2. **强制引用编号**：把上下文编号成 [1][2]…，要求模型在句末标注来源。
   这样前端可以把 [1] 渲染成可点击的引用卡片，用户能自己核对。
   这也是"可验证"的基础。

3. **允许说不知道**：必须显式给出"资料中没有提到"的出口。
   否则模型在检索失败时会硬编一个答案 —— 幻觉在 RAG 里最主要的表现就是
   "上下文不足但模型不肯说不知道"。

## 为什么把上下文放进 user 消息而不是 system

部分本地模型（尤其是量化后的小模型）对 system 消息的遵循度弱于 user 消息。
把参考资料与问题一起放在 user 侧，能让"用这些资料回答这个问题"的关联更紧密，
在 7B 级别的模型上实测更稳定。
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from rag_app.models import RetrievedChunk

logger = logging.getLogger(__name__)

# 系统提示：定义角色与硬约束
SYSTEM_PROMPT = """你是一个严谨的知识库问答助手。你的回答必须完全基于用户提供的【参考资料】。

请遵守以下规则：
1. 只使用【参考资料】中的信息作答，不要使用你自己的先验知识补充或推测。
2. 每个结论后面必须标注来源编号，格式为 [1]、[2]，可同时标注多个，例如 [1][3]。
3. 如果【参考资料】不足以回答问题，直接说明"根据已有资料无法回答该问题"，并指出还缺少什么信息。禁止编造。
4. 如果【参考资料】之间互相矛盾，请指出矛盾之处，并说明你倾向于哪一个及理由。
5. 回答使用与提问相同的语言，结构清晰；需要列举时使用有序列表。
6. 不要复述这些规则，也不要提及"参考资料"这个说法本身，直接给出答案。"""

# 找不到答案时的兜底文案（检索为空时直接返回，不调用模型，省一次推理）
NO_CONTEXT_REPLY = "我在当前知识库中没有检索到与该问题相关的内容。请尝试换个说法，或先向知识库补充相关文档。"


def _format_source_label(chunk: RetrievedChunk, index: int) -> str:
    """生成单条上下文的来源标签

    形如 ``[1] 来源: docs/manual.pdf · 第 3 块``。页码信息（如果有）优先于
    分块序号，因为它对用户更直观。
    """
    parts = [f"[{index}]"]

    if chunk.source:
        # 只取文件名，完整路径在对话里又长又没用
        name = chunk.source.replace("\\", "/").split("/")[-1]
        parts.append(f"来源: {name}")
    if chunk.doc_id:
        # 给出短 ID 便于在日志里定位是哪个文档
        parts.append(f"文档: {chunk.doc_id[:8]}")
    if chunk.chunk_index >= 0:
        parts.append(f"第 {chunk.chunk_index + 1} 块")

    return " · ".join(parts)


def format_context(chunks: Sequence[RetrievedChunk], max_chars: int) -> str:
    """把检索到的分块拼成带编号的参考资料块

    Args:
        chunks: 已按相关性降序排列的检索结果。
        max_chars: 总字符数上限。超限时从**末尾**截断 ——
            因为列表已按相关性排序，末尾的块最不重要，优先丢弃它们
            比平均截断每条更合理。

    Returns:
        可直接嵌入提示词的参考文本。chunks 为空时返回空串。
    """
    if not chunks:
        return ""

    blocks: list[str] = []
    used = 0

    for index, chunk in enumerate(chunks, start=1):
        label = _format_source_label(chunk, index)
        text = chunk.text.strip()
        block = f"{label}\n{text}"

        # 加上分隔符后的实际占用长度
        projected = used + len(block) + 2  # "\n\n" 两个字符
        if projected > max_chars:
            # 若这是第一块就超限，说明单块本身就超长。
            # 此时仍截取一部分 —— 完全不放上下文会让模型无据可依，
            # 放入截断内容至少能覆盖开头部分的信息。
            if not blocks:
                remaining = max(0, max_chars - used)
                blocks.append(block[:remaining])
                logger.warning(
                    "首条上下文即超过 max_context_chars=%d，已截断", max_chars
                )
            else:
                logger.info(
                    "上下文达到 %d 字符上限，已保留前 %d 条，丢弃剩余 %d 条",
                    max_chars,
                    len(blocks),
                    len(chunks) - len(blocks),
                )
            break

        blocks.append(block)
        used = projected

    return "\n\n".join(blocks)


def build_messages(
    question: str,
    chunks: Sequence[RetrievedChunk],
    max_context_chars: int,
) -> list[BaseMessage]:
    """构造送给对话模型的完整消息列表。

    Args:
        question: 用户问题。
        chunks: 检索到的上下文分块（按相关性降序）。
        max_context_chars: 上下文总长度上限。

    Returns:
        ``[SystemMessage, HumanMessage]``。
        检索为空时 HumanMessage 里会明确说明"未检索到资料"，
        让模型据此回复"无法回答"而不是自由发挥。
    """
    context = format_context(chunks, max_context_chars)

    if context:
        user_content = (
            f"【参考资料】\n{context}\n\n"
            f"【问题】\n{question}\n\n"
            "请依据上述参考资料回答问题，并在结论处标注来源编号。"
        )
    else:
        # 没有上下文时不留白，明确告知模型检索无结果
        user_content = (
            "【参考资料】\n（本次检索未返回任何相关内容）\n\n"
            f"【问题】\n{question}\n\n"
            "由于没有检索到参考资料，请直接说明无法回答，不要使用你自己的知识作答。"
        )

    return [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]
