"""Server-Sent Events (SSE) 协议工具。

## 为什么用手写 SSE 而不是 WebSocket

本场景是**单向**的服务端推送（服务端持续吐 token，客户端只发一次请求），
SSE 正好匹配：
  * 基于普通 HTTP，无需协议升级，天然穿过大多数代理与防火墙；
  * 浏览器有内置的断线重连语义（EventSource）；
  * 服务端实现就是一个"持续返回文本"的响应，无需维护连接状态机。

WebSocket 是双向通道，用在这里属于能力过剩，且要额外处理心跳、重连与
连接生命周期管理。

## 帧格式

SSE 是纯文本协议，一个事件帧由若干字段行加一个空行结束::

    event: token
    data: {"text": "你好"}

    （空行表示本帧结束）

字段值中的换行必须避免，否则会被解析成新字段。因此这里统一用
``json.dumps`` 序列化负载 —— JSON 会把 ``\\n`` 转义成 ``\\n`` 字面量，
天然保证了单行，也顺便解决了引号转义问题。
"""

from __future__ import annotations

import json
from typing import Any

# 事件类型常量。集中定义避免前后端拼错字符串 —— 这类错误在运行时
# 表现为"消息静默丢失"，非常难查。
EVENT_SOURCES = "sources"
"""检索结果元数据，在 token 流之前发送，让前端先渲染引用。"""

EVENT_TOKEN = "token"
"""文本增量。"""

EVENT_DONE = "done"
"""正常结束标志。"""

EVENT_ERROR = "error"
"""错误通知。"""


def sse_frame(event: str, payload: Any) -> str:
    """把事件类型与负载编码成一个 SSE 帧。

    Args:
        event: 事件类型（见本模块的 EVENT_* 常量）。
        payload: 任意可 JSON 序列化的对象。

    Returns:
        形如 ``"event: token\\ndata: {...}\\n\\n"`` 的完整帧字符串。

    Note:
        ``ensure_ascii=False`` 是刻意的：中文直接以 UTF-8 输出而不是
        ``\\uXXXX`` 转义，可显著减小响应体积（中文场景下约减少 3 倍），
        也让 ``curl`` 调试时肉眼可读。
    """
    # separators 去掉 JSON 中多余空格，进一步压缩体积
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {data}\n\n"


# SSE 响应的标准响应头
SSE_HEADERS: dict[str, str] = {
    # 关闭 Nginx 等反向代理的缓冲，否则 token 会被攒起来一次性下发，
    # 流式效果完全消失 —— 这是线上最常见的"流式不生效"原因
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    # 显式禁用代理缓冲（Nginx 专用头，其他代理忽略即可）
    "X-Accel-Buffering": "no",
}

# SSE 媒体类型。charset 必不可少，否则中文可能被按 latin-1 解析而乱码
SSE_MEDIA_TYPE = "text/event-stream; charset=utf-8"
