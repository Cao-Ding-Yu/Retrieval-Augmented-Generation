"""HTTP 接口与 SSE 协议的测试。

这里用 FastAPI 的 TestClient 直接驱动应用，但不启动 lifespan
（即不用 ``with TestClient(app)``），因此**不会真的连接 Milvus 或加载模型**。
所有依赖通过 ``app.dependency_overrides`` 替换为可控的假实现。

覆盖重点：
  * SSE 帧格式是否符合协议（前端解析依赖它，格式错了表现是"消息静默丢失"）；
  * 事件顺序是否合理（sources 必须在 token 之前）；
  * 降级路径（服务未就绪）是否返回明确的 503 而不是 500；
  * 检索为空时是否走兜底文案而不是调用模型。
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from rag_app.api.app import create_app
from rag_app.deps import get_context
from rag_app.models import IngestReport
from rag_app.retrieval.hybrid import FusionMode
from tests.conftest import FakeChat, make_chunk


# ---------------------------------------------------------------------------
# 测试替身
# ---------------------------------------------------------------------------
class FakeRetriever:
    """假检索器：返回预置分块。"""

    def __init__(self, chunks=None, fusion_mode: FusionMode = FusionMode.CLIENT):
        self.chunks = chunks or []
        self.fusion_mode = fusion_mode
        self.received_queries: list[str] = []

    async def retrieve(self, query: str):
        self.received_queries.append(query)
        return list(self.chunks)


class FailingChat:
    """生成时报错的假对话模型，用于验证 error 事件。"""

    async def astream_answer(self, question, chunks):
        raise RuntimeError("模拟模型推理失败")
        yield  # pragma: no cover - 让函数成为异步生成器


class FakeIngest:
    """假入库服务。"""

    def __init__(self, report: IngestReport | None = None):
        self.report = report or IngestReport(
            files_scanned=1, files_loaded=1, chunks_written=7
        )
        self.received_paths: list[str] = []

    async def ingest_path(self, path, recursive=True):
        self.received_paths.append(str(path))
        return self.report


class FakeStore:
    """假存储：仅用于 /stats 与 /health。"""

    server_version = "v2.5.4"

    def num_entities(self) -> int:
        return 42


# ---------------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------------
@pytest.fixture
def build_client(settings, make_context):
    """返回一个工厂：按需构造带假依赖的 TestClient。

    用法::

        client = build_client(retriever=FakeRetriever([...]))
    """

    def _factory(
        *,
        ready: bool = True,
        retriever=None,
        chat=None,
        ingest=None,
        store=None,
        startup_error: str | None = None,
    ) -> TestClient:
        ctx = make_context(
            settings,
            ready=ready,
            retriever=retriever if retriever is not None else FakeRetriever(),
            chat=chat if chat is not None else FakeChat(),
            ingest=ingest,
            store=store if store is not None else FakeStore(),
            startup_error=startup_error,
        )

        app = create_app(settings)
        # 直接挂 state：/health 是直接读 app.state.ctx 的（不走依赖注入）
        app.state.ctx = ctx
        # 覆盖依赖：业务路由通过 get_context 获取上下文
        app.dependency_overrides[get_context] = lambda: ctx

        # 不进入 context manager，因此 lifespan 不会执行 ——
        # 这正是我们想要的：不连 Milvus、不加载重排模型。
        return TestClient(app)

    return _factory


def parse_sse(body: str) -> list[tuple[str, dict]]:
    """把 SSE 响应体解析成 (事件名, 负载) 列表。

    实现与前端 web/index.html 里的 parseFrame 逻辑保持一致 ——
    测试解析器与生产解析器语义相同，才能真正验证协议。
    """
    events: list[tuple[str, dict]] = []

    for raw_frame in body.split("\n\n"):
        if not raw_frame.strip():
            continue
        event_name = "message"
        data_parts: list[str] = []

        for line in raw_frame.split("\n"):
            if not line or line.startswith(":"):
                continue
            if ":" not in line:
                continue
            field, _, value = line.partition(":")
            value = value[1:] if value.startswith(" ") else value
            if field == "event":
                event_name = value.strip()
            elif field == "data":
                data_parts.append(value)

        if data_parts:
            events.append((event_name, json.loads("\n".join(data_parts))))

    return events


# ---------------------------------------------------------------------------
# 健康检查与降级
# ---------------------------------------------------------------------------
class TestHealth:
    def test_ready_service_reports_ok(self, build_client):
        client = build_client(ready=True)

        resp = client.get("/health")

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["ready"] is True
        assert body["milvus"]["server_version"] == "v2.5.4"
        assert body["milvus"]["entities"] == 42

    def test_degraded_service_still_returns_200_with_reason(self, build_client):
        """依赖不可用时状态码仍是 200，但 ready=false 且带原因。

        刻意不用 503：很多编排系统看到 503 会反复重启容器，
        反而让"Milvus 没起来"这个真实原因被重启日志淹没。
        """
        client = build_client(ready=False, startup_error="Milvus 初始化失败：连接被拒绝")

        resp = client.get("/health")

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "degraded"
        assert body["ready"] is False
        assert "连接被拒绝" in body["error"]

    def test_retrieval_config_is_exposed(self, build_client):
        """健康检查应暴露检索配置，便于确认线上实际生效的参数。"""
        client = build_client()

        body = client.get("/health").json()

        assert body["retrieval"]["rrf_k"] == 60
        assert body["retrieval"]["embedding_dim"] == 1024
        assert "fusion_mode" in body["retrieval"]


class TestNotReadyResponses:
    """未就绪时业务接口应返回 503 并说明原因。"""

    def test_chat_returns_503(self, build_client):
        client = build_client(ready=False, startup_error="Milvus 初始化失败")

        resp = client.post("/chat", json={"message": "你好"})

        assert resp.status_code == 503
        assert "Milvus" in resp.json()["detail"]

    def test_search_returns_503(self, build_client):
        client = build_client(ready=False, startup_error="Milvus 初始化失败")

        resp = client.post("/search", json={"query": "你好"})

        assert resp.status_code == 503

    def test_stats_returns_503(self, build_client):
        client = build_client(ready=False, startup_error="未就绪")

        assert client.get("/stats").status_code == 503


# ---------------------------------------------------------------------------
# SSE 流式对话
# ---------------------------------------------------------------------------
class TestChatStream:
    def test_response_headers_are_sse_friendly(self, build_client):
        """响应头必须能穿透反向代理，否则流式会退化成一次性返回。"""
        client = build_client(retriever=FakeRetriever([make_chunk("k1")]))

        resp = client.post("/chat", json={"message": "问题"})

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        # 关键：禁用 Nginx 缓冲，否则 token 会被攒起来一次性下发
        assert resp.headers.get("x-accel-buffering") == "no"

    def test_event_sequence_sources_then_tokens_then_done(self, build_client):
        """事件顺序：sources -> token* -> done。

        sources 必须先发，前端才能立刻渲染引用卡片，
        让用户在等待生成时就有内容可看。
        """
        retriever = FakeRetriever([make_chunk("k1", text="相关内容")])
        client = build_client(
            retriever=retriever,
            chat=FakeChat(deltas=["第一段", "第二段"]),
        )

        resp = client.post("/chat", json={"message": "问题"})
        events = parse_sse(resp.text)
        names = [name for name, _ in events]

        assert names[0] == "sources", "sources 必须是第一个事件"
        assert names[-1] == "done", "done 必须是最后一个事件"
        assert names.count("token") == 2

    def test_token_events_concatenate_to_full_answer(self, build_client):
        """把所有 token 事件拼起来应等于完整答案。"""
        client = build_client(
            retriever=FakeRetriever([make_chunk("k1")]),
            chat=FakeChat(deltas=["混合", "检索", "很好"]),
        )

        resp = client.post("/chat", json={"message": "问题"})
        tokens = [d["text"] for name, d in parse_sse(resp.text) if name == "token"]

        assert "".join(tokens) == "混合检索很好"

    def test_sources_event_carries_ranking_detail(self, build_client):
        """sources 事件必须带上分支名次与命中标记，供前端解释排序。"""
        chunk = make_chunk("k1", source="manual.pdf", chunk_index=3)
        chunk.dense_rank = 0
        chunk.sparse_rank = 2
        chunk.rrf_score = 0.032
        client = build_client(retriever=FakeRetriever([chunk]))

        resp = client.post("/chat", json={"message": "问题"})
        events = parse_sse(resp.text)
        sources_event = next(d for name, d in events if name == "sources")

        assert sources_event["count"] == 1
        source = sources_event["sources"][0]
        assert source["dense_rank"] == 0
        assert source["sparse_rank"] == 2
        assert source["is_hybrid_hit"] is True
        assert source["source"] == "manual.pdf"
        assert source["chunk_index"] == 3

    def test_sources_can_be_disabled(self, build_client):
        """include_sources=false 时不应发送 sources 事件。"""
        client = build_client(retriever=FakeRetriever([make_chunk("k1")]))

        resp = client.post("/chat", json={"message": "问题", "include_sources": False})
        names = [name for name, _ in parse_sse(resp.text)]

        assert "sources" not in names
        assert "done" in names

    def test_snippet_is_truncated(self, build_client):
        """引用片段应被截断，避免超长正文撑大响应体。"""
        long_text = "长" * 500
        client = build_client(retriever=FakeRetriever([make_chunk("k1", text=long_text)]))

        resp = client.post("/chat", json={"message": "问题"})
        events = parse_sse(resp.text)
        sources_event = next(d for name, d in events if name == "sources")

        snippet = sources_event["sources"][0]["snippet"]
        assert len(snippet) <= 281  # 280 + 省略号
        assert snippet.endswith("…")

    def test_done_event_reports_answer_length(self, build_client):
        """done 事件带上答案长度，便于前端做校验与埋点。"""
        client = build_client(
            retriever=FakeRetriever([make_chunk("k1")]),
            chat=FakeChat(deltas=["中文"]),  # 2 个字符
        )

        resp = client.post("/chat", json={"message": "问题"})
        done = next(d for name, d in parse_sse(resp.text) if name == "done")

        assert done["answer_chars"] == 2

    def test_empty_retrieval_skips_model_and_returns_fallback(self, build_client):
        """检索为空时应直接返回兜底文案，不调用模型。

        这一步既省一次推理（本地 7B 模型一次生成要好几秒），
        也从机制上避免"没有依据却硬编答案"。
        """
        client = build_client(retriever=FakeRetriever([]), chat=FailingChat())

        resp = client.post("/chat", json={"message": "问题"})
        events = parse_sse(resp.text)

        assert [name for name, _ in events] == ["sources", "token", "done"]
        token_text = events[1][1]["text"]
        assert "没有检索到" in token_text

    def test_generation_failure_emits_error_event(self, build_client):
        """生成阶段异常应发出 error 事件，而不是直接断开连接。

        响应头早已发出（HTTP 200），无法再改状态码，
        因此只能靠事件类型把失败告知前端。
        """
        client = build_client(retriever=FakeRetriever([make_chunk("k1")]), chat=FailingChat())

        resp = client.post("/chat", json={"message": "问题"})
        events = parse_sse(resp.text)

        assert any(name == "error" for name, _ in events)
        error_payload = next(d for name, d in events if name == "error")
        assert "RuntimeError" in error_payload["message"]

    def test_validation_rejects_empty_message(self, build_client):
        """空消息应被 Pydantic 校验拦下，返回 422。"""
        client = build_client()

        assert client.post("/chat", json={"message": ""}).status_code == 422


# ---------------------------------------------------------------------------
# 非流式与纯检索
# ---------------------------------------------------------------------------
class TestChatSync:
    def test_returns_answer_and_sources(self, build_client):
        client = build_client(
            retriever=FakeRetriever([make_chunk("k1")]),
            chat=FakeChat(deltas=["完整", "答案"]),
        )

        resp = client.post("/chat/sync", json={"message": "问题"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["answer"] == "完整答案"
        assert len(body["sources"]) == 1

    def test_no_context_returns_fallback_without_model(self, build_client):
        client = build_client(retriever=FakeRetriever([]), chat=FailingChat())

        resp = client.post("/chat/sync", json={"message": "问题"})

        assert resp.status_code == 200
        assert "没有检索到" in resp.json()["answer"]


class TestSearchEndpoint:
    def test_returns_ranking_detail(self, build_client):
        """纯检索接口应返回完整打分明细，用于排查"答案不对"的根因。"""
        chunk = make_chunk("k1")
        chunk.dense_rank = 1
        chunk.sparse_rank = None
        chunk.rrf_score = 0.0161
        client = build_client(retriever=FakeRetriever([chunk]))

        resp = client.post("/search", json={"query": "查询"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["query"] == "查询"
        assert body["count"] == 1
        result = body["results"][0]
        assert result["dense_rank"] == 1
        assert result["sparse_rank"] is None
        assert result["is_hybrid_hit"] is False

    def test_reports_effective_fusion_mode(self, build_client):
        """应回显当前生效的融合模式。"""
        client = build_client(retriever=FakeRetriever([], fusion_mode=FusionMode.CLIENT))

        body = client.post("/search", json={"query": "查询"}).json()

        assert body["fusion_mode"] == "client"


# ---------------------------------------------------------------------------
# 入库
# ---------------------------------------------------------------------------
class TestIngestEndpoints:
    def test_missing_path_returns_404(self, build_client, tmp_path):
        """路径不存在应返回 404，而不是 500。"""
        ingest = FakeIngest()
        client = build_client(ingest=ingest)

        resp = client.post("/ingest", json={"path": str(tmp_path / "不存在")})

        assert resp.status_code == 404
        assert ingest.received_paths == []

    def test_existing_path_triggers_ingest(self, build_client, tmp_path):
        """路径存在时应调用入库服务并回传统计。"""
        target = tmp_path / "doc.txt"
        target.write_text("内容", encoding="utf-8")

        ingest = FakeIngest(
            IngestReport(files_scanned=1, files_loaded=1, chunks_written=5)
        )
        client = build_client(ingest=ingest)

        resp = client.post("/ingest", json={"path": str(target)})

        assert resp.status_code == 200
        body = resp.json()
        assert body["chunks_written"] == 5
        assert ingest.received_paths == [str(target)]

    def test_upload_saves_file_and_ingests(self, build_client):
        """上传接口应把文件落盘后再入库，并返回统计。"""
        ingest = FakeIngest()
        client = build_client(ingest=ingest)

        resp = client.post(
            "/ingest/upload",
            files={"file": ("note.txt", b"hello", "text/plain")},
        )

        assert resp.status_code == 200
        assert len(ingest.received_paths) == 1
        # 落盘路径应包含 uploads 子目录
        assert "uploads" in ingest.received_paths[0]

    def test_upload_sanitizes_path_traversal(self, build_client, settings):
        """恶意文件名中的目录成分必须被剥掉，防止目录穿越写文件。"""
        ingest = FakeIngest()
        client = build_client(ingest=ingest)

        resp = client.post(
            "/ingest/upload",
            files={"file": ("../../evil.txt", b"payload", "text/plain")},
        )

        assert resp.status_code == 200
        saved = ingest.received_paths[0]
        # 只保留文件名部分，且必须落在配置的 data 目录内
        assert saved.endswith("evil.txt")
        assert ".." not in saved
        assert str(settings.data_dir) in saved or "uploads" in saved


class TestStats:
    def test_reports_collection_and_config(self, build_client):
        client = build_client()

        body = client.get("/stats").json()

        assert body["entities"] == 42
        assert body["server_version"] == "v2.5.4"
        assert body["embedding_dim"] == 1024
