"""HTTP 接口与 SSE 协议的测试。

用 FastAPI 的 TestClient 驱动应用，但**不启动 lifespan**
（即不用 ``with TestClient(app)``），因此不会真的连 Milvus 或加载模型。
RAG 实例用 :class:`tests.conftest.FakeRAG` 直接塞进 ``app.state``。

重点覆盖：
  * SSE 帧格式与事件顺序（格式错了前端会静默丢消息）；
  * 服务未就绪时返回 503 而不是 500；
  * 检索为空时走兜底文案、不调用模型；
  * 降级原因（重排失败）能出现在 /health 的 warnings 里。
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from rag_app.api import create_app
from tests.conftest import FakeRAG, make_hit


def build_client(settings, rag: FakeRAG | None = None,
                 startup_error: str | None = None) -> TestClient:
    """构造带假 RAG 的 TestClient（不触发 lifespan）。"""
    app = create_app(settings)
    app.state.rag = rag
    app.state.startup_error = startup_error
    return TestClient(app)


def parse_sse(body: str) -> list[tuple[str, dict]]:
    """把 SSE 响应体解析成 (事件名, 负载) 列表。

    实现与前端 web/index.html 里的解析逻辑保持一致 ——
    只有测试解析器与生产解析器语义相同，才真正验证了协议。
    """
    events: list[tuple[str, dict]] = []
    for frame in body.split("\n\n"):
        if not frame.strip():
            continue
        name, data = "message", []
        for line in frame.split("\n"):
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                data.append(line[5:].lstrip())
        if data:
            events.append((name, json.loads("\n".join(data))))
    return events


# ---------------------------------------------------------------------------
# 健康检查
# ---------------------------------------------------------------------------
class TestHealth:
    def test_ready_reports_ok(self, settings):
        client = build_client(settings, FakeRAG(settings))

        body = client.get("/health").json()

        assert body["ready"] is True
        assert body["status"] == "ok"
        assert body["milvus"]["chunks"] == 42
        assert body["retrieval"]["rrf_k"] == 60

    def test_degraded_still_returns_200_with_reason(self, settings):
        """依赖不可用时状态码仍为 200，但 ready=false 并带原因。

        不用 503：很多编排系统看到 503 会反复重启容器，
        反而把真实原因淹没在重启日志里。
        """
        client = build_client(settings, None, startup_error="Milvus 连接被拒绝")

        response = client.get("/health")
        body = response.json()

        assert response.status_code == 200
        assert body["ready"] is False
        assert "连接被拒绝" in body["error"]

    def test_rerank_failure_surfaces_in_warnings(self, settings):
        """重排加载失败的原因必须能被前端读到（用于顶部告警条）。"""
        rag = FakeRAG(settings, rerank_enabled=False, warning="重排模型加载失败：boom")
        client = build_client(settings, rag)

        body = client.get("/health").json()

        assert body["reranker"]["enabled"] is False
        assert any("重排模型加载失败" in w for w in body["warnings"])

    def test_not_ready_chat_returns_503(self, settings):
        client = build_client(settings, None, startup_error="初始化失败")

        response = client.post("/chat", json={"message": "你好"})

        assert response.status_code == 503
        assert "初始化失败" in response.json()["detail"]


# ---------------------------------------------------------------------------
# 流式对话
# ---------------------------------------------------------------------------
class TestChatStream:
    def test_headers_are_sse_friendly(self, settings):
        """必须禁用代理缓冲，否则 token 会被攒起来一次性下发。"""
        client = build_client(settings, FakeRAG(settings, hits=[make_hit("k1")]))

        response = client.post("/chat", json={"message": "问题"})

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert response.headers.get("x-accel-buffering") == "no"

    def test_event_order_sources_then_tokens_then_done(self, settings):
        """sources 必须先发，前端才能立刻渲染引用。"""
        rag = FakeRAG(settings, hits=[make_hit("k1")], deltas=["第一段", "第二段"])
        client = build_client(settings, rag)

        events = parse_sse(client.post("/chat", json={"message": "问题"}).text)
        names = [n for n, _ in events]

        assert names[0] == "sources"
        assert names[-1] == "done"
        assert names.count("token") == 2

    def test_tokens_concatenate_to_answer(self, settings):
        rag = FakeRAG(settings, hits=[make_hit("k1")], deltas=["混合", "检索", "很好"])
        client = build_client(settings, rag)

        events = parse_sse(client.post("/chat", json={"message": "问题"}).text)
        tokens = [d["text"] for n, d in events if n == "token"]

        assert "".join(tokens) == "混合检索很好"

    def test_sources_carry_ranking_detail(self, settings):
        hit = make_hit("k1", source="manual.pdf", chunk_index=3)
        hit.dense_rank, hit.sparse_rank = 0, 2
        client = build_client(settings, FakeRAG(settings, hits=[hit]))

        events = parse_sse(client.post("/chat", json={"message": "问题"}).text)
        sources = next(d for n, d in events if n == "sources")

        assert sources["count"] == 1
        entry = sources["sources"][0]
        assert entry["is_hybrid_hit"] is True
        assert entry["source"] == "manual.pdf"
        assert entry["chunk_index"] == 3

    def test_sources_carry_score_and_kind(self, settings):
        """前端靠 score + score_kind 决定显示"重排 x.xxx"还是"RRF x.xxxxx"。

        这两个字段缺一个，界面上就什么都不显示 —— 曾经因此丢过重排得分。
        """
        hit = make_hit("k1")
        hit.score = 0.9123
        hit.score_kind = "rerank"
        client = build_client(settings, FakeRAG(settings, hits=[hit]))

        events = parse_sse(client.post("/chat", json={"message": "问题"}).text)
        entry = next(d for n, d in events if n == "sources")["sources"][0]

        assert entry["score"] == pytest.approx(0.9123, abs=1e-6)
        assert entry["score_kind"] == "rerank"

    def test_sources_can_be_disabled(self, settings):
        client = build_client(settings, FakeRAG(settings, hits=[make_hit("k1")]))

        events = parse_sse(
            client.post("/chat", json={"message": "问题", "include_sources": False}).text
        )

        assert "sources" not in [n for n, _ in events]
        assert "done" in [n for n, _ in events]

    def test_empty_retrieval_skips_model(self, settings):
        """检索为空直接返回兜底文案，不消耗模型算力，也避免硬编答案。"""
        client = build_client(settings, FakeRAG(settings, hits=[]))

        events = parse_sse(client.post("/chat", json={"message": "问题"}).text)

        assert [n for n, _ in events] == ["sources", "token", "done"]
        assert "没有检索到" in events[1][1]["text"]

    def test_empty_message_is_rejected(self, settings):
        client = build_client(settings, FakeRAG(settings))

        assert client.post("/chat", json={"message": ""}).status_code == 422


# ---------------------------------------------------------------------------
# 非流式与纯检索
# ---------------------------------------------------------------------------
class TestChatSync:
    def test_returns_answer_and_sources(self, settings):
        rag = FakeRAG(settings, hits=[make_hit("k1")], deltas=["完整", "答案"])
        client = build_client(settings, rag)

        body = client.post("/chat/sync", json={"message": "问题"}).json()

        assert body["answer"] == "完整答案"
        assert len(body["sources"]) == 1

    def test_no_context_returns_fallback(self, settings):
        client = build_client(settings, FakeRAG(settings, hits=[]))

        body = client.post("/chat/sync", json={"message": "问题"}).json()

        assert "没有检索到" in body["answer"]


class TestSearch:
    def test_returns_ranking_detail(self, settings):
        hit = make_hit("k1")
        hit.dense_rank, hit.sparse_rank = 1, None
        client = build_client(settings, FakeRAG(settings, hits=[hit]))

        body = client.post("/search", json={"query": "查询"}).json()

        assert body["count"] == 1
        assert body["results"][0]["dense_rank"] == 1
        assert body["results"][0]["is_hybrid_hit"] is False


# ---------------------------------------------------------------------------
# 概况与入库
# ---------------------------------------------------------------------------
class TestStats:
    def test_reports_collection_and_config(self, settings):
        client = build_client(settings, FakeRAG(settings))

        body = client.get("/stats").json()

        assert body["chunks"] == 42
        assert body["server_version"] == "v2.5.4"
        assert body["embedding_dim"] == 1024


class TestIngest:
    def test_missing_path_returns_404(self, settings, tmp_path):
        client = build_client(settings, FakeRAG(settings))

        response = client.post("/ingest", json={"path": str(tmp_path / "不存在")})

        assert response.status_code == 404
