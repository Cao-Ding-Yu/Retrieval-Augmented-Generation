"""Milvus 存储层：建表、写入、两路检索、RRF 融合。

这是唯一直接接触 Milvus 的文件。上层（``rag.py``）只需要三个方法：
``dense_search`` / ``sparse_search`` / ``insert``，不需要知道 Milvus 的任何细节。

## 为什么用 MilvusClient

pymilvus 3.x 已弃用 ORM 风格的 ``Collection`` / ``utility`` API（3.1 移除）。
本文件全部使用 ``MilvusClient``。

## 为什么要求 Milvus 2.5+

词法分支依赖**服务端内置 BM25**：写入时服务端自动把 ``text`` 变成稀疏向量，
查询时也能直接吃原始文本。所以应用层完全不用管稀疏编码，也不需要第二个模型。
版本不足时会明确报错，**不做静默降级** —— 降级会让词法分支静默失效，
那种"能跑但效果差"的状态极难排查。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

from pymilvus import (
    AnnSearchRequest,
    DataType,
    Function,
    FunctionType,
    MilvusClient,
    RRFRanker,
)

from rag_app.config import Settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 字段名
# ---------------------------------------------------------------------------
FIELD_ID = "id"
FIELD_TEXT = "text"
FIELD_DOC_ID = "doc_id"
FIELD_SOURCE = "source"
FIELD_CHUNK_INDEX = "chunk_index"
FIELD_DENSE = "dense"
FIELD_SPARSE = "sparse"

# 检索时要取回的标量字段
OUTPUT_FIELDS = [FIELD_TEXT, FIELD_DOC_ID, FIELD_SOURCE, FIELD_CHUNK_INDEX]

# 服务端 BM25 自 Milvus 2.5 起提供
MIN_SERVER_VERSION = (2, 5)

# HNSW 检索期候选队列长度（越大召回越高、越慢）
HNSW_SEARCH_EF = 128


@dataclass(slots=True)
class Hit:
    """一条检索命中的分块。

    与 Milvus 文档的对应关系：同一个分块在稠密分支和 BM25 分支各有一条命中，
    融合时按 ``id`` 去重。
    """

    id: str
    text: str
    doc_id: str
    source: str
    chunk_index: int
    score: float
    """当前生效的分数。融合后是 RRF 分数；重排后会被替换成重排分数。"""

    dense_rank: int | None = None
    """在稠密分支中的名次（0 起）。未命中该分支为 None。"""

    sparse_rank: int | None = None
    """在 BM25 分支中的名次（0 起）。未命中该分支为 None。"""

    score_kind: str = "rrf"
    """``score`` 是哪种分数：``"rrf"``（融合分）或 ``"rerank"``（重排分）。

    前端靠它决定显示"RRF x.xxxxx"还是"重排 x.xxx"，以及保留几位小数 ——
    RRF 分很小（1/61 ≈ 0.016），重排分在 0~1 之间，两者的有效位数差别很大。
    """

    @property
    def is_hybrid_hit(self) -> bool:
        """两个分支同时命中 —— 既语义相关又命中关键词，是最可信的结果。"""
        return self.dense_rank is not None and self.sparse_rank is not None

    def to_dict(self) -> dict:
        """给前端的精简表示（不含全文，避免响应体过大）。"""
        return {
            "doc_id": self.doc_id,
            "source": self.source,
            "chunk_index": self.chunk_index,
            "score": round(self.score, 6),
            "score_kind": self.score_kind,
            "dense_rank": self.dense_rank,
            "sparse_rank": self.sparse_rank,
            "is_hybrid_hit": self.is_hybrid_hit,
            # snippet 用于折叠态预览，text 用于展开态看全文。
            # 只发 snippet 的话"展开"就成了空操作 —— 前端两个状态会显示同一段截断文字。
            "snippet": self.text[:280] + ("…" if len(self.text) > 280 else ""),
            "text": self.text,
        }


def fuse(dense: list[Hit], sparse: list[Hit], k: int = 60) -> list[Hit]:
    """RRF（倒数排名融合）：把两个分支的命中合并成一个排序。

    公式：``RRF(d) = Σ 1 / (k + rank)``，rank 从 1 开始。

    ## 为什么用 RRF 而不是分数加权

    稠密检索输出余弦相似度（约 -1~1），BM25 输出无上界的词频加权分。
    两者量纲不同，直接加权要先归一化，而归一化对分数分布尾部极敏感、不稳定。

    RRF **只用名次**，完全避开分数校准。且数学上恒有 ``2/(k+2) > 1/(k+1)``
    （任意 k>0），也就是"两路都排第二"稳定优于"单路排第一" ——
    这就是"多路证据一致比单路极端高分更可信"的形式化表达。

    k 越大越削弱头部差距；实践常用 60。

    Args:
        dense: 稠密分支的命中（按相关性降序）。
        sparse: BM25 分支的命中（按相关性降序）。
        k: RRF 平滑常数。

    Returns:
        按 RRF 分数降序的命中列表，每条带上 `dense_rank` / `sparse_rank`。
        同一分支内重复的 id 只按首次（最优）名次计一次分。

    Raises:
        ValueError: k 不是正数（否则会出现除零）。
    """
    if k <= 0:
        raise ValueError(f"RRF 常数 k 必须为正数，当前为 {k}")

    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}
    by_id: dict[str, Hit] = {}

    for branch, hits in (("dense", dense), ("sparse", sparse)):
        seen: set[str] = set()
        for position, hit in enumerate(hits):
            by_id.setdefault(hit.id, hit)
            if hit.id in seen:  # 同分支内重复，只算最靠前那次
                continue
            seen.add(hit.id)
            scores[hit.id] = scores.get(hit.id, 0.0) + 1.0 / (k + position + 1)
            ranks.setdefault(hit.id, {})[branch] = position

    fused = [
        replace(
            by_id[doc_id],
            score=score,
            dense_rank=ranks[doc_id].get("dense"),
            sparse_rank=ranks[doc_id].get("sparse"),
        )
        for doc_id, score in scores.items()
    ]
    # 分数相同时按"最佳名次"再按 id 排，保证结果可复现
    fused.sort(key=lambda h: (-h.score, min(ranks[h.id].values()), h.id))
    return fused


class MilvusStore:
    """Milvus 的读写封装。构造时只记配置，``connect()`` 才真正连。"""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: MilvusClient | None = None
        self.server_version = ""

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def connect(self) -> None:
        """连接并校验版本，然后确保集合就绪。

        Raises:
            RuntimeError: Milvus 版本低于 2.5（无服务端 BM25）。
        """
        logger.info("连接 Milvus：%s", self._settings.milvus_uri)
        self._client = MilvusClient(
            uri=self._settings.milvus_uri,
            token=self._settings.milvus_token_or_none,
        )
        self.server_version = self._client.get_server_version()

        major, minor = _parse_version(self.server_version)
        if (major, minor) < MIN_SERVER_VERSION:
            need = ".".join(map(str, MIN_SERVER_VERSION))
            raise RuntimeError(
                f"Milvus 版本过低：检测到 {self.server_version}，要求 >= {need}。\n"
                "原因：混合检索的词法分支依赖服务端内置 BM25，该能力自 Milvus 2.5 起提供。\n"
                "处理：升级服务端，或用 Docker 起一个 2.5+ 实例。"
            )

        logger.info("Milvus 版本=%s（满足服务端 BM25 要求）", self.server_version)
        self._ensure_collection()

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:  # pragma: no cover - 关闭期异常不该影响退出
                logger.warning("关闭 Milvus 连接失败", exc_info=True)
            finally:
                self._client = None

    @property
    def client(self) -> MilvusClient:
        if self._client is None:
            raise RuntimeError("MilvusStore 尚未连接，请先调用 connect()")
        return self._client

    # ------------------------------------------------------------------
    # 建表
    # ------------------------------------------------------------------
    def _ensure_collection(self) -> None:
        """集合不存在就建（含索引与 BM25 Function），存在就校验维度。"""
        name = self._settings.milvus_collection
        client = self.client

        if client.has_collection(name):
            self._check_dimension(name)
            client.load_collection(name)  # Milvus 重启后需重新 load 才能检索
            logger.info("复用已存在的集合：%s", name)
            return

        logger.info("集合 %s 不存在，开始创建", name)
        dim = self._settings.embedding_dim

        # ---- Schema ----
        schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)
        schema.add_field(FIELD_ID, DataType.INT64, is_primary=True)
        text_kwargs: dict = {"max_length": 8192, "enable_analyzer": True}
        if self._settings.bm25_analyzer_type:
            # 中文语料必须显式指定中文分析器，否则分词切不出词、BM25 形同虚设。
            # 只在建集合时生效，改它必须重建集合并重新入库。
            text_kwargs["analyzer_params"] = {"type": self._settings.bm25_analyzer_type}
        schema.add_field(FIELD_TEXT, DataType.VARCHAR, **text_kwargs)
        schema.add_field(FIELD_DOC_ID, DataType.VARCHAR, max_length=64)
        schema.add_field(FIELD_SOURCE, DataType.VARCHAR, max_length=512)
        schema.add_field(FIELD_CHUNK_INDEX, DataType.INT64)
        schema.add_field(FIELD_DENSE, DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field(FIELD_SPARSE, DataType.SPARSE_FLOAT_VECTOR)

        # 注册内置函数：写入 text 时服务端自动算出 sparse，应用层不用管稀疏向量
        schema.add_function(
            Function(
                name="bm25_text_to_sparse",
                function_type=FunctionType.BM25,
                input_field_names=[FIELD_TEXT],
                output_field_names=[FIELD_SPARSE],
            )
        )

        # ---- 索引 ----
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name=FIELD_DENSE,
            index_type="HNSW",
            metric_type="COSINE",
            M=16,
            efConstruction=200,
        )
        index_params.add_index(
            field_name=FIELD_SPARSE,
            # 稀疏字段的度量必须写 BM25（不是 IP），否则分词与打分口径不一致
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
            drop_ratio_build=0.0,
        )

        client.create_collection(name, schema=schema, index_params=index_params)
        client.load_collection(name)
        logger.info("集合 %s 创建完成并已加载", name)

    def _check_dimension(self, name: str) -> None:
        """已存在的集合必须与配置的向量维度一致，否则直接报错。

        不做"自动删库重建" —— 静默丢数据比启动失败糟糕得多。
        """
        info = self.client.describe_collection(name)
        for f in info.get("fields", []):
            if f.get("name") != FIELD_DENSE:
                continue
            params = f.get("params") or {}
            existing = int(params.get("dim", f.get("dim", -1)))
            if existing != self._settings.embedding_dim:
                raise RuntimeError(
                    f"集合 {name} 的向量维度是 {existing}，但配置 EMBEDDING_DIM="
                    f"{self._settings.embedding_dim}（模型 {self._settings.embedding_model}）。\n"
                    "请改 MILVUS_COLLECTION 换一个新集合名，或删除旧集合后重新入库。"
                )
            return
        raise RuntimeError(f"集合 {name} 里没有 {FIELD_DENSE} 字段，可能不是本项目创建的")

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------
    def insert(self, rows: list[dict]) -> int:
        """写入分块。每行需含 text/doc_id/source/chunk_index/dense。

        **不要**传 sparse —— 它由服务端 BM25 Function 自动生成。
        这里做一次剔除，免得调用方踩坑。

        Returns:
            写入条数。
        """
        if not rows:
            raise ValueError("插入数据为空")

        payload = [dict(row) for row in rows]
        for row in payload:
            row.pop(FIELD_SPARSE, None)

        result = self.client.insert(self._settings.milvus_collection, data=payload)
        inserted = int(result.get("insert_count", len(payload)))
        # flush 让数据立刻可检索。Milvus 是最终一致的，不 flush 就要等一段时间；
        # 对"入库后立刻提问"的体验不可接受。代价是写入吞吐下降。
        self.client.flush(self._settings.milvus_collection)
        logger.info("写入 %d 条分块", inserted)
        return inserted

    def delete_doc(self, doc_id: str) -> int:
        """按 doc_id 删除该文档的所有分块，返回删除条数。

        ``doc_id`` 由文件路径哈希得到（见 ``ingest._doc_id``），不随内容变化，
        所以"重新入库同一个文件"能正确替换掉它之前的全部分块。

        先查主键再按主键删，而不是按表达式删 —— 这样能准确知道删了多少条。

        Args:
            doc_id: 文档 ID。
        """
        name = self._settings.milvus_collection
        rows = self.client.query(
            name, filter=f'{FIELD_DOC_ID} == "{doc_id}"', output_fields=[FIELD_ID]
        )
        if not rows:
            return 0
        ids = [row[FIELD_ID] for row in rows]
        result = self.client.delete(name, ids=ids)
        self.client.flush(name)
        return int(result.get("deleted_count", len(ids)))

    def count(self) -> int:
        """集合中的分块总数（健康检查 / 入库后校验用）。"""
        try:
            return int(self.client.get_collection_stats(
                self._settings.milvus_collection).get("row_count", -1))
        except Exception:  # pragma: no cover
            return -1

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------
    def dense_search(self, vector: list[float], limit: int) -> list[Hit]:
        """稠密语义检索（COSINE）。"""
        results = self.client.search(
            collection_name=self._settings.milvus_collection,
            data=[vector],
            anns_field=FIELD_DENSE,
            search_params={
                "metric_type": "COSINE",
                "params": {"ef": max(HNSW_SEARCH_EF, limit)},
            },
            limit=limit,
            output_fields=OUTPUT_FIELDS,
        )
        return _to_hits(results)

    def sparse_search(self, query_text: str, limit: int) -> list[Hit]:
        """BM25 词法检索。

        ``data`` 传的是**原始文本**而非向量：Milvus 会用字段的 analyzer 分词
        并计算 BM25 权重。注意不能用 ``TEXT_MATCH`` —— 那只做布尔匹配、
        不产生相关性分数，给不出 RRF 需要的排名。
        """
        results = self.client.search(
            collection_name=self._settings.milvus_collection,
            data=[query_text],
            anns_field=FIELD_SPARSE,
            search_params={"metric_type": "BM25", "params": {}},
            limit=limit,
            output_fields=OUTPUT_FIELDS,
        )
        return _to_hits(results)

    def hybrid_search(self, vector: list[float], query_text: str, limit: int,
                      rrf_k: int) -> list[Hit]:
        """一次调用交给 Milvus 做混合检索（服务端 RRF，单次往返）。

        比"两次检索 + 本地 fuse"少一次网络往返，但拿不到各分支名次。
        注意 MilvusClient 的参数名是 ``ranker``（不是 ORM 的 ``rerank``），
        且 BM25 分支同样用 ``AnnSearchRequest`` —— pymilvus 里**没有**
        ``BM25SearchRequest`` 这个类。
        """
        results = self.client.hybrid_search(
            collection_name=self._settings.milvus_collection,
            reqs=[
                AnnSearchRequest(
                    data=[vector], anns_field=FIELD_DENSE,
                    param={"metric_type": "COSINE",
                           "params": {"ef": max(HNSW_SEARCH_EF, limit)}},
                    limit=limit,
                ),
                AnnSearchRequest(
                    data=[query_text], anns_field=FIELD_SPARSE,
                    param={"metric_type": "BM25"}, limit=limit,
                ),
            ],
            ranker=RRFRanker(k=rrf_k),
            limit=limit,
            output_fields=OUTPUT_FIELDS,
        )
        return _to_hits(results)


def _parse_version(raw: str) -> tuple[int, int]:
    """把 'v2.5.4' / '2.5.4-lite' 解析成 (2, 5)，失败返回 (0, 0)。"""
    cleaned = (raw or "").strip().lstrip("vV").split("-")[0]
    parts = cleaned.split(".")
    try:
        return int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
    except (ValueError, IndexError):
        return 0, 0


def _to_hits(results) -> list[Hit]:
    """把 MilvusClient 的嵌套字典结果转成 Hit。

    返回结构形如 ``[[{"id":1,"distance":0.8,"entity":{...}}, ...]]``：
    外层是每个查询（每次只传一个），内层按相关性降序。
    """
    if not results:
        return []
    hits: list[Hit] = []
    for row in results[0]:
        entity = row.get("entity") or {}
        hits.append(
            Hit(
                id=str(row.get("id")),
                text=str(entity.get(FIELD_TEXT, "") or ""),
                doc_id=str(entity.get(FIELD_DOC_ID, "") or ""),
                source=str(entity.get(FIELD_SOURCE, "") or ""),
                chunk_index=int(entity.get(FIELD_CHUNK_INDEX, -1) or -1),
                score=float(row.get("distance", 0.0)),
            )
        )
    return hits
