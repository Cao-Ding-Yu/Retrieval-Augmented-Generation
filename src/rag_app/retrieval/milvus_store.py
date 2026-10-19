"""Milvus 存储封装（MilvusClient 版）：集合生命周期、写入、以及两条检索路径。

## 为什么用 MilvusClient 而不是 ORM 风格 API

pymilvus 3.x 已把 ``Collection`` / ``utility`` / ``connections`` 这套 ORM 风格 API
标记为弃用，并会在 **3.1 中移除**，同时建议迁移到 ``MilvusClient``。
本模块已按此迁移，因此运行时不再出现 ``PyMilvusDeprecationWarning``。

迁移中的主要接口差异（都已在本文件中体现）：

============================  ==========================================
ORM 风格                        MilvusClient
============================  ==========================================
``connections.connect``         ``MilvusClient(uri=..., token=...)``
``utility.has_collection``      ``client.has_collection``
``utility.get_server_version``  ``client.get_server_version()``
``Collection(...)`` 构造         ``client.create_collection(schema=...)``
``collection.create_index``     ``client.create_index`` / ``prepare_index_params``
``collection.load``             ``client.load_collection``
``collection.insert``           ``client.insert``（返回 dict，非 MutationResult）
``collection.flush``            ``client.flush``
``collection.query(expr=)``     ``client.query(filter=)``
``collection.delete(expr=)``    ``client.delete(ids=...)``
``collection.search``           ``client.search(search_params=)``
``collection.hybrid_search``    ``client.hybrid_search(ranker=)``
``collection.num_entities``     ``client.get_collection_stats()["row_count"]``
结果对象 ``Hit``                结果 dict：``{"id":.., "distance":.., "entity":{..}}``
============================  ==========================================

## 两条检索路径

**路径 A：客户端融合（`search_and_fuse_branches`）**
分别对稠密字段和稀疏字段各做一次检索，再用本模块的
:func:`reciprocal_rank_fusion` 自行融合。优点：
  * 能拿到**每个候选在两条分支上的原始名次**，这些信息会一路传到前端，
    用于解释排序依据、并标出"双分支同时命中"的高质量结果；
  * 融合算法是纯函数，可单元测试、可审计（见 tests/test_rrf.py）。

**路径 B：服务端融合（`hybrid_search`）**
一次 ``hybrid_search`` 调用，把两个请求和 ``RRFRanker`` 一起交给 Milvus，
由服务端完成融合。优点：往返次数少（1 次 vs 2 次），网络开销更低。
代价：拿不到分支名次，只能得到融合后的分数。

两条路径都保留，是刻意的工程取舍而非冗余 —— 生产环境若 QPS 压力大可以切到
路径 B；调试排序问题或需要引用解释时用路径 A。两者算出的排序是一致的，
因为 Milvus 的 ``RRFRanker`` 用的就是同一个 RRF 公式。
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from pymilvus import AnnSearchRequest, MilvusClient, RRFRanker

from rag_app.config import Settings
from rag_app.schema import (
    FIELD_CHUNK_INDEX,
    FIELD_DENSE,
    FIELD_DOC_ID,
    FIELD_ID,
    FIELD_SOURCE,
    FIELD_SPARSE,
    FIELD_TEXT,
    OUTPUT_FIELDS,
    MIN_SERVER_VERSION,
    build_schema,
    dense_index_params,
    sparse_index_params,
    supports_server_bm25,
)

logger = logging.getLogger(__name__)

# HNSW 检索时的动态候选队列长度。越大召回越高、延迟越高。
# 128 在本地十万级数据下是延迟与召回的平衡点。
HNSW_SEARCH_EF = 128

# ---------------------------------------------------------------------------
# RRF（倒数排名融合）
# ---------------------------------------------------------------------------
# ## 公式
#
#     RRF(d) = Σ_r  1 / (k + rank_r(d))
#
# 其中 rank 从 1 开始，r 遍历各检索分支（本项目是稠密 + BM25 两支）。
#
# ## 为什么用 RRF 而不是分数加权
#
# 稠密检索输出的是余弦相似度（约 -1~1），BM25 输出的是无上界的词频加权分数。
# 两者量纲不同、分布不同，直接加权需要先归一化，而归一化本身就是不稳定的
# 经验操作（对分数分布的尾部极其敏感）。
#
# RRF 只使用**名次**，完全避开分数校准问题：某个文档只要在任一分支排得靠前
# 就会获得较高贡献；两个分支都命中的文档会累加，因而自然上浮。
# 数学上恒有 2/(k+2) > 1/(k+1)（对任意 k>0 成立），也就是
# "两路都排第二"稳定优于"单路排第一" —— 这正是"多路证据一致性比单路极端
# 高分更可信"的形式化表达。
#
# 常数 k 起到削弱头部差距的作用：k 越大，第 1 名与第 5 名的差距越小，
# 融合结果越平均；k 越小则越倚重单分支的头部结果。实践常用 k=60。
#
# ## 与 Milvus 内置 RRFRanker 的关系
#
# 服务端融合路径（`hybrid_search`）用的是 Milvus 内置的 `RRFRanker`，
# 它计算的就是同一个公式。本模块自己实现一份，是为了让客户端融合路径
# 能拿到各分支的原始名次，并让这段逻辑可以被单元测试覆盖 ——
# 而不是"相信数据库算对了"。
# ---------------------------------------------------------------------------

# 业界默认值。改动它等于改变"头部集中度"，属于需要评测支撑的调参。
DEFAULT_RRF_K = 60


@dataclass(slots=True)
class FusedHit:
    """融合后的单个命中项。"""

    key: str
    """文档/分块标识（本项目中用 Milvus 主键的字符串形式）。"""

    score: float
    """RRF 融合分数，越大越相关。"""

    ranks: dict[str, int] = field(default_factory=dict)
    """各分支中的名次（0 起，越小越靠前），键为分支名，例如 {"dense": 0, "sparse": 3}。"""

    def rank_in(self, branch: str) -> int | None:
        """返回该命中在指定分支的名次；未命中该分支时返回 None。"""
        return self.ranks.get(branch)


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[str]],
    k: int = DEFAULT_RRF_K,
) -> list[FusedHit]:
    """对多个有序排名列表做 RRF 融合。

    Args:
        rankings: 分支名 -> 该分支的**有序**标识列表（索引 0 为最相关）。
            例如 ``{"dense": ["a", "b"], "sparse": ["b", "c"]}``。
        k: RRF 平滑常数，必须为正数。越大越弱化头部差距。

    Returns:
        按 RRF 分数降序排列的 :class:`FusedHit` 列表。
        相同分数时按"最佳名次"升序、再按标识字典序排序 ——
        加入确定性的兜底排序是为了让结果可复现，避免测试与线上表现漂移。

    Raises:
        ValueError: k 不是正数。

    Example:
        >>> hits = reciprocal_rank_fusion({"dense": ["a", "b"], "sparse": ["b", "a"]})
        >>> sorted(h.key for h in hits)
        ['a', 'b']
        >>> hits[0].ranks
        {'dense': 0, 'sparse': 1}
    """
    if k <= 0:
        raise ValueError(f"RRF 常数 k 必须为正数，当前为 {k}")

    # 累加每个标识的 RRF 分数，同时记录其在各分支的名次
    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}

    for branch, ordered_keys in rankings.items():
        seen_in_branch: set[str] = set()

        for position, key in enumerate(ordered_keys):
            # 同一分支内出现重复标识时只计第一次（前一个名次更优），
            # 否则重复项会因为被累加多次而虚假上浮
            if key in seen_in_branch:
                continue
            seen_in_branch.add(key)

            # rank 从 1 起算，因此用 position + 1
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + position + 1)
            ranks.setdefault(key, {})[branch] = position

    hits = [
        FusedHit(key=key, score=score, ranks=ranks.get(key, {}))
        for key, score in scores.items()
    ]

    # 排序键解释：
    #   1. -score    : 分数高的在前
    #   2. best_rank : 分数相同时，两个分支里更靠前的名次优先
    #   3. key       : 仍相同时按字典序，保证结果确定
    hits.sort(
        key=lambda h: (
            -h.score,
            min(h.ranks.values()) if h.ranks else float("inf"),
            h.key,
        )
    )
    return hits


def fuse_two_branch(
    dense_keys: Sequence[str],
    sparse_keys: Sequence[str],
    k: int = DEFAULT_RRF_K,
) -> list[FusedHit]:
    """双分支融合的便捷封装（稠密 + 稀疏）。

    只是 :func:`reciprocal_rank_fusion` 的语法糖，让调用处更贴近业务语义。
    """
    return reciprocal_rank_fusion({"dense": dense_keys, "sparse": sparse_keys}, k=k)


@dataclass(slots=True)
class RawHit:
    """Milvus 返回的原始命中项（尚未附加分支名次）。"""

    key: str
    """Milvus 主键的字符串形式，用作后续融合与去重的标识。"""

    text: str
    doc_id: str
    source: str
    chunk_index: int
    score: float
    """该分支给出的原始分数（稠密为余弦相似度，稀疏为 BM25 分）。"""


class MilvusStore:
    """Milvus 集合的读写封装（基于 MilvusClient）。

    Args:
        settings: 全局配置。
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: MilvusClient | None = None
        self._server_version: str = ""

    # ------------------------------------------------------------------
    # 连接与生命周期
    # ------------------------------------------------------------------
    def connect(self) -> None:
        """建立 Milvus 连接并校验服务端版本。

        Raises:
            RuntimeError: 服务端版本低于 :data:`~rag_app.schema.MIN_SERVER_VERSION`。
                这里刻意**不做静默降级** —— 版本不足时混合检索的词法分支会
                整个失效，那种"能跑但效果差"的状态极难排查。宁可启动就报错。
        """
        logger.info("连接 Milvus：%s", self._settings.milvus_uri)

        self._client = MilvusClient(
            uri=self._settings.milvus_uri,
            token=self._settings.milvus_token_or_none,
        )

        self._server_version = self._client.get_server_version()

        if not supports_server_bm25(self._server_version):
            required = ".".join(str(p) for p in MIN_SERVER_VERSION)
            raise RuntimeError(
                f"Milvus 版本过低：检测到 {self._server_version}，本项目要求 >= {required}。\n"
                "原因：混合检索的词法分支依赖服务端内置 BM25 Function，该能力自 Milvus 2.5 起提供。\n"
                "处理方式：升级 Milvus 服务端（可用 docker/milvus-compose.yml 起一个 2.5+ 实例）。"
            )

        logger.info("Milvus 版本=%s（满足服务端 BM25 要求）", self._server_version)

    def close(self) -> None:
        """断开连接。在应用关闭时调用。"""
        if self._client is None:
            return
        try:
            self._client.close()
            logger.info("已断开 Milvus 连接")
        except Exception:  # pragma: no cover - 关闭阶段的异常不值得打断退出流程
            logger.warning("断开 Milvus 连接时出现异常", exc_info=True)
        finally:
            self._client = None

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------
    @property
    def client(self) -> MilvusClient:
        """底层 MilvusClient。未连接时抛错，避免上层在未初始化时误用。"""
        if self._client is None:
            raise RuntimeError("MilvusStore 尚未连接，请先调用 connect()")
        return self._client

    @property
    def server_version(self) -> str:
        """服务端版本字符串，用于健康检查展示。"""
        return self._server_version

    @property
    def collection_name(self) -> str:
        """当前操作的集合名。"""
        return self._settings.milvus_collection

    # ------------------------------------------------------------------
    # 集合管理
    # ------------------------------------------------------------------
    def ensure_collection(self) -> None:
        """确保集合存在且结构正确。

        行为：
          * 不存在 -> 建集合（含索引与 BM25 Function）、load；
          * 已存在 -> 校验稠密向量维度是否一致，不一致直接抛错。

        不做"维度不一致就自动删库重建"这种危险动作 —— 静默丢数据比启动失败糟糕得多。

        Raises:
            RuntimeError: 已存在集合的维度与配置不符。
        """
        name = self.collection_name
        client = self.client

        if client.has_collection(name):
            self._validate_dimension(name)
            # 已加载过则无需重复 load；Milvus 重启后需要重新 load 才能检索
            client.load_collection(name)
            logger.info("复用已存在的集合：%s", name)
            return

        logger.info("集合 %s 不存在，开始创建", name)

        schema = build_schema(
            embedding_dim=self._settings.embedding_dim,
            # 中文语料需要显式指定中文分析器，否则 BM25 词法分支形同虚设。
            # 该参数仅在**新建集合**时生效，因此改配置后必须重建集合并重新入库。
            analyzer_type=self._settings.bm25_analyzer_type,
        )

        # 索引参数用 prepare_index_params() 构造。
        # 注意 add_index 会把额外 kwargs 摊平进索引配置，所以 dense/sparse_index_params()
        # 直接返回扁平的字典（不套 params={...}）。
        index_params = client.prepare_index_params()
        index_params.add_index(field_name=FIELD_DENSE, **dense_index_params())
        index_params.add_index(field_name=FIELD_SPARSE, **sparse_index_params())

        client.create_collection(name, schema=schema, index_params=index_params)
        # create_collection 不会自动 load，必须显式调用才能检索
        client.load_collection(name)
        logger.info("集合 %s 创建完成并已加载", name)

    def _validate_dimension(self, collection_name: str) -> None:
        """校验已存在集合的稠密向量维度是否与配置一致。

        Raises:
            RuntimeError: 维度不匹配，并给出可操作的修复建议。
        """
        info = self.client.describe_collection(collection_name)

        for field in info.get("fields", []):
            if field.get("name") != FIELD_DENSE:
                continue
            # 维度信息在字段的 params 里；不同版本也可能直接放在字段顶层
            params = field.get("params") or {}
            existing_dim = int(params.get("dim", field.get("dim", -1)))
            if existing_dim != self._settings.embedding_dim:
                raise RuntimeError(
                    f"集合 {collection_name} 的稠密向量维度为 {existing_dim}，"
                    f"但配置 EMBEDDING_DIM={self._settings.embedding_dim}"
                    f"（模型 {self._settings.embedding_model}）。\n"
                    "请修改 MILVUS_COLLECTION 换一个新集合名，或删除旧集合后重新入库。"
                )
            return

        # 走到这里说明没找到稠密字段，说明集合结构不是本项目创建的
        raise RuntimeError(
            f"集合 {collection_name} 中不存在字段 {FIELD_DENSE}，"
            "它可能不是由本项目创建的。请更换 MILVUS_COLLECTION。"
        )

    def drop_collection(self) -> None:
        """删除集合。仅用于开发调试与"想彻底重建"的场景，会丢数据。"""
        name = self.collection_name
        if self.client.has_collection(name):
            self.client.drop_collection(name)
            logger.warning("集合 %s 已被删除", name)

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------
    def insert(self, rows: list[dict[str, Any]]) -> int:
        """批量写入分块。

        入参约定：每行必须包含 ``text`` / ``doc_id`` / ``source`` /
        ``chunk_index`` / ``dense``。

        **不要**传 ``sparse`` —— 它由 Schema 上注册的 BM25 Function 在服务端
        自动生成，手工塞入会触发"函数输出字段不可重复赋值"类错误。
        这里做一次防御性剔除，免得调用方踩坑。

        Args:
            rows: 行字典列表。

        Returns:
            实际写入条数。

        Raises:
            ValueError: rows 为空。
        """
        if not rows:
            raise ValueError("插入数据为空")

        name = self.collection_name
        payload = [dict(row) for row in rows]

        # 防御性剔除 sparse：即使调用方传了也不该报错，静默忽略更友好
        for row in payload:
            row.pop(FIELD_SPARSE, None)

        # MilvusClient.insert 返回 {"insert_count": N, "ids": [...]}
        result = self.client.insert(name, data=payload)
        inserted = int(result.get("insert_count", len(payload)))

        # flush 让数据立刻可检索。
        # Milvus 本身是最终一致的，不 flush 的话刚写入的数据要等一段时间
        # 才能被搜到；对演示与"入库后立刻提问"的交互体验来说，这个等待不可接受。
        # 代价是批量入库吞吐会下降 —— 若追求极致吞吐，可改为定期 flush。
        self.client.flush(name)

        logger.info("写入 %d 条分块到 %s", inserted, name)
        return inserted

    def delete_by_doc_id(self, doc_id: str) -> int:
        """按文档 ID 删除其全部分块，返回删除条数。

        采用"查询 -> 按主键删除"两步，而不是直接按表达式删除：
        这样能准确知道删了多少条，便于日志与幂等性验证。
        主键删除也更稳妥 —— Milvus 对非主键字段的表达式删除在部分版本上
        依赖字段先建索引。

        Args:
            doc_id: 文档 ID。

        Returns:
            实际删除条数。
        """
        client = self.client
        name = self.collection_name

        # 先查出该文档的所有主键（MilvusClient 用 filter 而非 expr）
        existing = client.query(
            name,
            filter=f'{FIELD_DOC_ID} == "{doc_id}"',
            output_fields=[FIELD_ID],
        )
        if not existing:
            return 0

        primary_keys = [row[FIELD_ID] for row in existing]
        # 按主键删除，返回 {"deleted_count": N}
        result = client.delete(name, ids=primary_keys)
        client.flush(name)

        deleted = int(result.get("deleted_count", len(primary_keys)))
        logger.debug("已删除文档 %s 的 %d 条分块", doc_id[:8], deleted)
        return deleted

    # ------------------------------------------------------------------
    # 检索：路径 A（两条独立检索，供客户端融合）
    # ------------------------------------------------------------------
    def dense_search(self, vector: list[float], limit: int) -> list[RawHit]:
        """稠密向量检索（COSINE）。

        Args:
            vector: 查询向量。
            limit: 返回条数。

        Returns:
            按相似度降序的命中列表。
        """
        results = self.client.search(
            collection_name=self.collection_name,
            data=[vector],
            anns_field=FIELD_DENSE,
            # MilvusClient 用 search_params（对应 ORM 的 param）
            search_params={
                "metric_type": "COSINE",
                # ef 是检索期的动态候选队列长度，需 >= limit
                "params": {"ef": max(HNSW_SEARCH_EF, limit)},
            },
            limit=limit,
            output_fields=list(OUTPUT_FIELDS),
        )
        return self._flatten_results(results)

    def sparse_search(self, query_text: str, limit: int) -> list[RawHit]:
        """BM25 词法检索（单分支）。

        做法：以 ``metric_type="BM25"`` 直接检索稀疏字段，``data`` 传**原始文本**
        而非向量 —— 由服务端用字段的 analyzer 分词并计算 BM25 权重。

        之所以能传文本：pymilvus 在 ``prepare._prepare_placeholder_str`` 里对
        ``isinstance(data[0], str)`` 有专门分支，会把字符串编码成 ``VARCHAR``
        placeholder 发给服务端。这正是全文检索的查询通道。

        ## 为什么不用 TEXT_MATCH

        ``TEXT_MATCH`` 需要字段声明 ``enable_match=True``，且它只做**布尔匹配**
        （命中/未命中），**不产生 BM25 相关性打分**。RRF 的输入必须是排名，
        所以 TEXT_MATCH 从根本上无法作为词法分支使用 —— 这也是本项目
        不在 Schema 上开启 ``enable_match`` 的原因。

        Args:
            query_text: 原始查询文本。
            limit: 返回条数。

        Returns:
            按词法相关性降序的命中列表。
        """
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_text],
            anns_field=FIELD_SPARSE,
            search_params={"metric_type": "BM25", "params": {}},
            limit=limit,
            output_fields=list(OUTPUT_FIELDS),
        )
        return self._flatten_results(results)

    def search_and_fuse_branches(
        self,
        vector: list[float],
        query_text: str,
        dense_limit: int,
        sparse_limit: int,
    ) -> tuple[list[RawHit], list[RawHit]]:
        """分别执行两个分支的检索，返回原始结果供上层自行融合。

        这条路径不在此处融合，是为了让上层能同时拿到"分支名次"与"融合分数"，
        从而把排序依据完整地暴露给调用方（见 HybridRetriever）。

        Returns:
            (稠密命中列表, 稀疏命中列表)，两者顺序即为各自名次顺序。
        """
        dense_hits = self.dense_search(vector, dense_limit)
        sparse_hits = self.sparse_search(query_text, sparse_limit)
        return dense_hits, sparse_hits

    # ------------------------------------------------------------------
    # 检索：路径 B（服务端融合）
    # ------------------------------------------------------------------
    def hybrid_search(
        self,
        vector: list[float],
        query_text: str,
        limit: int,
        rrf_k: int,
    ) -> list[RawHit]:
        """服务端 RRF 混合检索（一次往返）。

        Args:
            vector: 查询向量。
            query_text: 原始查询文本。
            limit: 返回条数。
            rrf_k: RRF 常数，传给服务端 RRFRanker。

        Returns:
            融合后的命中列表。注意此路径**不包含分支名次信息**。

        Note:
            两个分支都用 ``AnnSearchRequest`` 构造，**不存在** ``BM25SearchRequest``
            这个类（已核对本机 pymilvus 的 ``__init__`` 导出列表）。
            BM25 分支同样走 ``AnnSearchRequest``，靠两点区分于稠密分支：

              * ``param={"metric_type": "BM25"}`` —— 告诉服务端按 BM25 打分；
              * ``data=[query_text]`` —— 传**原始文本**而非向量。

            pymilvus 在 ``prepare._prepare_placeholder_str`` 中对
            ``isinstance(data[0], str)`` 有专门分支，会编码成 ``VARCHAR``
            placeholder 发给服务端，由服务端用字段的 analyzer 分词后计算
            BM25 权重。这正是全文检索的查询通道。

            MilvusClient 的 ``hybrid_search`` 参数名是 ``ranker``（不是 ORM 的 ``rerank``）。
        """
        # 稠密分支请求：传向量，按 COSINE 度量
        dense_req = AnnSearchRequest(
            data=[vector],
            anns_field=FIELD_DENSE,
            param={"metric_type": "COSINE", "params": {"ef": max(HNSW_SEARCH_EF, limit)}},
            limit=limit,
        )

        # 稀疏(BM25)分支请求：传原始文本，按 BM25 度量。
        # data 里的字符串会被编码成 VARCHAR placeholder，服务端负责分词与打分。
        sparse_req = AnnSearchRequest(
            data=[query_text],
            anns_field=FIELD_SPARSE,
            param={"metric_type": "BM25"},
            limit=limit,
        )

        results = self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[dense_req, sparse_req],
            # RRFRanker 只使用名次，因此无需关心两分支分数不可比的问题
            ranker=RRFRanker(k=rrf_k),
            limit=limit,
            output_fields=list(OUTPUT_FIELDS),
        )
        return self._flatten_results(results)

    # ------------------------------------------------------------------
    # 通用
    # ------------------------------------------------------------------
    def num_entities(self) -> int:
        """集合中的实体总数，用于健康检查与入库后校验。

        MilvusClient 没有 ``num_entities`` 属性，改从集合统计里取 row_count。
        """
        try:
            stats = self.client.get_collection_stats(self.collection_name)
            return int(stats.get("row_count", -1))
        except Exception:  # pragma: no cover - 集合未加载等边界情况
            logger.warning("读取 row_count 失败", exc_info=True)
            return -1

    @staticmethod
    def _flatten_results(results: Any) -> list[RawHit]:
        """把 MilvusClient 的检索结果拍平成 RawHit 列表。

        MilvusClient 的返回结构是**嵌套字典列表**（与 ORM 时代的 Hit 对象不同）::

            [[{"id": 1, "distance": 0.83, "entity": {"text": "...", ...}}, ...]]

        外层列表对应每个查询（本项目每次只传一个查询，因此取第 0 个），
        内层按相关性降序排列。

        Args:
            results: ``client.search()`` / ``client.hybrid_search()`` 的返回值。

        Returns:
            RawHit 列表，顺序保持 Milvus 给出的相关性降序。
        """
        hits: list[RawHit] = []
        if not results:
            return hits

        for hit in results[0]:
            # entity 里是通过 output_fields 请求的标量字段
            entity = hit.get("entity") or {}

            hits.append(
                RawHit(
                    key=str(hit.get("id")),
                    text=str(entity.get(FIELD_TEXT, "") or ""),
                    doc_id=str(entity.get(FIELD_DOC_ID, "") or ""),
                    source=str(entity.get(FIELD_SOURCE, "") or ""),
                    chunk_index=int(entity.get(FIELD_CHUNK_INDEX, -1) or -1),
                    score=float(hit.get("distance", 0.0)),
                )
            )
        return hits
