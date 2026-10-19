"""Milvus 集合结构定义与版本要求。

本模块是"混合检索"能成立的地基，设计要点：

1. **稠密字段** ``dense``：存嵌入模型输出的语义向量，用于 ANN 检索。
2. **稀疏字段** ``sparse``：承载 BM25 词法信号。它由 Schema 上注册的
   内置 BM25 ``Function`` 产出 —— 服务端在写入时自动把 ``text`` 转成稀疏向量，
   因此**应用层完全不用管稀疏编码**，也不需要第二个模型。
   这是 Milvus 2.5 引入的能力，本项目直接要求该版本（见 ``MIN_SERVER_VERSION``）。
3. **原始文本字段** ``text``：开启 analyzer，是 BM25 的输入来源，同时也用于
   最终拼装提示词，避免回表查询。
4. **溯源字段** ``doc_id`` / ``source`` / ``chunk_index``：让每个分块都能追溯回
   原文件与位置，前端据此渲染引用。

## 为什么只支持 Milvus 2.5+

Milvus 2.4 也能做混合检索，但稀疏向量必须由**客户端**用 BM25EmbeddingFunction
编码后写入，带来三个额外负担：
  * 编码前要先在语料上拟合 IDF 统计量，且拟合结果必须持久化
    （否则进程重启后新旧向量不在同一尺度，检索结果失真）；
  * 多一个分词语言配置项，中文语料配错就等于词法分支失效；
  * 检索侧要区分"传稀疏向量"还是"传原始文本"两套逻辑。

2.5 的服务端 Function 把这些问题一次解决。既然本项目面向 2.5+，
继续保留 2.4 回退路径只会增加阅读负担，因此明确**不兼容并给出清晰报错**，
而不是静默降级 —— 静默降级会让"检索效果为什么差"变得极难排查。
"""

from __future__ import annotations

from typing import Any

from pymilvus import DataType, MilvusClient

# ---------------------------------------------------------------------------
# pymilvus 版本能力探测
# ---------------------------------------------------------------------------
# Function / FunctionType 是 Milvus 2.5 引入的"服务端内置函数"能力。
# 本项目依赖它，因此这里不做静默降级：拿不到就直接报错并给出升级指引。
try:  # pragma: no cover - 取决于安装的 pymilvus 版本
    from pymilvus import Function, FunctionType
except ImportError as _exc:  # pragma: no cover
    raise ImportError(
        "当前 pymilvus 未提供 Function / FunctionType，无法使用 Milvus 服务端 BM25。\n"
        "本项目要求 pymilvus >= 3.0（Milvus >= 2.5）。请升级：pip install -U pymilvus"
    ) from _exc


# ---------------------------------------------------------------------------
# 字段名常量
# ---------------------------------------------------------------------------
# 集中定义，避免各处手写字符串拼错（Milvus 字段名写错只会在运行时报错）
FIELD_ID = "id"
FIELD_TEXT = "text"
FIELD_DOC_ID = "doc_id"
FIELD_SOURCE = "source"
FIELD_CHUNK_INDEX = "chunk_index"
FIELD_DENSE = "dense"
FIELD_SPARSE = "sparse"

# 建索引后需要能取回的字段（除向量外的标量字段）
OUTPUT_FIELDS: tuple[str, ...] = (
    FIELD_TEXT,
    FIELD_DOC_ID,
    FIELD_SOURCE,
    FIELD_CHUNK_INDEX,
)

# VARCHAR 最大长度：Milvus 要求显式声明。
# 取 8192 是因为分块目标 800 字符，留足余量以容纳未按预期切分的超长段落。
MAX_TEXT_LENGTH = 8192
# 来源路径可能较长（Windows 绝对路径 + 中文目录）
MAX_SOURCE_LENGTH = 512
# doc_id 用哈希，定长 64 足够（sha1 十六进制）
MAX_DOC_ID_LENGTH = 64

# BM25 函数名（服务端内置函数的名字，报错信息里会用到）
BM25_FUNCTION_NAME = "bm25_text_to_sparse"

# 要求的最低 Milvus 版本：服务端 BM25 Function 自 2.5 起提供
MIN_SERVER_VERSION: tuple[int, int] = (2, 5)


def parse_server_version(raw: str) -> tuple[int, int]:
    """把 Milvus 版本字符串解析成 (major, minor)。

    服务端返回的格式在不同版本间略有差异，可能出现 ``v2.5.4`` / ``2.5.4`` /
    ``2.5.4-lite`` 等形态，这里做宽松解析。

    Args:
        raw: ``utility.get_server_version()`` 的原始返回值。

    Returns:
        (major, minor) 元组；无法解析时为 (0, 0)。
    """
    if not raw:
        return (0, 0)

    # 去掉前缀 v/V 与可能的后缀（-lite、-gpu 等）
    cleaned = raw.strip().lstrip("vV").split("-")[0]
    parts = cleaned.split(".")
    try:
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
    except (ValueError, IndexError):
        return (0, 0)
    return (major, minor)


def supports_server_bm25(server_version_raw: str) -> bool:
    """判断服务端版本是否满足服务端 BM25 的要求。

    Args:
        server_version_raw: Milvus 服务端版本字符串。

    Returns:
        True 表示版本 >= :data:`MIN_SERVER_VERSION`。
    """
    return parse_server_version(server_version_raw) >= MIN_SERVER_VERSION


def build_schema(embedding_dim: int, analyzer_type: str = "") -> Any:
    """构建集合 Schema（含服务端 BM25 Function）。

    ## 为什么用 MilvusClient.create_schema()

    pymilvus 3.x 已弃用 ORM 风格的 ``Collection`` / ``FieldSchema`` API
    （将在 3.1 移除）。这里改用 MilvusClient 的 Schema 构建方式：
    ``create_schema()`` 返回一个 ``CollectionSchema``，再逐字段 ``add_field()``。

    注意 ``add_field`` 的签名与 ORM 时代不同：这里是
    ``add_field(field_name, datatype, **kwargs)``，而不是"传一个 FieldSchema 对象"。

    Args:
        embedding_dim: 稠密向量维度，必须与嵌入模型一致。
        analyzer_type: 文本字段的分析器类型。留空表示使用 Milvus 默认分析器；
            中文语料建议传 ``"chinese"``，否则 BM25 词法分支几乎不起作用
            （中文无空格，standard 分析器难以切出有意义的词项）。

    Returns:
        可直接传给 ``MilvusClient.create_collection(schema=...)`` 的 schema 对象。
    """
    # auto_id=True 让主键自增，客户端无需关心 ID 生成
    schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field(FIELD_ID, DataType.INT64, is_primary=True)

    # 原始文本：enable_analyzer=True 是 BM25 的前提，
    # Milvus 需要先对文本分词/建倒排才能计算词频与 IDF。
    #
    # analyzer_params 控制分词方式，对中文语料影响极大：
    #   * 不设置 -> 用 Milvus 默认的 standard 分析器（按空格/标点切分，
    #     中文没有空格，容易把整句当成一个 token，BM25 词法召回基本失效）；
    #   * {"type": "chinese"} -> 走中文分词（jieba 系），词法分支才有意义。
    # 默认留空是刻意的：分析器只在**建集合时**生效，改动它必须重建集合并重新入库，
    # 因此不做隐式变更，而是通过 BM25_ANALYZER_TYPE 显式开启（见 .env 与 README）。
    text_field_kwargs: dict = {"max_length": MAX_TEXT_LENGTH, "enable_analyzer": True}
    if analyzer_type:
        text_field_kwargs["analyzer_params"] = {"type": analyzer_type}
    schema.add_field(FIELD_TEXT, DataType.VARCHAR, **text_field_kwargs)

    # 文档级 ID：用于按文档批量删除、以及前端聚合同一文档的多个分块
    schema.add_field(FIELD_DOC_ID, DataType.VARCHAR, max_length=MAX_DOC_ID_LENGTH)
    # 来源文件路径/名，前端引用展示用
    schema.add_field(FIELD_SOURCE, DataType.VARCHAR, max_length=MAX_SOURCE_LENGTH)
    # 分块序号：便于在原文中定位，也便于按顺序拼接相邻块
    schema.add_field(FIELD_CHUNK_INDEX, DataType.INT64)
    # 稠密语义向量
    schema.add_field(FIELD_DENSE, DataType.FLOAT_VECTOR, dim=embedding_dim)
    # 稀疏词法向量：由下面的 BM25 Function 自动填充，插入数据时不要传它
    schema.add_field(FIELD_SPARSE, DataType.SPARSE_FLOAT_VECTOR)

    # 注册内置函数：写入 text 时，服务端自动计算出 sparse 字段。
    # 这样一来应用层插入数据时完全不用管稀疏向量。
    schema.add_function(
        Function(
            name=BM25_FUNCTION_NAME,
            function_type=FunctionType.BM25,
            input_field_names=[FIELD_TEXT],
            output_field_names=[FIELD_SPARSE],
        )
    )

    return schema


def dense_index_params() -> dict:
    """稠密向量的索引参数（供 ``IndexParams.add_index(**dense_index_params())`` 展开）。

    选 HNSW 而非 IVF 系列，原因：
      * 图索引在高召回区间表现更好，且无需训练阶段，插入即可查询；
      * 本地单机数据量（十万级）下，HNSW 的构建开销完全可以接受。

    COSINE 度量：文本嵌入向量通常已归一化，余弦相似度是这类模型的标准选择，
    也与训练目标一致。

    Note:
        MilvusClient 的 ``IndexParams.add_index`` 会把额外的 kwargs
        直接摊平进索引配置，因此这里**不要**再套一层 ``params={...}``。
    """
    return {
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "M": 16,                # 每个节点的邻居数：越大召回越高、内存越多
        "efConstruction": 200,  # 建图时的候选队列长度：越大索引质量越好、构建越慢
    }


def sparse_index_params() -> dict:
    """稀疏向量的索引参数（供 ``IndexParams.add_index(**sparse_index_params())`` 展开）。

    度量必须是 ``BM25``（不是 ``IP``）：稀疏向量由服务端 Function 生成，
    查询侧同样由服务端分词并计算查询词权重，整套打分都在 Milvus 内部完成。
    写成 ``IP`` 会导致分词与打分口径不一致，Milvus 会直接抛错或返回无意义结果。
    """
    return {
        "index_type": "SPARSE_INVERTED_INDEX",
        "metric_type": "BM25",
            # 倒排索引的丢弃阈值：权重过低的词项不建索引，以节省空间。
            # 对 BM25 这类本身很稀疏的表示，0.0 表示保留全部非零项。
        "drop_ratio_build": 0.0,
    }
