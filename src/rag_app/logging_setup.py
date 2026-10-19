"""日志配置。

抽成独立模块的原因：Web 服务（main.py）与命令行工具（cli.py）需要完全一致的
日志格式与降噪策略。若各写一份，迟早会漂移成"命令行看得见、服务里看不见"。

## 关于 PyMilvus 的弃用警告（已解决）

pymilvus 3.x 曾对 ORM 风格 API（``Collection.search`` / ``query`` / ``num_entities``）
抛出 ``PyMilvusDeprecationWarning``，提示这些 API 将在 3.1 中移除。

本项目**已完成到 MilvusClient 的迁移**（见 ``retrieval/milvus_store.py``），
不再调用任何被弃用的接口，因此这里也**不再需要**屏蔽警告的临时措施 ——
日志里若再次出现此类警告，说明有新代码用回了 ORM API，应当直接修掉而不是屏蔽。
"""

from __future__ import annotations

import logging

# 第三方库中在 INFO 级别过于聒噪的 logger。
# 不压低它们的话，连接心跳与 HTTP 细节会盖住业务日志。
NOISY_LOGGERS = (
    "pymilvus",
    "milvus_lite",
    "urllib3",
    "httpx",
    "httpcore",
    "sentence_transformers",
    "transformers",
    "filelock",
)


def configure_logging(level: str = "info", *, verbose: bool = False) -> None:
    """配置全局日志。

    Args:
        level: 日志级别名（debug/info/warning/error）。
        verbose: 为 True 时强制 DEBUG 级别（命令行 -v 使用）。
    """
    effective = "DEBUG" if verbose else level.upper()

    logging.basicConfig(
        level=effective,
        format="%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s",
        datefmt="%H:%M:%S",
    )

    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
