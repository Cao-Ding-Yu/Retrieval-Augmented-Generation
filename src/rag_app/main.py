"""
服务启动入口
rag-app or python -m rag_app.main
"""

from __future__ import annotations

import logging

import uvicorn

from rag_app.api import create_app
from rag_app.config import configure_logging, get_settings


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)

    logger = logging.getLogger(__name__)
    logger.info("=" * 64)
    logger.info("本地 RAG 知识库问答")
    logger.info("  对话模型 : %s", settings.chat_model)
    logger.info("  嵌入模型 : %s (dim=%d)", settings.embedding_model, settings.embedding_dim)
    logger.info("  Milvus   : %s / %s", settings.milvus_uri, settings.milvus_collection)
    logger.info("  重排模型 : %s (enabled=%s)", settings.reranker_model, settings.rerank_enabled)
    logger.info("  监听地址 : http://%s:%d", settings.host, settings.port)
    logger.info("  接口文档 : http://%s:%d/docs", settings.host, settings.port)
    logger.info("=" * 64)

    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
