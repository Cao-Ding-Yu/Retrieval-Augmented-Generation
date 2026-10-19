"""服务启动入口。

两种启动方式：
    1. 命令行：``rag-app``（安装后）或 ``python -m rag_app.main``
    2. 直接交给 uvicorn：``uvicorn rag_app.api.app:create_app --factory``

关于 ``reload``：
    开发期开启会自动重载代码，但同时会**重新加载重排模型**（几秒到几十秒）。
    因此在调整检索逻辑时，建议关掉 reload 手动重启，避免频繁等待。
"""

from __future__ import annotations

import logging

import uvicorn

from rag_app.api.app import create_app
from rag_app.config import get_settings
# 与命令行工具共用同一套日志配置，避免两处漂移
from rag_app.logging_setup import configure_logging


def main() -> None:
    """启动 HTTP 服务。"""
    settings = get_settings()
    configure_logging(settings.log_level)

    logger = logging.getLogger(__name__)
    logger.info("=" * 70)
    logger.info("本地 RAG 知识库问答服务")
    logger.info("  对话模型 : %s", settings.chat_model)
    logger.info("  嵌入模型 : %s (dim=%d)", settings.embedding_model, settings.embedding_dim)
    logger.info("  Milvus   : %s / %s", settings.milvus_uri, settings.milvus_collection)
    logger.info("  重排模型 : %s (enabled=%s)", settings.reranker_model, settings.rerank_enabled)
    logger.info("  监听地址 : http://%s:%d", settings.host, settings.port)
    logger.info("  接口文档 : http://%s:%d/docs", settings.host, settings.port)
    logger.info("=" * 70)

    uvicorn.run(
        # 传工厂函数而非实例：让 uvicorn 自己管理事件循环，
        # 也让 lifespan 在正确的时机执行（应用实例在 worker 进程内创建）
        create_app(),
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        # 开发期如需热重载，改为 uvicorn.run("rag_app.api.app:create_app", factory=True, reload=True)
    )


if __name__ == "__main__":
    main()
