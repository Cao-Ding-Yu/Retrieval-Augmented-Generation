"""命令行工具：入库 / 检索调试 / 查看概况。

复用与 Web 服务**完全相同**的 ``RAG`` 组装流程，所以 CLI 里验证过的行为
就是线上行为，不会出现"脚本能搜到、接口搜不到"这类不一致。

    rag-app-cli ingest ./data
    rag-app-cli search "RRF 是怎么工作的"
    rag-app-cli stats
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from rag_app.config import configure_logging, get_settings
from rag_app.ingest import ingest
from rag_app.rag import RAG


async def _run(args: argparse.Namespace) -> int:
    """构造 RAG 实例并执行子命令。"""
    settings = get_settings()
    rag = RAG(settings)

    try:
        rag.start()
    except Exception as exc:
        print(f"\n[错误] 初始化失败：{exc}", file=sys.stderr)
        print(
            "请确认：\n"
            "  1) Milvus 已启动（默认 http://localhost:19530，且版本 >= 2.5）\n"
            "  2) Ollama 已启动（默认 http://localhost:11434）\n"
            "  3) .env 里的连接配置正确",
            file=sys.stderr,
        )
        return 2

    try:
        if args.command == "ingest":
            return await _cmd_ingest(rag, args)
        if args.command == "search":
            return await _cmd_search(rag, args)
        if args.command == "stats":
            return _cmd_stats(rag)
        print(f"[错误] 未知命令：{args.command}", file=sys.stderr)
        return 2
    finally:
        rag.close()


async def _cmd_ingest(rag: RAG, args: argparse.Namespace) -> int:
    """导入文件或目录。"""
    target = Path(args.path)
    if not target.exists():
        print(f"[错误] 路径不存在：{target}", file=sys.stderr)
        return 2

    print(f"开始入库：{target.resolve()}")
    print(f"  切分参数：chunk_size={rag.settings.chunk_size}, "
          f"overlap={rag.settings.chunk_overlap}")
    if target.is_dir():
        print("  目录会递归处理所有子目录")

    report = await ingest(rag.settings, rag.store, rag.embedder, target)

    print("\n入库完成：")
    print(f"  成功文档 : {report.files_loaded}")
    print(f"  写入分块 : {report.chunks_written}")
    if report.skipped:
        print(f"  跳过 {len(report.skipped)} 项：")
        for reason in report.skipped[:10]:
            print(f"    - {reason}")
    print(f"  集合总数 : {rag.store.count()}")
    return 0


async def _cmd_search(rag: RAG, args: argparse.Namespace) -> int:
    """只跑检索，打印打分明细（不调用生成模型）。"""
    hits = await rag.aretrieve(args.query)

    print(f"\n查询：{args.query}")
    print(f"命中：{len(hits)} 条\n")
    if not hits:
        print("（无命中）若知识库为空，请先执行 ingest 子命令。")
        return 0

    for rank, hit in enumerate(hits, start=1):
        # 分支名次能直接看出这条是语义命中的还是关键词命中的
        branches = []
        if hit.dense_rank is not None:
            branches.append(f"稠密#{hit.dense_rank + 1}")
        if hit.sparse_rank is not None:
            branches.append(f"稀疏#{hit.sparse_rank + 1}")

        print(f"[{rank}] {hit.source}  第 {hit.chunk_index + 1} 块")
        print(f"    命中分支: {' + '.join(branches) or '-'}   分数: {hit.score:.4f}")
        if hit.is_hybrid_hit:
            print("    ★ 语义与关键词同时命中")
        preview = hit.text[:160].replace("\n", " ")
        print(f"    内容: {preview}{'…' if len(hit.text) > 160 else ''}\n")

    if args.json:
        print(json.dumps([h.to_dict() for h in hits], ensure_ascii=False, indent=2))
    return 0


def _cmd_stats(rag: RAG) -> int:
    """打印知识库规模与检索配置。"""
    cfg = rag.settings
    info = {
        "集合": cfg.milvus_collection,
        "Milvus 版本": rag.store.server_version,
        "分块总数": rag.store.count(),
        "嵌入模型": f"{cfg.embedding_model} (dim={cfg.embedding_dim})",
        "对话模型": cfg.chat_model,
        "重排": "开" if rag.rerank_enabled else "关",
        "重排模型": cfg.reranker_model if rag.rerank_enabled else "（未启用）",
        "dense_top_k": cfg.dense_top_k,
        "sparse_top_k": cfg.sparse_top_k,
        "rrf_k": cfg.rrf_k,
        "final_top_k": cfg.final_top_k,
    }
    width = max(len(k) for k in info)
    for key, value in info.items():
        print(f"  {key.ljust(width)} : {value}")

    # 降级原因要显式打印出来，不能只留在日志里
    if rag.rerank_warning:
        print(f"\n警告：{rag.rerank_warning}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-app-cli",
        description="本地 RAG 知识库的命令行工具（入库 / 检索 / 概况）",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="输出 DEBUG 日志")

    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser(
        "ingest", help="把文件或目录导入知识库（目录递归处理）"
    )
    p_ingest.add_argument("path", help="文件或目录路径")

    p_search = sub.add_parser("search", help="执行检索并查看打分明细")
    p_search.add_argument("query", help="查询文本")
    p_search.add_argument("--json", action="store_true", help="额外输出完整 JSON")

    sub.add_parser("stats", help="查看知识库规模与当前检索配置")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_logging(verbose=args.verbose)
    raise SystemExit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
