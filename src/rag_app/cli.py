"""
命令行工具：批量入库与检索调试

用途：
    * 建库时批量导入语料
    * 不启动 Web 服务即可验证检索效果

用法：
    python -m rag_app.cli ingest ./data
    python -m rag_app.cli search "text" --top-k 5
    python -m rag_app.cli stats

设计说明：
    CLI 复用与 Web 服务完全相同的 ``build_context``，
    因此它验证过的行为就是线上行为 —— 不会出现"脚本里能搜到、接口里搜不到"
    这类因两套初始化逻辑不一致导致的诡异问题
"""

from __future__ import annotations

import sys
import json
import argparse
import asyncio
from pathlib import Path

from rag_app.deps import AppContext, build_context, teardown_context
from rag_app.logging_setup import configure_logging


def _require_ready(ctx: AppContext) -> None:
    """
    确认上下文就绪，否则打印原因并退出

    明确错误信息 + 非零退出码
    """
    if ctx.ready: return
    print(f"\n[错误] 服务未就绪：{ctx.startup_error}", file=sys.stderr)
    print(
        "请确认：\n"
        "  1) Milvus 已启动（默认 http://localhost:19530）\n"
        "  2) Ollama 已启动（默认 http://localhost:11434）\n"
        "  3) .env 中的连接配置正确",
        file=sys.stderr,
    )
    raise SystemExit(2)


# ---------------------------------------------------------------------------
# 子命令实现
# ---------------------------------------------------------------------------
async def cmd_ingest(ctx: AppContext, args: argparse.Namespace) -> int:
    """执行入库"""
    assert ctx.ingest is not None

    target = Path(args.path)
    if not target.exists():
        print(f"[错误] 路径不存在：{target}", file=sys.stderr)
        return 2

    print(f"开始入库：{target.resolve()}")
    print(f"  分块参数：chunk_size={ctx.settings.chunk_size}, overlap={ctx.settings.chunk_overlap}")

    report = await ctx.ingest.ingest_path(target, recursive=not args.no_recursive)

    print("\n入库完成：")
    print(f"  扫描文件   : {report.files_scanned}")
    print(f"  成功入库   : {report.files_loaded} 个文档")
    print(f"  写入分块   : {report.chunks_written}")
    if report.skipped:
        print(f"  跳过 {len(report.skipped)} 项：")
        for reason in report.skipped[:10]:
            print(f"    - {reason}")
        if len(report.skipped) > 10:
            print(f"    … 其余 {len(report.skipped) - 10} 项略")

    if ctx.store is not None:
        print(f"  当前集合实体总数 : {ctx.store.num_entities()}")

    return 0


async def cmd_search(ctx: AppContext, args: argparse.Namespace) -> int:
    """执行检索并打印打分明细"""
    assert ctx.retriever is not None

    chunks = await ctx.retriever.retrieve(args.query)

    print(f"\n查询：{args.query}")
    print(f"融合模式：{ctx.fusion_mode}（BM25 由 Milvus 服务端计算）")
    print(f"命中：{len(chunks)} 条\n")

    if not chunks:
        print("（无命中）若知识库为空，请先执行 ingest 子命令。")
        return 0

    for rank, chunk in enumerate(chunks, start=1):
        # 命中分支信息用于判断结果来自语义还是关键词
        branches = []
        if chunk.dense_rank is not None:
            branches.append(f"稠密#{chunk.dense_rank + 1}")
        if chunk.sparse_rank is not None:
            branches.append(f"稀疏#{chunk.sparse_rank + 1}")
        branch_text = " + ".join(branches) if branches else "-"

        score_text = f"重排={chunk.rerank_score:.4f}" if chunk.rerank_score is not None else "-"
        rrf_text = f"RRF={chunk.rrf_score:.6f}" if chunk.rrf_score is not None else "-"

        print(f"[{rank}] {chunk.source}  第 {chunk.chunk_index + 1} 块")
        print(f"    命中分支: {branch_text}   {score_text}   {rrf_text}")
        if chunk.is_hybrid_hit:
            print("    ★ 稠密与稀疏同时命中")
        # 正文只打印前 160 字符，避免刷屏
        preview = chunk.text[:160].replace("\n", " ")
        print(f"    内容: {preview}{'…' if len(chunk.text) > 160 else ''}")
        print()

    if args.json:
        print(json.dumps([c.to_dict() for c in chunks], ensure_ascii=False, indent=2))

    return 0


async def cmd_stats(ctx: AppContext, _args: argparse.Namespace) -> int:
    """打印知识库与检索配置概况。"""
    assert ctx.store is not None

    rerank_impl = type(ctx.reranker).__name__
    info = {
        "集合": ctx.settings.milvus_collection,
        "Milvus 版本": ctx.store.server_version,
        "融合模式": ctx.fusion_mode,
        "实体总数": ctx.store.num_entities(),
        "嵌入模型": f"{ctx.settings.embedding_model} (dim={ctx.settings.embedding_dim})",
        "对话模型": ctx.settings.chat_model,
        "重排实现": rerank_impl,
        "重排模型": ctx.settings.reranker_model if rerank_impl != "NoOpReranker" else "（未启用）",
        "dense_top_k": ctx.settings.dense_top_k,
        "sparse_top_k": ctx.settings.sparse_top_k,
        "rrf_k": ctx.settings.rrf_k,
        "rerank_top_n": ctx.settings.rerank_top_n,
        "final_top_k": ctx.settings.final_top_k,
    }

    width = max(len(k) for k in info)
    for key, value in info.items():
        print(f"  {key.ljust(width)} : {value}")

    if ctx.warnings:
        print("\n警告：")
        for warning in ctx.warnings:
            print(f"  - {warning}")

    return 0


async def cmd_eval(ctx: AppContext, args: argparse.Namespace) -> int:
    """在金标准集上评估检索质量，可选叠加 RAGAS。

    为什么把两种评估放在同一个命令里：
    检索指标是"秒级、零依赖、必定能跑"，RAGAS 是"分钟级、需额外依赖、可能失败"。
    同一条命令里按 ``--ragas`` 开关切换，能让人自然地先看检索、再看生成。
    """
    # 延迟导入：不跑评估就不该加载评估模块（ragas_eval 里也只是惰性导入 ragas）
    from rag_app.eval.dataset import load_golden_set
    from rag_app.eval.runner import collect_records, run_retrieval_eval

    assert ctx.retriever is not None

    # ---- 加载金标准 ----------------------------------------------------
    # 先用当前集合里真实的来源文件名做校验，拦住"标注了不存在的文件"
    # 这类会让样本永远无法命中、而且从指标上完全看不出来的错误。
    try:
        known_sources = _collect_known_sources(ctx)
        samples = load_golden_set(args.golden, known_sources=known_sources or None)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 2

    if args.limit:
        samples = samples[: args.limit]

    top_k = args.top_k or ctx.settings.final_top_k

    print(f"金标准集：{args.golden}（{len(samples)} 条）")
    print(f"评估截断：top-{top_k}")
    print(f"检索配置：融合={ctx.fusion_mode}  重排={'开' if type(ctx.reranker).__name__ != 'NoOpReranker' else '关'}")
    print()

    # ---- 第一层：检索指标（不调用 LLM）----------------------------------
    try:
        report = await run_retrieval_eval(ctx.retriever, samples, k=top_k)
    except ValueError as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 2

    print("检索指标：")
    print(f"  命中率@k : {report.hit_rate:.4f}   （top-k 里至少命中一个期望来源的比例）")
    print(f"  召回@k   : {report.recall:.4f}   （命中的期望来源 / 期望来源总数）")
    print(f"  MRR@k    : {report.mrr:.4f}   （第一个命中位置的倒数均值）")
    print(f"  nDCG@k   : {report.ndcg:.4f}   （整体排序质量）")

    # 未命中的样本单独列出来 —— 这是最该看的部分
    misses = report.misses()
    if misses:
        print(f"\n未命中 {len(misses)} 条（这些是检索要优先解决的问题）：")
        for outcome in misses[:10]:
            print(f"  - {outcome.question[:48]}")
            print(f"      期望: {outcome.expected_sources}")
            print(f"      实际: {outcome.retrieved_sources}")
        if len(misses) > 10:
            print(f"  … 其余 {len(misses) - 10} 条略")

    result: dict = {"retrieval": report.to_dict()}

    # ---- 第二层：RAGAS（可选）------------------------------------------
    if args.ragas:
        from rag_app.eval.ragas_eval import RagasUnavailableError, evaluate_with_ragas

        assert ctx.chat is not None
        print(f"\n开始 RAGAS 评估（指标={args.metrics}）…")
        print("  提示：本地裁判模型可能无法稳定输出结构化 JSON，")
        print("        失败会表现为大量 NaN。建议先用 --limit 1 验证。")

        records = await collect_records(ctx.retriever, ctx.chat, samples, limit=args.limit)

        if not records:
            print("[错误] 没有收集到任何评估记录（检索全为空？）", file=sys.stderr)
            return 2

        if args.save_records:
            import json as _json

            with open(args.save_records, "w", encoding="utf-8") as fh:
                for record in records:
                    fh.write(_json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
            print(f"  中间记录已保存：{args.save_records}")

        metric_names = [m.strip() for m in args.metrics.split(",") if m.strip()]

        try:
            ragas_result = evaluate_with_ragas(
                records,
                # 复用同一个模型实例当裁判，避免重复加载本地大模型
                llm=ctx.chat.llm,
                embeddings=ctx.embedder,
                metric_names=metric_names,
            )
        except RagasUnavailableError as exc:
            print(f"\n[错误] RAGAS 不可用：{exc}", file=sys.stderr)
            return 2

        print("\nRAGAS 指标：")
        for name, value in ragas_result["metrics"].items():
            print(f"  {name:20s}: {value:.4f}")
        if ragas_result["skipped"]:
            print(f"  （已跳过不可用的指标：{ragas_result['skipped']}）")

        result["ragas"] = ragas_result["metrics"]

    if args.json:
        import json as _json

        print("\n" + _json.dumps(result, ensure_ascii=False, indent=2))

    return 0


def _collect_known_sources(ctx: AppContext) -> set[str]:
    """从集合里取出现过的来源文件名，用于校验金标准标注。

    用 query 拉一批样本即可 —— 同一批文档的来源通常高度重复，
    不需要遍历全部数据。取不到时返回空集合，调用方会跳过校验。
    """
    assert ctx.store is not None
    try:
        # 用主键范围过滤取若干行，避免全表扫描
        rows = ctx.store.client.query(
            ctx.store.collection_name,
            filter="",
            output_fields=["source"],
            limit=1000,
        )
        return {str(row.get("source", "")) for row in rows if row.get("source")}
    except Exception:
        # 校验只是"锦上添花"，拿不到就跳过，不能因此让评估失败
        print("（提示：未能读取集合中的来源列表，跳过标注校验）")
        return set()


# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        prog="rag-app-cli",
        description="本地 RAG 知识库的命令行工具",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="输出 DEBUG 日志")

    sub = parser.add_subparsers(dest="command", required=True)

    # 注册 ingest 子命令
    p_ingest = sub.add_parser("ingest", help="将文件或目录导入知识库")
    p_ingest.add_argument("path", help="文件或目录路径")
    p_ingest.add_argument("--no-recursive", action="store_true", help="只处理目录第一层，不递归")

    # 注册 search 子命令
    p_search = sub.add_parser("search", help="执行检索并查看打分明细")
    p_search.add_argument("query", help="查询文本")
    p_search.add_argument("--json", action="store_true", help="额外输出完整 JSON")

    # 注册 stats 子命令
    sub.add_parser("stats", help="查看知识库规模与当前检索配置")

    # 注册 eval 子命令
    p_eval = sub.add_parser("eval", help="在金标准集上评估检索质量（可选跑 RAGAS）")
    p_eval.add_argument("golden", help="金标准 JSONL 文件路径")
    p_eval.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=None,
        help="截断到前 k 条再算指标；默认用配置里的 FINAL_TOP_K",
    )
    p_eval.add_argument(
        "--ragas",
        action="store_true",
        help="额外跑 RAGAS 生成质量指标（需先 pip install -e \".[eval]\"，且很慢）",
    )
    p_eval.add_argument(
        "--metrics",
        default="faithfulness",
        help="RAGAS 指标，逗号分隔。可选：faithfulness,answer_relevancy,context_precision,context_recall",
    )
    p_eval.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只评估前 N 条（跑 RAGAS 时强烈建议先用 --limit 1 验证）",
    )
    p_eval.add_argument("--json", action="store_true", help="以 JSON 输出结果")
    p_eval.add_argument("--save-records", help="把 RAGAS 用的中间记录另存为 JSONL，便于事后复查")

    return parser


async def _run(args: argparse.Namespace) -> int:
    """构造上下文并分发到具体子命令。"""
    # 复用服务端同一套组装逻辑，保证 CLI 与 HTTP 行为一致
    ctx = build_context()
    try:
        _require_ready(ctx)
        if args.command == "ingest":
            return await cmd_ingest(ctx, args)
        if args.command == "search":
            return await cmd_search(ctx, args)
        if args.command == "stats":
            return await cmd_stats(ctx, args)
        if args.command == "eval":
            return await cmd_eval(ctx, args)
        print(f"[错误] 未知命令：{args.command}", file=sys.stderr)
        return 2
    finally:
        teardown_context(ctx)


def main() -> None:
    # 获取传入参数
    parser = build_parser()
    args = parser.parse_args()
    # 日志配置
    configure_logging(verbose=args.verbose)
    # 执行指令
    exit_code = asyncio.run(_run(args))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
