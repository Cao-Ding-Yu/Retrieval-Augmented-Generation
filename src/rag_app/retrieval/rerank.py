"""
交叉编码器重排（Cross-Encoder Rerank）

**交叉编码器**（cross-encoder）把 (查询, 文档) 拼成一个序列一起送进模型，
让注意力机制直接在两者之间做交互，相关性判断更准确
代价是无法预计算：N 个候选就要跑 N 次前向，复杂度与候选数成正比

标准的做法是两阶段：

    第一阶段（召回）：向量检索，从百万级里快速取回 top-20~30  —— 求"不漏"
    第二阶段（精排）：交叉编码器只对这几十条逐对打分           —— 求"排准"

本模块实现第二阶段。每次请求都加载会让响应时间从毫秒级退化到秒级，
因此在 FastAPI 的 lifespan 中作为单例初始化，见 rag_app/deps.py

"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# 送入交叉编码器的单条文本最大 token 数
# 512 是 BERT 系模型的典型上限；超出部分会被截断，这也是 chunk_size 取 800 字符的另一个理由
DEFAULT_MAX_LENGTH = 512
# 前向推理的批大小，批太小算力用不满，太大则显存/内存峰值升高
DEFAULT_BATCH_SIZE = 16


@runtime_checkable
class Reranker(Protocol):
    """
    重排器协议

    定义成 Protocol 的目的是让上层依赖"行为"而非"具体实现"：
    生产用 CrossEncoderReranker，关闭重排时用 NoOpReranker，
    测试里可以直接替换成假实现，无需加载真实模型。
    """

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
    ) -> list[tuple[int, float]]:
        """对候选文档按与查询的相关性重新排序。

        Args:
            query: 用户查询
            documents: 候选文档正文列表
            top_k: 只返回前 k 条；None 表示全部返回

        Returns:
            ``(原始下标, 分数)`` 列表，按分数从高到低排列。
            返回下标而非文档本身，是为了让调用方能把分数关联回原始对象
            （那些对象上还挂着溯源元数据）。
        """
        ...


def hf_cache_root() -> Path:
    """
    返回 HuggingFace 实际的 hub 缓存根目录

    刻意从 ``huggingface_hub.constants.HF_HUB_CACHE`` 读取，而不是自己拼
    ``~/.cache/huggingface/hub``：该常量已经正确处理了 ``HF_HOME``、
    ``HF_HUB_CACHE``、``HUGGINGFACE_HUB_CACHE`` 三个环境变量的优先级。
    自己拼路径很容易与库的实际取值不一致，从而"明明缓存存在却找不到"。

    Returns:
        hub 缓存目录（本机为 C:/Users/<用户>/.cache/huggingface/hub）。
    """
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        return Path(HF_HUB_CACHE)
    except Exception as e:
        logger.warning("无法读取 HuggingFace 缓存目录常量，回退到默认路径", exc_info=True)
        return Path.home() / ".cache" / "huggingface" / "hub"


def resolve_snapshot_dir(model_name: str, cache_root: Path | None = None) -> Path | None:
    """
    把 ``作者/模型名`` 解析成本地快照目录的绝对路径

    HF 缓存布局::

        <hub>/models--<作者>--<模型名>/
            refs/main                 # 内容为 commit 哈希
            snapshots/<commit>/       # 真正的模型文件

    Args:
        model_name: 形如 ``BAAI/bge-reranker-v2-m3`` 的仓库名
        cache_root: hub 缓存根目录；None 时自动探测

    Returns:
        快照目录路径；找不到时返回 None
    """
    if "/" not in model_name: return None

    root = cache_root or hf_cache_root()
    org, _, name = model_name.partition("/")
    repo_dir = root / f"models--{org}--{name}"
    snapshots_dir = repo_dir / "snapshots"

    if not snapshots_dir.is_dir():
        return None

    # 优先按 refs/<版本> 指定的 commit 定位
    ref_file = repo_dir / "refs" / "main"
    if ref_file.is_file():
        try:
            commit = ref_file.read_text(encoding="utf-8").strip()
        except OSError: commit = ""

        if commit:
            candidate = snapshots_dir / commit
            if candidate.is_dir():
                return candidate

    candidates = [d for d in snapshots_dir.iterdir() if d.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda d: d.stat().st_mtime)


def describe_hf_env() -> str:
    """汇总与 HF 加载相关的环境变量，用于排障日志"""
    keys = ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    parts = [f"{k}={os.environ.get(k, '<未设置>')}" for k in keys]
    return "，".join(parts)


class CrossEncoderReranker:
    """
    基于 sentence-transformers CrossEncoder 的实际重排器

    Args:
        model_name: 交叉编码器模型名。多语言场景推荐
            ``BAAI/bge-reranker-v2-m3``（与 bge-m3 同族，效果对齐）；
            内存/延迟敏感时可换 ``BAAI/bge-reranker-base``。
        device: 推理设备（"cpu" / "cuda"）。None 表示自动选择。
        max_length: 单条文本最大 token 数。
        batch_size: 推理批大小。
        offline_first: 优先以 ``local_files_only=True`` 加载（默认 True）。
        allow_download: 离线加载失败（缓存缺失）时是否回退到联网下载。
        cache_folder: HF 缓存目录；None 表示用默认位置。

    Note:
        **为什么要 offline_first**：HuggingFace 的 ``from_pretrained`` 默认会先向
        huggingface.co 发请求校验 etag（判断远端有没有更新），命中缓存也照发。
        在国内网络下这一步经常超时数秒甚至直接失败，表现为"模型明明在本地，
        每次启动却还要联网"。

        ``local_files_only=True`` 会跳过全部网络调用，只读本地缓存。
        sentence-transformers 会把该参数一路透传到
        ``AutoConfig.from_pretrained`` / ``transformers.from_pretrained``，
        因此是**纯缓存读取**，不是"先联网失败再回退"。

        策略是"离线优先 + 可选回退"：
          * 缓存命中 → 零网络，启动快；
          * 缓存缺失 → 若 allow_download 为真则联网下载一次，之后又回到离线；
            为假则直接报错，并给出明确的操作指引。
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str | None = None,
        max_length: int = DEFAULT_MAX_LENGTH,
        batch_size: int = DEFAULT_BATCH_SIZE,
        offline_first: bool = True,
        allow_download: bool = True,
        cache_folder: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._max_length = max_length
        self._batch_size = batch_size
        self._offline_first = offline_first
        self._allow_download = allow_download
        self._cache_folder = cache_folder
        # 设备在这里就确定下来（而不是交给库去猜），便于日志里明确写出实际使用了什么
        self._device = device or self._auto_device()
        self._model = None
        """延迟加载：构造时不下载/加载模型，避免 import 阶段就产生秒级阻塞。"""
        self._loaded_offline = False
        """记录本次是否以离线方式加载成功，供健康检查/日志展示。"""

        self._resolved_path: str | None = None
        """实际加载所用的本地快照路径（若走了本地目录加载）。"""

    @staticmethod
    def _auto_device() -> str:
        """自动选择推理设备：有 CUDA 就用 GPU，否则退回 CPU"""
        try:
            import torch
            if torch.cuda.is_available(): return "cuda"
        except Exception:  # pragma: no cover - torch 缺失或 CUDA 异常
            logger.warning("无法探测 CUDA，重排将使用 CPU", exc_info=True)
        return "cpu"

    def _build_model(
        self,
        local_files_only: bool = False,
        model_path: str | None = None,
    ):
        """
        构造 CrossEncoder 实例。

        Args:
            local_files_only: True 表示禁止网络请求
            model_path: 本地快照目录的绝对路径。给出时以**本地目录**方式加载，HuggingFace 会跳过全部仓库解析与 etag 校验

        Returns:
            已加载权重的 CrossEncoder 实例。
        """
        from sentence_transformers import CrossEncoder

        # 本地目录优先：以目录加载时无需也不应再传 local_files_only
        target = model_path or self._model_name

        kwargs: dict[str, object] = {
            "device": self._device,
            "max_length": self._max_length,
        }
        if model_path is None:
            kwargs["local_files_only"] = local_files_only
            if self._cache_folder:
                kwargs["cache_folder"] = self._cache_folder

        return CrossEncoder(target, **kwargs)

    def load(self) -> None:
        """
        显式加载模型。应在应用启动时调用一次

        单独提供 load() 而不是在 __init__ 里加载，是为了让"构造对象"与
        "占用 2GB 内存"这两件事解耦，测试时可以只构造不加载

        ## 三级加载策略（从最稳到最兜底）

        1. **本地快照目录**：自己按 HF 缓存布局定位 ``snapshots/<commit>/``，
           把绝对路径交给 CrossEncoder。以本地目录加载会完全跳过仓库解析，
        2. **``local_files_only=True``**：仍按仓库名加载但禁止联网
           它在文件齐全时也能工作，但依赖缓存元数据完整
        3. **联网加载**：前两级都失败才走这里

        Raises:
            RuntimeError: 缓存缺失且不允许下载时
        """
        if self._model is not None: return

        logger.info(
            "正在加载重排模型 %s（device=%s, 离线优先=%s）…",
            self._model_name,
            self._device,
            self._offline_first,
        )
        logger.debug("HF 环境：%s；hub 缓存=%s", describe_hf_env(), hf_cache_root())

        if self._offline_first:
            # ---- 第 1 级：直接定位本地快照目录 ----
            snapshot = resolve_snapshot_dir(self._model_name)
            if snapshot is not None:
                try:
                    logger.info("在本地缓存中找到模型快照：%s", snapshot)
                    self._model = self._build_model(model_path=str(snapshot))
                    self._resolved_path = str(snapshot)
                    self._loaded_offline = True
                    logger.info("重排模型已从本地快照加载（无网络请求）")
                    return
                except Exception:
                    # 快照存在但读取失败（文件损坏/权限问题）。
                    # 这里必须打完整堆栈：单行消息看不出是缺哪个文件。
                    logger.warning("从本地快照加载失败，将尝试其他方式", exc_info=True)
            else:
                logger.warning(
                    "未在缓存中找到 %s 的快照目录（缓存根=%s），"
                    "将尝试按仓库名离线加载",
                    self._model_name,
                    hf_cache_root(),
                )

            # ---- 第 2 级：按仓库名 + local_files_only ----
            try:
                self._model = self._build_model(local_files_only=True)
                self._loaded_offline = True
                logger.info("重排模型已从本地缓存加载（无网络请求）")
                return
            except Exception as exc:
                # 打完整堆栈：真正的缺失文件名在异常链里，单行 str(exc) 看不到
                logger.warning("按仓库名离线加载失败：%s", exc, exc_info=True)

                if not self._allow_download:
                    raise RuntimeError(
                        f"重排模型 {self._model_name} 无法从本地加载，且当前配置禁止联网下载。\n"
                        f"已尝试的缓存根目录：{hf_cache_root()}\n"
                        f"HF 环境变量：{describe_hf_env()}\n"
                        "排查步骤：\n"
                        "  1) 确认模型目录存在：<缓存根>/models--<作者>--<模型名>/snapshots/<commit>/，\n"
                        "     且其中包含 config.json 与 model.safetensors（权重文件）；\n"
                        "  2) 若缓存不在默认位置，在 .env 里设置 HF_HOME 指向它；\n"
                        "  3) 首次使用可把 RERANKER_ALLOW_DOWNLOAD 设为 true 下载一次；\n"
                        "  4) 暂时不需要重排：把 RERANK_ENABLED 设为 false。"
                    ) from exc

                logger.warning(
                    "本地缓存不可用，将联网下载重排模型 %s（仅此一次，后续回到离线）",
                    self._model_name,
                )

        # ---- 第 3 级：联网加载 ----
        self._model = self._build_model(local_files_only=False)
        self._loaded_offline = False
        logger.info("重排模型加载完成（本次经由网络/默认路径）")

    @property
    def resolved_path(self) -> str | None:
        """本次实际加载所用模型的位置（本地快照路径或 None），供健康检查展示。"""
        return self._resolved_path

    @property
    def loaded_offline(self) -> bool:
        """本次加载是否走了纯离线路径，用于健康检查确认配置真的生效。"""
        return self._loaded_offline

    def warmup(self) -> None:
        """
        用一对假数据预热，把首帧开销提前到启动阶段

        目的：避免第一个真实用户请求承担模型初始化/图优化的额外延迟
        """
        self.load()
        assert self._model is not None
        try:
            self._model.predict([("预热查询", "预热文档")], batch_size=1)
            logger.info("重排模型预热完成")
        except Exception:  # pragma: no cover - 预热失败不应阻止服务启动
            logger.warning("重排模型预热失败，不影响功能", exc_info=True)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def device(self) -> str:
        return self._device

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
    ) -> list[tuple[int, float]]:
        """
        对候选文档重排

        Args:
            query: 用户查询
            documents: 候选文档正文
            top_k: 返回条数上限

        Returns:
            ``(原始下标, 分数)`` 列表，分数降序
        """
        if not documents: return []

        self.load()
        assert self._model is not None

        # 构造 (查询, 文档) 对
        pairs = [(query, doc) for doc in documents]

        # predict 返回 numpy 数组，每个元素是该对的 logit 分数（越大越相关）
        # 我们不关心绝对值的量纲，只用到相对大小，因此不需要额外的激活函数
        scores = self._model.predict(
            pairs,
            batch_size=self._batch_size,
            show_progress_bar=False,
        )
        # 按分数降序；分数相同时按原始下标升序，保证结果可复现
        order = sorted(
            range(len(documents)),
            key=lambda i: (-float(scores[i]), i),
        )
        if top_k is not None: order = order[:top_k]
        return [(i, float(scores[i])) for i in order]


class NoOpReranker:
    """
    空重排器：保持原始顺序，分数沿用直传值

    存在的意义是让 ``RERANK_ENABLED=false`` 时上层代码无需任何分支判断
    检索链路依旧走同一套流程，只是精排阶段不改变顺序
    这让"开启/关闭重排"的对比实验变得非常简单，也是消融验证的基础
    """

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
    ) -> list[tuple[int, float]]:
        """按原始顺序返回，分数统一给 0（调用方会保留融合分数）"""
        indices = list(range(len(documents)))
        if top_k is not None:
            indices = indices[:top_k]
        return [(i, 0.0) for i in indices]


def build_reranker(
    enabled: bool,
    model_name: str,
    offline_first: bool = True,
    allow_download: bool = True,
    cache_folder: str | None = None,
) -> Reranker:
    """
    按配置构造重排器

    Args:
        enabled: 是否启用真实重排。
        model_name: 交叉编码器模型名（enabled 为 False 时忽略）
        offline_first: 优先从本地缓存加载，避免每次启动联网校验
        allow_download: 本地缓存缺失时是否允许联网下载
        cache_folder: HF 缓存目录；None 表示使用默认位置

    Returns:
        CrossEncoderReranker 或 NoOpReranker。
    """
    if not enabled:
        logger.info("重排已关闭（RERANK_ENABLED=false），使用 NoOpReranker")
        return NoOpReranker()

    logger.info(
        "重排已启用，模型=%s，离线优先=%s，允许下载=%s",
        model_name,
        offline_first,
        allow_download,
    )
    return CrossEncoderReranker(
        model_name=model_name,
        offline_first=offline_first,
        allow_download=allow_download,
        cache_folder=cache_folder,
    )
