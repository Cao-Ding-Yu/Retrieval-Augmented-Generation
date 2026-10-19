"""本地模型快照目录解析的测试（rerank.resolve_snapshot_dir）。

## 为什么这块需要专门的测试

这是"模型明明在本地，却报 it couldn't find them in the cached files"这一类
问题的定位关键。解析逻辑要自己按 HuggingFace 的缓存布局找目录，涉及
refs / snapshots / commit 哈希三层结构，任一环节判断错误都会退化成
"被迫联网"。而它又是纯函数，可以不装模型、不连网络就完整验证。

用 tmp_path 手工搭出与真实缓存一致的目录结构，逐种情况断言。
"""

from __future__ import annotations

from pathlib import Path

from rag_app.retrieval.rerank import (
    describe_hf_env,
    hf_cache_root,
    resolve_snapshot_dir,
)

ORG = "BAAI"
NAME = "bge-reranker-v2-m3"
REPO_DIR_NAME = f"models--{ORG}--{NAME}"
COMMIT = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"

# 一个可用的 CrossEncoder 缓存至少要有这两样
REQUIRED_FILES = ("config.json", "model.safetensors")


def build_cache(
    root: Path,
    commit: str = COMMIT,
    refs_commit: str | None = None,
    files: tuple[str, ...] = REQUIRED_FILES,
    with_refs: bool = True,
) -> Path:
    """在 root 下搭出一份模仿 HF 缓存的目录结构。

    Args:
        root: 充当 hub 缓存根的目录。
        commit: 快照目录名。
        refs_commit: ``refs/main`` 的内容；None 表示写入 commit。
            传入不存在的哈希可模拟 refs 与快照不一致。
        files: 快照目录内放置的文件。
        with_refs: 是否创建 refs/main。

    Returns:
        快照目录路径。
    """
    repo = root / REPO_DIR_NAME
    snapshot = repo / "snapshots" / commit
    snapshot.mkdir(parents=True)

    for filename in files:
        (snapshot / filename).write_text("{}", encoding="utf-8")

    if with_refs:
        refs = repo / "refs"
        refs.mkdir(exist_ok=True)
        (refs / "main").write_text(refs_commit or commit, encoding="utf-8")

    return snapshot


class TestResolveSnapshotDir:
    """正常与异常情况下的目录解析。"""

    def test_resolves_snapshot_pointed_by_refs(self, tmp_path):
        """refs/main 指向哪个 commit，就应解析到哪个快照目录。"""
        snapshot = build_cache(tmp_path)

        result = resolve_snapshot_dir(f"{ORG}/{NAME}", cache_root=tmp_path)

        assert result == snapshot
        assert result is not None and result.is_dir()

    def test_snapshot_contains_required_files(self, tmp_path):
        """解析出的目录必须真的能用于加载（含配置与权重）。"""
        resolve_snapshot_dir(f"{ORG}/{NAME}", cache_root=tmp_path)

        result = resolve_snapshot_dir(f"{ORG}/{NAME}", cache_root=tmp_path)
        assert result is not None
        for filename in REQUIRED_FILES:
            assert (result / filename).is_file(), f"快照缺少 {filename}"

    def test_falls_back_to_newest_when_refs_missing(self, tmp_path):
        """refs/main 缺失时应退化为取最新快照，而不是直接判定无缓存。"""
        snapshot = build_cache(tmp_path, with_refs=False)

        result = resolve_snapshot_dir(f"{ORG}/{NAME}", cache_root=tmp_path)

        assert result == snapshot

    def test_falls_back_to_newest_when_refs_points_to_unknown_commit(self, tmp_path):
        """refs 指向一个不存在的 commit 时，也应退化为取现有快照。

        这种情况在缓存被手工修补过、或下载中断后残留 refs 时会出现。
        能救回来就别报错 —— 报错的后果是白白去联网。
        """
        snapshot = build_cache(tmp_path, refs_commit="deadbeef" * 5)

        result = resolve_snapshot_dir(f"{ORG}/{NAME}", cache_root=tmp_path)

        assert result == snapshot

    def test_picks_newest_among_multiple_snapshots(self, tmp_path):
        """存在多个版本快照时应取最新的一个。"""
        build_cache(tmp_path, commit="a" * 40, with_refs=False)
        newest = build_cache(tmp_path, commit="b" * 40, with_refs=False)

        # 显式把 newest 的 mtime 推后，避免依赖创建顺序的偶然性
        import os
        import time

        future = time.time() + 10
        os.utime(newest, (future, future))

        result = resolve_snapshot_dir(f"{ORG}/{NAME}", cache_root=tmp_path)

        assert result == newest

    def test_prefers_refs_over_newest(self, tmp_path):
        """refs 指向旧版本时，应以 refs 为准而不是取最新。

        refs 才是 HuggingFace 的权威指向；若忽略它而取"最新"，
        可能加载到与配置不符的版本。
        """
        older = build_cache(tmp_path, commit="a" * 40)
        newer = build_cache(tmp_path, commit="b" * 40, with_refs=False)
        # 把 older 设成「明明是旧版本但 mtime 更新」来制造冲突
        import os
        import time

        future = time.time() + 10
        os.utime(older, (future, future))

        # refs/main 仍指向 a*40
        result = resolve_snapshot_dir(f"{ORG}/{NAME}", cache_root=tmp_path)

        assert result == older
        assert result != newer


class TestResolveSnapshotDirEdgeCases:
    """边界：不存在、非仓库名、空目录。"""

    def test_returns_none_when_repo_dir_absent(self, tmp_path):
        """完全没有该模型的缓存时应返回 None，交给上层回退。"""
        assert resolve_snapshot_dir(f"{ORG}/{NAME}", cache_root=tmp_path) is None

    def test_returns_none_when_snapshots_dir_empty(self, tmp_path):
        """只有空 snapshots 目录时也算未命中。"""
        (tmp_path / REPO_DIR_NAME / "snapshots").mkdir(parents=True)

        assert resolve_snapshot_dir(f"{ORG}/{NAME}", cache_root=tmp_path) is None

    def test_returns_none_for_non_repo_name(self, tmp_path):
        """不含 "/" 的输入不是仓库名（可能是本地路径），不应在此处理。"""
        assert resolve_snapshot_dir("local-model-dir", cache_root=tmp_path) is None
        assert resolve_snapshot_dir("C:/models/reranker", cache_root=tmp_path) is None

    def test_other_model_cache_is_ignored(self, tmp_path):
        """缓存里只有别的模型时，不应误命中。"""
        build_cache(tmp_path)  # 造的是 BAAI/bge-reranker-v2-m3

        assert resolve_snapshot_dir("BAAI/bge-reranker-base", cache_root=tmp_path) is None

    def test_case_sensitive_repo_name(self, tmp_path):
        """仓库名大小写不匹配时不应命中（HF 缓存目录名是大小写敏感的）。"""
        build_cache(tmp_path)

        assert resolve_snapshot_dir("baai/bge-reranker-v2-m3", cache_root=tmp_path) is None


class TestHfEnvDiagnostics:
    """排障辅助函数。"""

    def test_cache_root_returns_path(self):
        """应返回一个 Path，而不是字符串或 None。"""
        root = hf_cache_root()

        assert isinstance(root, Path)
        assert str(root)

    def test_describe_hf_env_lists_all_relevant_vars(self):
        """诊断串必须包含所有可能影响缓存位置的变量名。

        少列一个变量，"配了没用"这类问题就会查不出来。
        """
        description = describe_hf_env()

        for key in (
            "HF_HOME",
            "HF_HUB_CACHE",
            "HUGGINGFACE_HUB_CACHE",
            "HF_HUB_OFFLINE",
            "TRANSFORMERS_OFFLINE",
        ):
            assert key in description, f"诊断信息里缺少 {key}"

    def test_describe_hf_env_marks_unset_vars(self, monkeypatch):
        """未设置的变量应显式标注，而不是留空造成歧义。"""
        for key in ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
            monkeypatch.delenv(key, raising=False)

        description = describe_hf_env()

        assert "<未设置>" in description
