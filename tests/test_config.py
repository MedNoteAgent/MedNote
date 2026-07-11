from pathlib import Path

from mednote.config import get_config


def test_get_config_loads_expected_defaults() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cfg = get_config(config_path=str(repo_root / "config.yml"))

    assert cfg.vector_store.dense_weight == 0.7
    assert cfg.vector_store.top_k_rerank == 3
    assert cfg.llm.model == "claude-sonnet-4-20250514"
