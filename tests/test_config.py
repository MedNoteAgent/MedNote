from pathlib import Path

from mednote.config import get_config


def test_get_config_loads_expected_defaults() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cfg = get_config(config_path=str(repo_root / "config.yml"))

    assert cfg.vector_store.dense_weight == 0.7
    assert cfg.vector_store.top_k_rerank == 3
    # Provider/model are deploy-time choices; assert shape, not a pinned name.
    assert cfg.llm.provider in {"anthropic", "openai", "google"}
    assert cfg.llm.model

    # ETL settings (Task 5): raw CMS XML lives in data/corpus/.
    assert cfg.paths.icd10_source_dir == "data/corpus"
    assert cfg.paths.icd10_tabular_path.endswith("icd10cm_tabular_2026.xml")
    assert cfg.paths.icd10_index_path.endswith("icd10cm_index_2026.xml")
    assert cfg.etl.max_index_synonyms == 10

    # Indexing settings (Task 6): guidelines corpus + embedded Qdrant store.
    assert cfg.paths.guidelines_path.endswith("clinical_guidelines.md")
    assert cfg.vector_store.local_path == "data/qdrant_data"
    assert cfg.vector_store.collection_name == "icd10_codes"
