"""Loads and validates config.yml as the single source of truth for tunable parameters."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel


class FastLLMConfig(BaseModel):
    provider: str
    model: str
    max_tokens: int = 512


class LLMConfig(BaseModel):
    provider: str
    model: str
    temperature: float = 0.0
    max_tokens: int = 4096
    fast: FastLLMConfig


class EmbeddingsConfig(BaseModel):
    model: str
    batch_size: int = 64


class RerankerConfig(BaseModel):
    model: str


class VectorStoreConfig(BaseModel):
    local_path: str
    collection_name: str
    dense_weight: float
    sparse_weight: float
    top_k_retrieve: int
    top_k_rerank: int
    confidence_threshold: float


class CacheConfig(BaseModel):
    rag_max_size: int = 128


class EhrApiConfig(BaseModel):
    host: str
    port: int


class MemoryConfig(BaseModel):
    db_path: str


class ObservabilityConfig(BaseModel):
    trace_dir: str


class PathsConfig(BaseModel):
    icd10_source_dir: str
    icd10_tabular_path: str
    icd10_index_path: str
    icd10_processed_path: str
    transcripts_path: str
    ehr_store_path: str
    corpus_dir: str
    guidelines_path: str


class EtlConfig(BaseModel):
    max_index_synonyms: int = 10
    download_url: str


class EdgeCasesConfig(BaseModel):
    min_transcript_words: int = 10
    max_transcript_words: int = 5000


class DemoConfig(BaseModel):
    latency_budget_ms: int = 15000


class MedNoteConfig(BaseModel):
    llm: LLMConfig
    embeddings: EmbeddingsConfig
    reranker: RerankerConfig
    vector_store: VectorStoreConfig
    cache: CacheConfig
    ehr_api: EhrApiConfig
    memory: MemoryConfig
    observability: ObservabilityConfig
    paths: PathsConfig
    etl: EtlConfig
    edge_cases: EdgeCasesConfig
    demo: DemoConfig


@lru_cache(maxsize=1)
def get_config(config_path: str | None = None) -> MedNoteConfig:
    """Load config once per process and cache the parsed result."""
    path = Path(config_path or os.getenv("MEDNOTE_CONFIG_PATH", "config.yml"))
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return MedNoteConfig.model_validate(raw)
