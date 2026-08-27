from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np


@dataclass(slots=True)
class OcrLine:
    text: str
    confidence: float
    box: list[list[float]] = field(default_factory=list)
    searchable: bool = True
    detection_confidence: float = 0.0
    source_pass: str = "full"
    search_terms: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ObjectDetection:
    label: str
    confidence: float
    box: list[float] = field(default_factory=list)
    label_zh: str = ""

    def __post_init__(self) -> None:
        self.label = str(self.label)
        self.label_zh = str(self.label_zh)
        self.confidence = float(self.confidence)
        self.box = [float(value) for value in self.box]


@dataclass(slots=True)
class RegionEmbedding:
    id: int | None
    asset_id: int
    region_index: int
    box: list[int]
    vector: np.ndarray
    model_version: str


@dataclass(slots=True)
class VlmVerdict:
    match: int
    hard_fail: bool

    def __post_init__(self) -> None:
        if isinstance(self.match, bool) or not isinstance(self.match, int) or not 0 <= self.match <= 100:
            raise ValueError("VLM match must be an integer from 0 to 100")
        if not isinstance(self.hard_fail, bool):
            raise ValueError("VLM hard_fail must be a boolean")


@dataclass(slots=True)
class Asset:
    id: int | None
    path: str
    size: int
    modified_ns: int
    sha256: str
    width: int
    height: int
    thumbnail: str
    ocr_text: str = ""
    ocr_json: str = "[]"
    ocr_raw_text: str = ""
    ocr_diagnostics: str = "{}"
    model_version: str = ""
    status: str = "ready"
    error: str = ""


@dataclass(slots=True)
class SearchHit:
    asset: Asset
    score: float
    semantic_score: float = 0.0
    text_score: float = 0.0
    semantic_rank: int | None = None
    text_rank: int | None = None
    object_rank: int | None = None
    region_rank: int | None = None
    region_score: float = 0.0
    matched_region: list[int] | None = None
    metadata_rank: int | None = None
    query_type: str = "visual"
    bonus_score: float = 0.0
    reasons: list[str] = field(default_factory=list)
    channel_scores: dict[str, float] = field(default_factory=dict)
    deterministic_score: float = 0.0
    vlm_score: float | None = None
    vlm_hard_fail: bool = False
    rerank_source: str = "deterministic"
    protected_exact_match: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SearchDebugTrace:
    """One developer-only snapshot of the complete retrieval and rerank path."""

    generation: int
    query: str
    query_type: str
    filters: dict[str, str] = field(default_factory=dict)
    candidate_counts: dict[str, int] = field(default_factory=dict)
    effective_weights: dict[str, float] = field(default_factory=dict)
    index_coverage: dict[str, Any] = field(default_factory=dict)
    deterministic_candidates: list[SearchHit] = field(default_factory=list)
    final_results: list[SearchHit] = field(default_factory=list)
    reranker_requested: bool = False
    reranker_source: str = "deterministic"
    formula: str = "deterministic = min(1, 0.75 x weighted_RRF + bonus)"


Vector = np.ndarray
AssetSequence = Sequence[Asset]
