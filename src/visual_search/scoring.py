from __future__ import annotations

from .domain import SearchHit
from .text import classify_query


RRF_K = 60
WEIGHTS = {
    "identifier": {"semantic": 0.20, "region": 0.10, "text": 0.60, "object": 0.05, "metadata": 0.05},
    "visual": {"semantic": 0.35, "region": 0.35, "text": 0.10, "object": 0.15, "metadata": 0.05},
    "mixed": {"semantic": 0.30, "region": 0.25, "text": 0.25, "object": 0.15, "metadata": 0.05},
    "metadata": {"semantic": 0.15, "region": 0.10, "text": 0.15, "object": 0.20, "metadata": 0.40},
}


def rank_score(rank: int) -> float:
    return (RRF_K + 1) / (RRF_K + rank)


def effective_weights(query_type: str, available: set[str]) -> dict[str, float]:
    """Redistribute configured weights only across channels that returned candidates."""
    configured = WEIGHTS.get(query_type, WEIGHTS["visual"])
    active = {name for name in available if name in configured}
    denominator = sum(configured[name] for name in active) or 1.0
    return {name: configured[name] / denominator for name in active}


def merge_hits(
    query: str,
    semantic: list[SearchHit],
    textual: list[SearchHit],
    limit: int | None = 60,
    objects: list[SearchHit] | None = None,
    metadata: list[SearchHit] | None = None,
    regions: list[SearchHit] | None = None,
) -> list[SearchHit]:
    query_type = classify_query(query)
    objects = objects or []
    metadata = metadata or []
    regions = regions or []
    merged: dict[int, SearchHit] = {}
    for rank, hit in enumerate(semantic, 1):
        if hit.asset.id is not None:
            hit.semantic_rank = hit.semantic_rank or rank
            hit.query_type = query_type
            merged[hit.asset.id] = hit
    for rank, hit in enumerate(textual, 1):
        if hit.asset.id is None:
            continue
        current = merged.get(hit.asset.id)
        if current is None:
            current = SearchHit(hit.asset, 0.0)
            merged[hit.asset.id] = current
        current.text_score = max(current.text_score, hit.text_score)
        current.text_rank = hit.text_rank or rank
        current.bonus_score = max(current.bonus_score, hit.bonus_score)
        current.reasons = list(dict.fromkeys([*current.reasons, *hit.reasons]))

    for rank, hit in enumerate(regions, 1):
        if hit.asset.id is None:
            continue
        current = merged.get(hit.asset.id)
        if current is None:
            current = SearchHit(hit.asset, 0.0)
            merged[hit.asset.id] = current
        if current.region_rank is None or (hit.region_rank or rank) < current.region_rank:
            current.region_rank = hit.region_rank or rank
            current.region_score = hit.region_score
            current.matched_region = hit.matched_region
        current.reasons = list(dict.fromkeys([*current.reasons, *hit.reasons]))

    for channel, hits in (("object", objects), ("metadata", metadata)):
        rank_field = f"{channel}_rank"
        for rank, hit in enumerate(hits, 1):
            if hit.asset.id is None:
                continue
            current = merged.get(hit.asset.id)
            if current is None:
                current = SearchHit(hit.asset, 0.0)
                merged[hit.asset.id] = current
            setattr(current, rank_field, getattr(hit, rank_field) or rank)
            current.reasons = list(dict.fromkeys([*current.reasons, *hit.reasons]))

    available = {
        "semantic" if semantic else None,
        "region" if regions else None,
        "text" if textual else None,
        "object" if objects else None,
        "metadata" if metadata else None,
    } - {None}
    weights = effective_weights(query_type, available)
    for hit in merged.values():
        hit.query_type = query_type
        channels: dict[str, float] = {}
        if hit.semantic_rank is not None:
            channels["semantic"] = rank_score(hit.semantic_rank)
            hit.reasons.append(f"語意排名：{hit.semantic_rank}")
        if hit.region_rank is not None:
            channels["region"] = rank_score(hit.region_rank)
            hit.reasons.append(f"局部語意排名：{hit.region_rank}")
        if hit.text_rank is not None:
            channels["text"] = rank_score(hit.text_rank)
            hit.reasons.append(f"文字排名：{hit.text_rank}")
        if hit.object_rank is not None:
            channels["object"] = rank_score(hit.object_rank)
            hit.reasons.append(f"物件排名：{hit.object_rank}")
        if hit.metadata_rank is not None:
            channels["metadata"] = rank_score(hit.metadata_rank)
            hit.reasons.append(f"資料條件排名：{hit.metadata_rank}")
        hit.channel_scores = channels
        weighted_rrf = sum(weights.get(name, 0.0) * value for name, value in channels.items())
        hit.score = min(1.0, 0.75 * weighted_rrf + hit.bonus_score)
        hit.deterministic_score = hit.score
        hit.protected_exact_match = any(
            reason.startswith(("檔名完整命中", "OCR 完整命中", "程式碼識別字匹配"))
            for reason in hit.reasons
        )
        hit.reasons = list(dict.fromkeys(hit.reasons))
    ordered = sorted(merged.values(), key=lambda item: item.score, reverse=True)
    return ordered if limit is None else ordered[:max(0, int(limit))]
