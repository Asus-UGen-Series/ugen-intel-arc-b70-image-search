import pytest

from visual_search.domain import Asset, SearchHit
from visual_search.scoring import effective_weights, merge_hits


def test_merge_prefers_exact_text_match():
    a = Asset(1, "red-asus.jpg", 1, 1, "a", 10, 10, "a.jpg", "ASUS UX3405")
    b = Asset(2, "other.jpg", 1, 1, "b", 10, 10, "b.jpg", "")
    hits = merge_hits("UX3405", [SearchHit(b, .9, .9), SearchHit(a, .7, .7)], [SearchHit(a, 1, text_score=1)])
    assert hits[0].asset.id == 1


def test_identifier_query_prefers_text_and_keeps_text_only_candidate():
    semantic_asset = Asset(1, "photo.jpg", 1, 1, "a", 10, 10, "a.jpg")
    code_asset = Asset(2, "screen.png", 1, 1, "b", 10, 10, "b.jpg", "minimumconfidence")
    results = merge_hits(
        "minimum_confidence",
        [SearchHit(semantic_asset, 0.9, semantic_score=0.9, semantic_rank=1)],
        [SearchHit(code_asset, 0.0, text_score=1.0, text_rank=1, bonus_score=0.15,
                   reasons=["程式碼識別字匹配：minimumconfidence"])],
    )
    assert results[0].asset.id == 2
    assert results[0].query_type == "identifier"
    assert results[0].text_rank == 1


def test_visual_query_uses_semantic_rank_reason():
    asset = Asset(1, "dog.jpg", 1, 1, "a", 10, 10, "a.jpg")
    result = merge_hits("dog running in a park", [SearchHit(asset, 0.8, semantic_rank=3)], [])[0]
    assert result.query_type == "visual"
    assert "語意排名：3" in result.reasons


def test_effective_weights_redistribute_only_across_available_channels():
    weights = effective_weights("visual", {"semantic", "text"})
    assert weights == pytest.approx({"semantic": 0.35 / 0.45, "text": 0.10 / 0.45})
    assert sum(weights.values()) == pytest.approx(1.0)
