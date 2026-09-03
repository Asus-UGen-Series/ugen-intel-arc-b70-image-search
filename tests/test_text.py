from visual_search.text import (
    character_ngrams,
    classify_query,
    code_search_terms,
    compact_identifier,
    fts_query,
    identifier_trigrams,
    metadata_filters,
    normalize_text,
    search_document,
)


def test_normalize_text_full_width_and_case():
    assert normalize_text("ＡＳＵＳ  UX3405") == "asus ux3405"


def test_chinese_bigrams():
    assert character_ngrams("紅色筆電") == ["紅色", "色筆", "筆電"]
    assert '"紅色"' in fts_query("紅色筆電")
    assert fts_query("紅色筆電") == '"紅色" AND "色筆" AND "筆電"'


def test_search_document_indexes_cjk_characters_and_bigrams():
    document = search_document("這是紅色筆電")
    assert "色筆" in document.split()
    assert "筆" in document.split()


def test_code_identifiers_generate_symmetric_compact_and_trigrams():
    assert compact_identifier("minimum_confidence") == "minimumconfidence"
    terms = code_search_terms("PaddleOcrEngine")
    assert "paddleocrengine" in terms
    assert "pad" in identifier_trigrams("PaddleOcrEngine")
    expression = fts_query("minimum_confidence")
    assert '\"min\"' in expression and " OR " in expression


def test_query_intent_classification():
    assert classify_query("minimum_confidence") == "identifier"
    assert classify_query("red laptop on a desk") == "visual"
    assert classify_query("dog") == "visual"
    assert classify_query("laptop") == "visual"
    assert classify_query("2026-08 screenshot.png") == "mixed"
    assert metadata_filters("2026-08 screenshot.png") == {
        "extension": ".png", "date_prefix": "2026-08",
    }
