from __future__ import annotations

import re
import unicodedata


IDENTIFIER_RE = re.compile(
    r"(?:[A-Za-z]:[\\/]|[/\\]|\.[A-Za-z0-9]{1,8}$|[_./\\:-]|[a-z][A-Z]|[A-Za-z]+\d|\d+[A-Za-z])"
)

OBJECT_ALIASES = {
    "person": {"person", "people", "man", "woman", "boy", "girl", "人", "人物", "男人", "女人", "男生", "女生"},
    "bicycle": {"bicycle", "bike", "自行車", "腳踏車"},
    "car": {"car", "automobile", "vehicle", "汽車", "車子", "轎車"},
    "motorcycle": {"motorcycle", "motorbike", "機車", "摩托車"},
    "bus": {"bus", "公車", "巴士"},
    "train": {"train", "火車", "列車"},
    "truck": {"truck", "卡車", "貨車"},
    "boat": {"boat", "ship", "船", "船隻"},
    "bird": {"bird", "鳥"},
    "cat": {"cat", "kitten", "貓", "貓咪"},
    "dog": {"dog", "puppy", "犬", "狗", "狗狗"},
    "horse": {"horse", "馬"},
    "sheep": {"sheep", "羊", "綿羊"},
    "cow": {"cow", "cattle", "牛"},
    "elephant": {"elephant", "大象"},
    "bear": {"bear", "熊"},
    "zebra": {"zebra", "斑馬"},
    "giraffe": {"giraffe", "長頸鹿"},
    "backpack": {"backpack", "背包"},
    "umbrella": {"umbrella", "雨傘", "傘"},
    "handbag": {"handbag", "purse", "手提包", "包包"},
    "tie": {"tie", "necktie", "領帶"},
    "suitcase": {"suitcase", "luggage", "行李箱"},
    "bottle": {"bottle", "瓶子", "水瓶"},
    "cup": {"cup", "杯子", "杯"},
    "fork": {"fork", "叉子"},
    "knife": {"knife", "刀子", "刀"},
    "spoon": {"spoon", "湯匙", "匙"},
    "bowl": {"bowl", "碗"},
    "banana": {"banana", "香蕉"},
    "apple": {"apple", "蘋果"},
    "sandwich": {"sandwich", "三明治"},
    "orange": {"orange", "柳橙", "橘子"},
    "broccoli": {"broccoli", "花椰菜"},
    "carrot": {"carrot", "胡蘿蔔"},
    "pizza": {"pizza", "披薩"},
    "donut": {"donut", "甜甜圈"},
    "cake": {"cake", "蛋糕"},
    "chair": {"chair", "椅子"},
    "couch": {"couch", "sofa", "沙發"},
    "bed": {"bed", "床"},
    "table": {"table", "dining table", "桌子", "餐桌"},
    "toilet": {"toilet", "馬桶"},
    "tv": {"tv", "television", "電視"},
    "laptop": {"laptop", "notebook", "筆電", "筆記型電腦"},
    "mouse": {"mouse", "滑鼠"},
    "keyboard": {"keyboard", "鍵盤"},
    "cell phone": {"cell phone", "phone", "smartphone", "手機"},
    "book": {"book", "書", "書本"},
    "clock": {"clock", "時鐘", "鐘"},
    "scissors": {"scissors", "剪刀"},
    "multimeter": {"multimeter", "digital multimeter", "萬用電表", "三用電表"},
    "oscilloscope": {"oscilloscope", "scope", "示波器"},
    "soldering iron": {"soldering iron", "soldering station", "烙鐵", "焊台", "焊接台"},
    "gpu bracket": {"gpu bracket", "graphics card holder", "gpu support", "顯卡支架", "gpu支架"},
}


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.findall(r"[\w\-\.]+", value, flags=re.UNICODE))


def contains_cjk(value: str) -> bool:
    return any("\u3400" <= char <= "\u9fff" for char in value)


def character_ngrams(value: str, size: int = 2) -> list[str]:
    compact = normalize_text(value).replace(" ", "")
    if not compact:
        return []
    if len(compact) <= size:
        return [compact]
    return [compact[i : i + size] for i in range(len(compact) - size + 1)]


def compact_identifier(value: str) -> str:
    """Remove OCR-unstable separators while retaining identifier characters."""
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(re.findall(r"[a-z0-9\u3400-\u9fff]+", normalized))


def identifier_tokens(value: str) -> list[str]:
    """Split snake_case, CamelCase, paths, functions and extensions."""
    raw = unicodedata.normalize("NFKC", value)
    raw = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", raw)
    raw = re.sub(r"([A-Za-z])(\d)", r"\1 \2", raw)
    raw = re.sub(r"(\d)([A-Za-z])", r"\1 \2", raw)
    parts = re.findall(r"[A-Za-z0-9\u3400-\u9fff]+", raw)
    return list(dict.fromkeys(part.casefold() for part in parts if part))


def identifier_trigrams(value: str) -> list[str]:
    compact = compact_identifier(value)
    if not 4 <= len(compact) <= 64 or not re.search(r"[a-z0-9]", compact):
        return []
    return list(dict.fromkeys(compact[index : index + 3] for index in range(len(compact) - 2)))


def code_search_terms(value: str) -> list[str]:
    normalized = normalize_text(value)
    compact = compact_identifier(value)
    terms = [normalized, compact, *identifier_tokens(value), *identifier_trigrams(value)]
    return list(dict.fromkeys(term for term in terms if term))


def is_identifier_query(value: str) -> bool:
    stripped = value.strip()
    if not stripped or contains_cjk(stripped):
        return False
    all_caps_code = bool(re.fullmatch(r"[A-Z]{2,}[A-Z0-9-]*", stripped))
    return bool(IDENTIFIER_RE.search(stripped) or all_caps_code)


def classify_query(value: str) -> str:
    """Deterministic intent classification used to select ranking weights."""
    normalized = normalize_text(value)
    metadata = bool(
        re.search(r"\b(?:19|20)\d{2}[-/.年](?:0?[1-9]|1[0-2])", value)
        or re.search(r"\b(?:png|jpe?g|webp|bmp|tiff?)\b", normalized)
        or re.search(r"\b(?:folder|directory|date|before|after)\b|資料夾|日期|之前|之後", normalized)
    )
    identifier = is_identifier_query(value)
    visual = bool(normalized) and not identifier
    if metadata and (identifier or visual):
        return "mixed"
    if metadata:
        return "metadata"
    if identifier:
        return "identifier"
    return "visual" if visual else "mixed"


def metadata_filters(value: str) -> dict[str, str]:
    """Extract only explicit conditions that are safe to use as hard filters."""
    filters: dict[str, str] = {}
    extension = re.search(r"(?i)(?:^|\s|\.)(png|jpe?g|webp|bmp|tiff?)(?:$|\s)", value)
    if extension:
        suffix = extension.group(1).casefold()
        filters["extension"] = ".jpg" if suffix == "jpeg" else f".{suffix}"
    date = re.search(r"\b((?:19|20)\d{2})(?:[-/.年](0?[1-9]|1[0-2]))?(?:[-/.月](0?[1-9]|[12]\d|3[01]))?", value)
    if date:
        parts = [date.group(1)]
        if date.group(2):
            parts.append(date.group(2).zfill(2))
        if date.group(3):
            parts.append(date.group(3).zfill(2))
        filters["date_prefix"] = "-".join(parts)
    folder = re.search(r"(?i)(?:folder|directory|資料夾)\s*[:：]\s*([^\s]+)", value)
    if folder:
        filters["folder"] = normalize_text(folder.group(1))
    return filters


def object_query_labels(value: str) -> list[str]:
    normalized = normalize_text(value)
    compact = normalized.replace(" ", "")
    matched: list[str] = []
    for label, aliases in OBJECT_ALIASES.items():
        if any(normalize_text(alias) in normalized or normalize_text(alias).replace(" ", "") in compact for alias in aliases):
            matched.append(label)
    return matched


def search_document(value: str) -> str:
    """Build symmetric FTS terms for filenames and OCR text.

    SQLite's default unicode tokenizer does not provide dependable CJK word
    segmentation. Store the normalized text, individual CJK characters and
    overlapping bigrams so both exact phrases and short substrings are found.
    """
    normalized = normalize_text(value)
    terms = list(normalized.split())
    for token in normalized.split():
        if contains_cjk(token):
            terms.extend(char for char in token if contains_cjk(char))
            terms.extend(character_ngrams(token))
        elif re.search(r"[a-z0-9]", token):
            terms.extend(code_search_terms(token))
    return " ".join(dict.fromkeys(term for term in terms if term))


def fts_query(value: str) -> str:
    normalized = normalize_text(value)
    tokens: list[str] = []
    for token in normalized.split():
        if contains_cjk(token):
            compact = "".join(char for char in token if not char.isspace())
            tokens.extend(character_ngrams(compact) if len(compact) > 1 else [compact])
        elif not is_identifier_query(value):
            tokens.append(token)
    if is_identifier_query(value):
        # OR trigrams maximize recall; ranking later measures coverage and
        # awards exact compact/phrase matches.
        tokens = identifier_trigrams(value) or code_search_terms(value)
    escaped = [token.replace('"', '""') for token in tokens if token]
    operator = " OR " if is_identifier_query(value) else " AND "
    return operator.join(f'"{token}"' for token in escaped)
