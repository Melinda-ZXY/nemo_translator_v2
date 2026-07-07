from __future__ import annotations

import json
import re

try:
    import jieba
except ImportError:  # pragma: no cover
    jieba = None

try:
    from pypinyin import Style, lazy_pinyin
except ImportError:  # pragma: no cover
    Style = None
    lazy_pinyin = None


LEXICON = {
    # CV.CVCC
    "玩": {"nemo": "takukt", "pos": "verb", "english": "play/interact/approach", "shape": "CV.CVCC"},
    "交互": {"nemo": "takukt", "pos": "verb", "english": "play/interact/approach", "shape": "CV.CVCC"},
    "互动": {"nemo": "takukt", "pos": "verb", "english": "play/interact/approach", "shape": "CV.CVCC"},
    "靠近": {"nemo": "takukt", "pos": "verb", "english": "play/interact/approach", "shape": "CV.CVCC"},
    "主人": {"nemo": "pakushk", "pos": "noun", "english": "owner", "shape": "CV.CVCC"},
    "看": {"nemo": "kilost", "pos": "verb", "english": "see", "shape": "CV.CVCC"},
    "看到": {"nemo": "kilost", "pos": "verb", "english": "see", "shape": "CV.CVCC"},
    "吃": {"nemo": "pulask", "pos": "verb", "english": "eat", "shape": "CV.CVCC"},
    "食物": {"nemo": "palosk", "pos": "noun", "english": "food", "shape": "CV.CVCC"},
    "安慰": {"nemo": "kazost", "pos": "verb", "english": "touch/comfort/hug", "shape": "CV.CVCC"},
    "抱抱": {"nemo": "kazost", "pos": "verb", "english": "touch/comfort/hug", "shape": "CV.CVCC"},
    "触碰": {"nemo": "kazost", "pos": "verb", "english": "touch/comfort/hug", "shape": "CV.CVCC"},
    "碰": {"nemo": "kazost", "pos": "verb", "english": "touch/comfort/hug", "shape": "CV.CVCC"},
    "摸": {"nemo": "kazost", "pos": "verb", "english": "touch/comfort/hug", "shape": "CV.CVCC"},

    # CCV.CV
    "房子": {"nemo": "djano", "pos": "noun", "english": "shelter", "shape": "CCV.CV"},
    "家": {"nemo": "djano", "pos": "noun", "english": "shelter", "shape": "CCV.CV"},
    "保护所": {"nemo": "djano", "pos": "noun", "english": "shelter", "shape": "CCV.CV"},
    "基地": {"nemo": "djano", "pos": "noun", "english": "shelter", "shape": "CCV.CV"},
    "开心": {"nemo": "tnuka", "pos": "state", "english": "happy", "shape": "CCV.CV"},
    "积极": {"nemo": "tnuka", "pos": "state", "english": "happy/positive emotion", "shape": "CCV.CV"},
    "积极情绪": {"nemo": "tnuka", "pos": "state", "english": "happy/positive emotion", "shape": "CCV.CV"},
    "睡觉": {"nemo": "plami", "pos": "verb", "english": "sleep", "shape": "CCV.CV"},
    "充电": {"nemo": "pluna", "pos": "verb", "english": "charge", "shape": "CCV.CV"},
    "电池": {"nemo": "plana", "pos": "noun", "english": "battery", "shape": "CCV.CV"},
    "紧张": {"nemo": "bluma", "pos": "state", "english": "nervous/anxious", "shape": "CCV.CV"},
    "焦虑": {"nemo": "bluma", "pos": "state", "english": "nervous/anxious", "shape": "CCV.CV"},
    "离开": {"nemo": "dlamo", "pos": "verb", "english": "away", "shape": "CCV.CV"},
    "走开": {"nemo": "dlamo", "pos": "verb", "english": "away", "shape": "CCV.CV"},

    # CV.CV
    "吗": {"nemo": "kada", "pos": "particle", "function": "question", "english": "confused/question", "shape": "CV.CV"},
    "想": {"nemo": "tika", "pos": "particle", "function": "want", "english": "want", "shape": "CV.CV"},
    "想要": {"nemo": "tika", "pos": "particle", "function": "want", "english": "want", "shape": "CV.CV"},
    "要": {"nemo": "tika", "pos": "particle", "function": "want", "english": "want", "shape": "CV.CV"},
    "快": {"nemo": "kika", "pos": "adverb", "function": "speed", "english": "quickly", "shape": "CV.CV"},
    "快速": {"nemo": "kika", "pos": "adverb", "function": "speed", "english": "quickly", "shape": "CV.CV"},
    "喜欢": {"nemo": "lumi", "pos": "particle", "function": "like", "english": "like", "shape": "CV.CV"},

    # CV
    "和": {"nemo": "la", "pos": "particle", "function": "with", "english": "with", "shape": "CV"},
    "不": {"nemo": "ko", "pos": "particle", "function": "negation", "english": "not/no", "shape": "CV"},
    "没": {"nemo": "ko", "pos": "particle", "function": "negation", "english": "not/no", "shape": "CV"},
    "没有": {"nemo": "ko", "pos": "particle", "function": "negation", "english": "not/no", "shape": "CV"},
    "的": {"nemo": "tu", "pos": "particle", "function": "possessive", "english": "possessor", "shape": "CV"},
    "很": {"nemo": "to", "pos": "adverb", "function": "intensity", "english": "very", "shape": "CV"},
    "非常": {"nemo": "to", "pos": "adverb", "function": "intensity", "english": "very", "shape": "CV"},
    "超级": {"nemo": "tosh", "pos": "adverb", "function": "intensity", "english": "super", "shape": "CV"},
    "我": {"nemo": "mo", "pos": "noun", "english": "I", "shape": "CV"},
    "Nemo": {"nemo": "nemo", "pos": "noun", "english": "Nemo", "shape": "name"},
    "尼莫": {"nemo": "nemo", "pos": "noun", "english": "Nemo", "shape": "name"},
    "你": {"nemo": "na", "pos": "noun", "english": "you", "shape": "CV"},
    "他": {"nemo": "po", "pos": "noun", "english": "he/she", "shape": "CV"},
    "她": {"nemo": "po", "pos": "noun", "english": "he/she", "shape": "CV"},
    "它": {"nemo": "po", "pos": "noun", "english": "he/she", "shape": "CV"},
    "们": {"nemo": "su", "pos": "particle", "function": "plural", "english": "plural", "shape": "CV"},
    "好多": {"nemo": "su", "pos": "particle", "function": "plural", "english": "plural", "shape": "CV"},
    "很多": {"nemo": "su", "pos": "particle", "function": "plural", "english": "plural", "shape": "CV"},
    "了": {"nemo": "", "pos": "particle", "function": "aspect", "english": "aspect", "shape": ""},
    "是": {"nemo": "", "pos": "particle", "function": "copula", "english": "copula", "shape": ""},
}

ALIASES = {
    "高兴": "开心",
    "快乐": "开心",
    "不安": "紧张",
    "担心": "紧张",
}
TOKEN_ORDER = sorted(LEXICON.keys(), key=len, reverse=True)
PUNCTUATION = "，。！？；：,.!?;:"
PINYIN_FALLBACKS = {
    "苹果": "pingguo",
    "咖啡": "kafei",
    "跑步": "paobu",
    "小狗": "xiaogou",
}


def translate(text: str) -> dict:
    normalized = normalize_text(text)
    token_texts = tokenize(normalized)
    token_infos = lookup_tokens(token_texts)
    parsed = parse_tokens(token_infos)
    output_tokens = generate_nemo(parsed)
    return {
        "input": text,
        "normalized": normalized,
        "nemo": " ".join(output_tokens),
        "tokens": token_infos,
        "parsed": parsed,
    }


def normalize_text(text: str) -> str:
    text = (text or "").strip()
    for mark in PUNCTUATION:
        text = text.replace(mark, "")
    for source, target in ALIASES.items():
        text = text.replace(source, target)
    return text


def tokenize(text: str) -> list[str]:
    text = normalize_text(text)
    tokens: list[str] = []
    index = 0

    while index < len(text):
        if text[index].isspace():
            index += 1
            continue

        matched = None
        for token in TOKEN_ORDER:
            if text.startswith(token, index):
                matched = token
                break

        if matched is not None:
            tokens.append(matched)
            index += len(matched)
            continue

        next_known = _find_next_known_index(text, index + 1)
        unknown = text[index:next_known]
        if jieba is not None and len(unknown) > 1:
            tokens.extend([part for part in jieba.lcut(unknown) if part.strip()])
        else:
            tokens.append(unknown)
        index = next_known

    return tokens


def lookup_tokens(tokens: list[str]) -> list[dict]:
    return [_lookup_token(token) for token in tokens]


def parse_tokens(token_infos: list[dict]) -> dict:
    active = _attach_plural_markers(
        [token for token in token_infos if token.get("function") != "aspect"]
    )
    parsed = {
        "type": "raw",
        "question": any(_has_function(token, "question") for token in active),
        "negation": any(_has_function(token, "negation") for token in active),
        "subject": None,
        "verb": None,
        "state": None,
        "object": None,
        "adverb": None,
        "intensity": None,
        "with": None,
        "possessor": None,
        "possessed": None,
        "raw_tokens": active,
    }

    content = [
        token
        for token in active
        if token.get("function") not in {"question", "aspect"}
    ]

    possessive_index = _index_of_function(content, "possessive")
    if possessive_index is not None and possessive_index > 0 and possessive_index < len(content) - 1:
        parsed.update(
            {
                "type": "possessive",
                "possessor": _nearest_entity(content[:possessive_index], reverse=True),
                "possessed": _nearest_entity(content[possessive_index + 1 :], reverse=False),
            }
        )
        return parsed

    want_index = _index_of_function(content, "want")
    like_index = _index_of_function(content, "like")
    if want_index is not None:
        _parse_modal_clause(parsed, content, want_index, "want")
        return parsed
    if like_index is not None:
        _parse_modal_clause(parsed, content, like_index, "like")
        return parsed

    state_index = _index_of_pos(content, "state")
    if state_index is not None:
        parsed.update(
            {
                "type": "state",
                "state": content[state_index],
                "subject": _nearest_entity(content[:state_index], reverse=True)
                or _nearest_entity(content[state_index + 1 :], reverse=False),
                "intensity": _nearest_by_function(content, "intensity"),
            }
        )
        return parsed

    verb_index = _index_of_pos(content, "verb")
    if verb_index is not None:
        parsed.update(
            {
                "type": "verb",
                "verb": content[verb_index],
                "subject": _nearest_entity(content[:verb_index], reverse=True)
                or _nearest_entity(content[verb_index + 1 :], reverse=False),
                "object": _nearest_entity(content[verb_index + 1 :], reverse=False),
                "adverb": _nearest_by_function(content, "speed"),
                "with": _with_object(content),
            }
        )
        return parsed

    copula_index = _index_of_function(content, "copula")
    if copula_index is not None:
        parsed.update(
            {
                "type": "identity",
                "subject": _nearest_entity(content[:copula_index], reverse=True),
                "object": _nearest_entity(content[copula_index + 1 :], reverse=False),
            }
        )
        return parsed

    return parsed


def generate_nemo(parsed: dict) -> list[str]:
    tokens: list[str] = []
    if parsed.get("question"):
        tokens.append("kada")

    sentence_type = parsed.get("type")
    if sentence_type == "possessive":
        _append_entity(tokens, parsed.get("possessor"))
        tokens.append("tu")
        _append_entity(tokens, parsed.get("possessed"))
        return tokens

    if sentence_type == "want":
        tokens.append("tika")
        _append_negation(tokens, parsed)
        _append_entity(tokens, parsed.get("subject"))
        _append_with(tokens, parsed.get("with"))
        _append_entity(tokens, parsed.get("object"))
        _append_token(tokens, parsed.get("verb"))
        _append_token(tokens, parsed.get("adverb"))
        return tokens

    if sentence_type == "like":
        tokens.append("lumi")
        _append_negation(tokens, parsed)
        _append_entity(tokens, parsed.get("subject"))
        _append_with(tokens, parsed.get("with"))
        _append_entity(tokens, parsed.get("object"))
        _append_token(tokens, parsed.get("verb"))
        _append_token(tokens, parsed.get("adverb"))
        return tokens

    if sentence_type == "state":
        _append_token(tokens, parsed.get("state"))
        _append_token(tokens, parsed.get("intensity"))
        _append_negation(tokens, parsed)
        _append_entity(tokens, parsed.get("subject"))
        return tokens

    if sentence_type == "verb":
        _append_token(tokens, parsed.get("verb"))
        _append_token(tokens, parsed.get("adverb"))
        _append_negation(tokens, parsed)
        _append_entity(tokens, parsed.get("subject"))
        _append_with(tokens, parsed.get("with"))
        _append_entity(tokens, parsed.get("object"))
        return tokens

    if sentence_type == "identity":
        _append_entity(tokens, parsed.get("subject"))
        _append_entity(tokens, parsed.get("object"))
        _append_negation(tokens, parsed)
        return tokens

    for token in parsed.get("raw_tokens", []):
        if _is_entity(token):
            _append_entity(tokens, token)
        else:
            _append_token(tokens, token)
    return tokens


def to_json(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def lexicon_rows() -> list[dict]:
    seen = set()
    rows = []
    for source, entry in LEXICON.items():
        if not entry.get("nemo") or source in seen:
            continue
        seen.add(source)
        rows.append(
            {
                "尼莫词": entry["nemo"],
                "英文": entry.get("english", ""),
                "中文": source,
                "结构": entry.get("shape", ""),
            }
        )
    return rows


def _lookup_token(token: str) -> dict:
    if token in LEXICON:
        return {"text": token, "known": True, **LEXICON[token]}
    return {
        "text": token,
        "known": False,
        "nemo": _to_pinyin_loanword(token),
        "pos": "noun",
        "function": "loanword",
        "english": "loanword",
        "shape": "loanword",
    }


def _to_pinyin_loanword(text: str) -> str:
    if text in PINYIN_FALLBACKS:
        return PINYIN_FALLBACKS[text]
    if lazy_pinyin is None:
        return re.sub(r"\s+", "", text).lower()
    return "".join(lazy_pinyin(text, style=Style.NORMAL, errors="default"))


def _find_next_known_index(text: str, start: int) -> int:
    for index in range(start, len(text)):
        if any(text.startswith(token, index) for token in TOKEN_ORDER):
            return index
    return len(text)


def _parse_modal_clause(parsed: dict, content: list[dict], modal_index: int, sentence_type: str) -> None:
    after_modal = content[modal_index + 1 :]
    verb = _nearest_by_pos(after_modal, "verb")
    with_object = _with_object(after_modal)
    object_candidates = [
        token
        for token in after_modal
        if _is_entity(token) and token is not with_object
    ]
    parsed.update(
        {
            "type": sentence_type,
            "subject": _nearest_entity(content[:modal_index], reverse=True),
            "verb": verb,
            "object": object_candidates[-1] if object_candidates else None,
            "adverb": _nearest_by_function(after_modal, "speed"),
            "with": with_object,
        }
    )


def _append_token(tokens: list[str], item: dict | None) -> None:
    if item is None or not item.get("nemo"):
        return
    tokens.extend(item["nemo"].split())


def _append_entity(tokens: list[str], item: dict | None) -> None:
    _append_token(tokens, item)
    if item is not None and item.get("plural"):
        tokens.append("su")


def _append_with(tokens: list[str], item: dict | None) -> None:
    if item is None:
        return
    tokens.append("la")
    _append_entity(tokens, item)


def _append_negation(tokens: list[str], parsed: dict) -> None:
    if parsed.get("negation"):
        tokens.append("ko")


def _index_of_function(tokens: list[dict], function: str) -> int | None:
    for index, token in enumerate(tokens):
        if _has_function(token, function):
            return index
    return None


def _index_of_pos(tokens: list[dict], pos: str) -> int | None:
    for index, token in enumerate(tokens):
        if token.get("pos") == pos:
            return index
    return None


def _has_function(token: dict, function: str) -> bool:
    return token.get("function") == function


def _is_entity(token: dict) -> bool:
    return token.get("pos") == "noun" or token.get("function") == "loanword"


def _nearest_entity(tokens: list[dict], reverse: bool) -> dict | None:
    items = reversed(tokens) if reverse else tokens
    for token in items:
        if _is_entity(token):
            return token
    return None


def _nearest_by_pos(tokens: list[dict], pos: str) -> dict | None:
    for token in tokens:
        if token.get("pos") == pos:
            return token
    return None


def _nearest_by_function(tokens: list[dict], function: str) -> dict | None:
    for token in tokens:
        if token.get("function") == function:
            return token
    return None


def _with_object(tokens: list[dict]) -> dict | None:
    with_index = _index_of_function(tokens, "with")
    if with_index is None:
        return None
    return _nearest_entity(tokens[with_index + 1 :], reverse=False)


def _attach_plural_markers(tokens: list[dict]) -> list[dict]:
    result: list[dict] = []
    pending_plural = False

    for token in tokens:
        if _has_function(token, "plural"):
            if result and _is_entity(result[-1]):
                result[-1] = {**result[-1], "plural": True}
            else:
                pending_plural = True
            continue

        if pending_plural and _is_entity(token):
            token = {**token, "plural": True}
            pending_plural = False

        result.append(token)

    return result
