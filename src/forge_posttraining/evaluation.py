"""Frozen official answer extraction and normalization, with no benchmark data."""

from __future__ import annotations
import re
import math
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

FAMILY_TO_ANSWER_TYPE = {
    "feature_source_view": "view",
    "feature_to_same_geometry_dimension": "diameter",
    "balloon_bom_component": "component",
    "bom_quantity": "quantity",
    "cross_feature_dimension_comparison": "feature",
    "feature_owner_balloon_bom_component": "component",
    "feature_owner_balloon_bom_quantity": "quantity",
    "hard_compare_then_component": "component",
    "hard_compare_then_quantity": "quantity",
    "hard_dual_traverse_quantity_compare": "feature",
    "hard_filter_quantity_then_diameter": "feature",
}
NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}
COMMON_ANSWER_PATTERNS = [
    re.compile("(?im)^\\s*answer\\s*:\\s*(.+?)\\s*$"),
    re.compile("(?im)^\\s*final answer\\s*:\\s*(.+?)\\s*$"),
    re.compile("(?im)^\\s*the answer is\\s+(.+?)\\s*[\\.\\n]?$"),
]


def nfkc(text: Any) -> str:
    return unicodedata.normalize("NFKC", str(text or "")).strip()


def strip_outer_punct(text: str) -> str:
    text = text.strip()
    text = re.sub("^[\\s`*_#'\\\"\\[\\](){}<>:;,.!?]+", "", text)
    text = re.sub("[\\s`*_#'\\\"\\[\\](){}<>:;,.!?]+$", "", text)
    return text.strip()


def normalize_generic(text: Any) -> str:
    s = nfkc(text).casefold()
    s = strip_outer_punct(s)
    s = re.sub("\\s+", " ", s)
    return s.strip()


def normalize_component(text: Any) -> str:
    s = nfkc(text).casefold()
    s = strip_outer_punct(s)
    s = re.sub("^(?:the\\s+)?(?:component|part)\\s+(?:is|identified\\s+is)\\s+", "", s, flags=re.I)
    s = re.sub("[-_/]+", " ", s)
    s = re.sub("[^\\w.\\s&()+]+", " ", s, flags=re.UNICODE)
    s = re.sub("\\s+", " ", s).strip()
    return s


def parse_quantity_value(text: Any) -> Optional[int]:
    s = nfkc(text).casefold()
    m = re.search("(?<![\\d.])-?\\d+(?![\\d.])", s)
    if m:
        try:
            return int(m.group(0))
        except ValueError:
            pass
    for word, value in NUMBER_WORDS.items():
        if re.search(f"\\b{re.escape(word)}\\b", s):
            return value
    return None


def parse_diameter_value_mm(text: Any) -> Optional[float]:
    s = nfkc(text).casefold().replace("⌀", "ø")
    patterns = [
        "[øφ]\\s*([+-]?\\d+(?:\\.\\d+)?)",
        "([+-]?\\d+(?:\\.\\d+)?)\\s*mm\\b",
        "(?:diameter|dia\\.?)\\s*(?:is|=|:)?\\s*([+-]?\\d+(?:\\.\\d+)?)",
    ]
    for pat in patterns:
        matches = re.findall(pat, s, flags=re.I)
        if matches:
            try:
                return float(matches[-1])
            except ValueError:
                pass
    short = strip_outer_punct(s)
    if re.fullmatch("[+-]?\\d+(?:\\.\\d+)?", short):
        return float(short)
    return None


def normalize_view(text: Any) -> str:
    s = nfkc(text).casefold()
    s = strip_outer_punct(s)
    s = re.sub("\\b(?:the\\s+)?view\\b", " ", s)
    s = re.sub("\\bdrawing\\b", " ", s)
    s = re.sub("\\s+", " ", s).strip()
    s = re.sub("\\s*-\\s*", "-", s)
    return s


def normalize_feature(text: Any) -> Optional[str]:
    s = nfkc(text).upper()
    matches = re.findall("\\bF\\d{1,3}\\b", s)
    return matches[-1] if matches else None


def extract_explicit_final(raw: str) -> Optional[str]:
    matches = re.findall("(?im)^\\s*FINAL\\s*:\\s*(.+?)\\s*$", raw)
    if matches:
        return strip_outer_punct(matches[-1])
    return None


def extract_common_answer_line(raw: str) -> Optional[str]:
    for pat in COMMON_ANSWER_PATTERNS:
        matches = pat.findall(raw)
        if matches:
            return strip_outer_punct(matches[-1])
    return None


def last_nonempty_line(raw: str) -> Optional[str]:
    lines = [strip_outer_punct(x) for x in raw.splitlines() if x.strip()]
    lines = [x for x in lines if x]
    return lines[-1] if lines else None


def extract_feature(raw: str) -> Optional[str]:
    final = extract_explicit_final(raw)
    if final:
        return normalize_feature(final)
    common = extract_common_answer_line(raw)
    if common:
        feat = normalize_feature(common)
        if feat:
            return feat
    patterns = [
        "(?i)(?:select|answer(?:\\s+is)?|therefore[,:\\s]*|thus[,:\\s]*)(F\\d{1,3})\\b",
        "(?i)\\b(F\\d{1,3})\\b\\s+(?:is|has|belongs|contains)",
    ]
    for pat in patterns:
        matches = re.findall(pat, raw)
        if matches:
            return matches[-1].upper()
    matches = re.findall("(?i)\\bF\\d{1,3}\\b", raw)
    return matches[-1].upper() if matches else None


def extract_quantity(raw: str) -> Optional[str]:
    final = extract_explicit_final(raw)
    if final:
        v = parse_quantity_value(final)
        return str(v) if v is not None else strip_outer_punct(final)
    common = extract_common_answer_line(raw)
    if common:
        v = parse_quantity_value(common)
        if v is not None:
            return str(v)
    patterns = [
        "(?i)(?:quantity|qty)\\s*(?:is|=|:)?\\s*(\\d+)\\b",
        "(?i)\\b(?:uses?|used|contains?|has|lists?)\\s+(\\d+)\\s+(?:instance|instances|component|components|part|parts)\\b",
        "(?i)\\b(\\d+)\\s+(?:instance|instances|units?|pieces?)\\b",
    ]
    for pat in patterns:
        matches = re.findall(pat, raw)
        if matches:
            return str(int(matches[-1]))
    last = last_nonempty_line(raw)
    if last:
        v = parse_quantity_value(last)
        if v is not None:
            return str(v)
    return None


def extract_diameter(raw: str) -> Optional[str]:
    final = extract_explicit_final(raw)
    if final:
        v = parse_diameter_value_mm(final)
        if v is not None:
            return format_float(v) + " mm"
        return strip_outer_punct(final)
    common = extract_common_answer_line(raw)
    if common:
        v = parse_diameter_value_mm(common)
        if v is not None:
            return format_float(v) + " mm"
    patterns = ["(?i)[øφ]\\s*([+-]?\\d+(?:\\.\\d+)?)", "(?i)([+-]?\\d+(?:\\.\\d+)?)\\s*mm\\b"]
    values: List[float] = []
    for pat in patterns:
        for m in re.findall(pat, raw):
            try:
                values.append(float(m))
            except ValueError:
                pass
    if values:
        return format_float(values[-1]) + " mm"
    last = last_nonempty_line(raw)
    if last:
        v = parse_diameter_value_mm(last)
        if v is not None:
            return format_float(v) + " mm"
    return None


def extract_view(raw: str) -> Optional[str]:
    final = extract_explicit_final(raw)
    if final:
        return strip_outer_punct(final)
    common = extract_common_answer_line(raw)
    if common:
        return strip_outer_punct(common)
    section_matches = re.findall("(?i)\\bSECTION\\s+[A-Z0-9]+\\s*-\\s*[A-Z0-9]+\\b", raw)
    if section_matches:
        return section_matches[-1]
    known = re.findall("(?i)\\b(?:FRONT|REAR|BACK|LEFT|RIGHT|TOP|BOTTOM|ISOMETRIC|ISO)\\b(?:\\s+VIEW)?", raw)
    if known:
        return known[-1]
    last = last_nonempty_line(raw)
    return last


def extract_component(raw: str) -> Optional[str]:
    final = extract_explicit_final(raw)
    if final:
        return strip_outer_punct(final)
    common = extract_common_answer_line(raw)
    if common:
        return strip_outer_punct(common)
    patterns = [
        "(?im)^\\s*(?:the\\s+)?component\\s+(?:is|identified\\s+is)\\s+(.+?)\\s*[\\.\\n]?$",
        "(?im)^\\s*(?:balloon\\s+\\d+\\s+)?identifies\\s+(.+?)\\s*[\\.\\n]?$",
        "(?i)(?:therefore|thus|so)[,:\\s]+(?:the\\s+)?(?:component|part)\\s+(?:is\\s+)?(.+?)(?:\\.|\\n|$)",
        "(?i)\\bBOM\\s+item\\s+\\d+\\s+identifies\\s+(.+?)(?:\\.|\\n|$)",
    ]
    for pat in patterns:
        matches = re.findall(pat, raw)
        if matches:
            return strip_outer_punct(matches[-1])
    return last_nonempty_line(raw)


def answer_type_for_family(family: str) -> str:
    if family not in FAMILY_TO_ANSWER_TYPE:
        raise KeyError(
            f"Unknown reasoning_family={family!r}. Add it to FAMILY_TO_ANSWER_TYPE before evaluating."
        )
    return FAMILY_TO_ANSWER_TYPE[family]


def extract_prediction(raw: str, answer_type: str) -> Tuple[Optional[str], str]:
    if not raw or not raw.strip():
        return (None, "empty_response")
    final = extract_explicit_final(raw)
    if final is not None:
        if answer_type == "feature":
            return (normalize_feature(final) or final, "explicit_final")
        if answer_type == "quantity":
            v = parse_quantity_value(final)
            return (str(v) if v is not None else final, "explicit_final")
        if answer_type == "diameter":
            v = parse_diameter_value_mm(final)
            return (format_float(v) + " mm" if v is not None else final, "explicit_final")
        return (final, "explicit_final")
    extractor = {
        "feature": extract_feature,
        "quantity": extract_quantity,
        "diameter": extract_diameter,
        "view": extract_view,
        "component": extract_component,
    }[answer_type]
    pred = extractor(raw)
    return (pred, f"{answer_type}_fallback" if pred is not None else "parse_failed")


def format_float(v: float) -> str:
    if math.isclose(v, round(v), abs_tol=1e-12):
        return str(int(round(v)))
    return f"{v:.12f}".rstrip("0").rstrip(".")


def candidate_gold_strings(row: Dict[str, Any]) -> List[str]:
    vals = []
    for key in ("canonical_answer", "display_answer"):
        value = row.get(key)
        if value is not None and str(value).strip():
            vals.append(str(value).strip())
    out = []
    seen = set()
    for x in vals:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def score_prediction(
    prediction: Optional[str], row: Dict[str, Any], answer_type: str, diameter_abs_tol: float
) -> Tuple[bool, Optional[str], Optional[str]]:
    golds = candidate_gold_strings(row)
    if not golds:
        raise RuntimeError(f"{row.get('global_qa_id')}: missing canonical/display answer")
    if prediction is None:
        return (False, None, normalize_generic(golds[0]))
    if answer_type == "feature":
        p = normalize_feature(prediction)
        gs = [normalize_feature(g) for g in golds]
        gs = [g for g in gs if g]
        return (p in gs if p else False, p, gs[0] if gs else None)
    if answer_type == "quantity":
        p = parse_quantity_value(prediction)
        gs = [parse_quantity_value(g) for g in golds]
        gs = [g for g in gs if g is not None]
        ok = p is not None and p in gs
        return (ok, str(p) if p is not None else None, str(gs[0]) if gs else None)
    if answer_type == "diameter":
        p = parse_diameter_value_mm(prediction)
        gs = [parse_diameter_value_mm(g) for g in golds]
        gs = [g for g in gs if g is not None]
        ok = p is not None and any((math.isclose(p, g, rel_tol=0.0, abs_tol=diameter_abs_tol) for g in gs))
        return (
            ok,
            format_float(p) + " mm" if p is not None else None,
            format_float(gs[0]) + " mm" if gs else None,
        )
    if answer_type == "view":
        p = normalize_view(prediction)
        gs = [normalize_view(g) for g in golds]
        return (p in gs, p, gs[0])
    if answer_type == "component":
        p = normalize_component(prediction)
        gs = [normalize_component(g) for g in golds]
        return (p in gs, p, gs[0])
    raise KeyError(answer_type)
