"""Unchanged answer normalization and format checks."""

from __future__ import annotations
import json
import math
import re
import unicodedata
from typing import Any

FAMILY_TO_ANSWER_TYPE = {
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


def parse_ground_truth(ground_truth: Any) -> dict:
    if isinstance(ground_truth, dict):
        return ground_truth
    if isinstance(ground_truth, str):
        try:
            x = json.loads(ground_truth)
            if isinstance(x, dict):
                return x
        except Exception:
            pass
        return {"canonical_answer": ground_truth, "answer_type": "generic"}
    return {"canonical_answer": str(ground_truth), "answer_type": "generic"}


def nfkc(x: Any) -> str:
    return unicodedata.normalize("NFKC", str(x or "")).strip()


def extract_final(response: str) -> str:
    s = nfkc(response)
    m = re.findall("(?im)^\\s*FINAL\\s*:\\s*(.+?)\\s*$", s)
    if m:
        return m[-1].strip()
    m = re.findall("(?is)<answer>\\s*(.*?)\\s*</answer>", s)
    if m:
        return m[-1].strip()
    m = re.findall("(?s)\\\\boxed\\{([^{}]*)\\}", s)
    if m:
        return m[-1].strip()
    m = re.findall("(?im)^\\s*(?:final\\s+answer|answer)\\s*:\\s*(.+?)\\s*$", s)
    if m:
        return m[-1].strip()
    lines = [x.strip() for x in s.splitlines() if x.strip()]
    return lines[-1] if lines else ""


def normalize_component(x: Any) -> str:
    s = nfkc(x).casefold()
    s = re.sub("^(?:the\\s+)?(?:component|part)\\s+(?:is|identified\\s+is)\\s+", "", s)
    s = re.sub("[-_/]+", " ", s)
    s = re.sub("[^\\w.\\s&()+]+", " ", s, flags=re.UNICODE)
    return re.sub("\\s+", " ", s).strip()


def normalize_feature(x: Any):
    m = re.findall("\\bF\\d{1,3}\\b", nfkc(x).upper())
    return m[-1] if m else None


def parse_quantity(x: Any):
    s = nfkc(x).casefold()
    m = re.search("(?<![\\d.])-?\\d+(?![\\d.])", s)
    if m:
        try:
            return int(m.group(0))
        except Exception:
            pass
    for w, v in NUMBER_WORDS.items():
        if re.search(f"\\b{re.escape(w)}\\b", s):
            return v
    return None


def parse_diameter(x: Any):
    s = nfkc(x).casefold().replace("⌀", "ø")
    for pat in [
        "[øφ]\\s*([+-]?\\d+(?:\\.\\d+)?)",
        "([+-]?\\d+(?:\\.\\d+)?)\\s*mm\\b",
        "(?:diameter|dia\\.?)\\s*(?:is|=|:)?\\s*([+-]?\\d+(?:\\.\\d+)?)",
    ]:
        ms = re.findall(pat, s, flags=re.I)
        if ms:
            try:
                return float(ms[-1])
            except Exception:
                pass
    s = s.strip(" \t\r\n`*_#'\"[](){}<>:;,.!?")
    if re.fullmatch("[+-]?\\d+(?:\\.\\d+)?", s):
        return float(s)
    return None


def normalize_view(x: Any) -> str:
    s = nfkc(x).casefold()
    s = re.sub("\\b(?:the\\s+)?view\\b", " ", s)
    s = re.sub("\\bdrawing\\b", " ", s)
    s = re.sub("\\s*-\\s*", "-", s)
    s = re.sub("[^\\w\\s-]+", " ", s)
    return re.sub("\\s+", " ", s).strip()


def answer_correct(response: str, payload: dict) -> float:
    pred = extract_final(response)
    gold = str(payload.get("canonical_answer", ""))
    typ = payload.get("answer_type") or FAMILY_TO_ANSWER_TYPE.get(payload.get("reasoning_family"), "generic")
    if typ == "component":
        return float(normalize_component(pred) == normalize_component(gold))
    if typ == "feature":
        p, g = (normalize_feature(pred), normalize_feature(gold))
        return float(p is not None and g is not None and (p == g))
    if typ == "quantity":
        p, g = (parse_quantity(pred), parse_quantity(gold))
        return float(p is not None and g is not None and (p == g))
    if typ == "diameter":
        p, g = (parse_diameter(pred), parse_diameter(gold))
        return float(p is not None and g is not None and math.isclose(p, g, rel_tol=0.0, abs_tol=1e-06))
    if typ == "view":
        return float(normalize_view(pred) == normalize_view(gold))
    return float(nfkc(pred).casefold() == nfkc(gold).casefold())


def format_reward(response: str) -> float:
    s = nfkc(response)
    finals = list(re.finditer("(?im)^\\s*FINAL\\s*:\\s*\\S.*$", s))
    if len(finals) != 1:
        return 0.0
    final = finals[0]
    think = re.search("(?is)<think>.*?</think>", s)
    if think:
        return float(final.start() >= think.end())
    close = re.search("(?is)</think>", s)
    if close:
        return float(final.start() >= close.end())
    return 0.0


def reasoning_part(response: str) -> str:
    s = nfkc(response)
    m = re.search("(?is)<think>(.*?)</think>", s)
    if m:
        return m.group(1)
    close = re.search("(?is)</think>", s)
    if close:
        return s[: close.start()]
    return re.split("(?im)^\\s*FINAL\\s*:", s, maxsplit=1)[0]
