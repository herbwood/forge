"""Dependency-limited FORGE text grammar used by the strict verifier."""

from __future__ import annotations
from typing import Any
import re

HARD_C1 = "hard_compare_then_component"
HARD_C2 = "hard_compare_then_quantity"
HARD_C3 = "hard_dual_traverse_quantity_compare"
HARD_C4 = "hard_filter_quantity_then_diameter"


def _clean(x: Any) -> str:
    s = str(x or "").strip().casefold()
    s = re.sub("[-_/]+", " ", s)
    return re.sub("\\s+", " ", s)


def _feature(x: Any) -> str:
    m = re.search("\\bF\\s*0*(\\d+)\\b", str(x or ""), flags=re.I)
    return f"F{int(m.group(1)):02d}" if m else str(x or "").strip().upper()


def _sents(text: str) -> list[str]:
    return [x.strip() for x in re.split("(?<=[.!?])\\s+|\\n+", str(text or "")) if x.strip()]


def _num_eq(a: Any, b: Any, tol: float = 1e-05) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except Exception:
        return False


def _graph(payload: dict[str, Any]) -> dict[str, Any]:
    g = payload.get("reward_graph")
    return g if isinstance(g, dict) else {}


def _family(payload: dict[str, Any]) -> str:
    return str(payload.get("reasoning_family") or "")


def _op(graph: dict[str, Any], typ: str) -> dict[str, Any] | None:
    return next((x for x in graph.get("operations") or [] if x.get("type") == typ), None)


def _facts(graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {_feature(k): v for k, v in (graph.get("facts_by_feature") or {}).items() if isinstance(v, dict)}


def _feature_bom_values(text: str, feature: str) -> list[int]:
    f = _feature(feature)
    ef = re.escape(f)
    vals: list[int] = []
    sents = _sents(text)
    for i, sent in enumerate(sents):
        pats = [
            f"(?:owner\\s+marker\\s+(?:associated\\s+with|for)\\s+)?\\b{ef}\\b[^.!?]{{0,90}}?\\bBOM\\s*(?:item)?\\s*#?\\s*(\\d+)\\b",
            f"\\bBOM\\s*(?:item)?\\s*#?\\s*(\\d+)\\b[^.!?]{{0,90}}?\\b{ef}\\b",
        ]
        for p in pats:
            vals += [int(x) for x in re.findall(p, sent, flags=re.I)]
        if i > 0 and re.search("\\b(?:its|the)\\s+(?:associated\\s+)?owner\\s+marker\\b", sent, flags=re.I):
            prev = sents[i - 1]
            if re.search(f"\\b{ef}\\b", prev, flags=re.I) and re.search(
                "\\b(?:select|selected|branch)\\b", prev, flags=re.I
            ):
                vals += [
                    int(x) for x in re.findall("\\bBOM\\s*(?:item)?\\s*#?\\s*(\\d+)\\b", sent, flags=re.I)
                ]
    return list(dict.fromkeys(vals))


def _bom_component_assertions(text: str, bom: int) -> list[str]:
    vals = []
    b = int(bom)
    for sent in _sents(text):
        m = re.search(
            f"\\bBOM\\s*(?:item)?\\s*#?\\s*{b}\\b[^.!?]{{0,55}}?(?:identifies|is|corresponds\\s+to)\\s+(?:the\\s+)?(?:component\\s+(?:as\\s+)?)?(.+?)(?:[.;]|$)",
            sent,
            flags=re.I,
        )
        if m:
            vals.append(m.group(1).strip(" ."))
    return vals


def _feature_component_assertions(text: str, feature: str) -> list[str]:
    vals = []
    f = re.escape(_feature(feature))
    pats = [
        f"(?:component\\s+containing|component\\s+associated\\s+with|component\\s+for)\\s+\\b{f}\\b\\s+(?:is|=)\\s+(.+?)(?:[.;]|$)",
        f"\\b{f}\\b[^.!?]{{0,45}}?\\bbelongs\\s+to\\s+(?:the\\s+)?component\\s+(.+?)(?:[.;]|$)",
    ]
    for sent in _sents(text):
        for p in pats:
            m = re.search(p, sent, flags=re.I)
            if m:
                vals.append(m.group(1).strip(" ."))
    return vals


def _feature_diameter_values(text: str, feature: str, component: str | None = None) -> list[float]:
    f = _feature(feature)
    ef = re.escape(f)
    vals: list[float] = []
    sents = _sents(text)
    for i, sent in enumerate(sents):
        features_in_sent = {_feature(x) for x in re.findall("\\bF\\d+\\b", sent, flags=re.I)}
        safe_patterns = [
            f"(?:displayed\\s+)?diameter\\s+of\\s+\\b{ef}\\b\\s+(?:is|=)\\s*([+-]?\\d+(?:\\.\\d+)?)\\s*mm\\b",
            f"(?:displayed\\s+)?diameter\\s+of\\s+([+-]?\\d+(?:\\.\\d+)?)\\s*mm\\s+corresponds\\s+to\\s+\\b{ef}\\b",
            f"([+-]?\\d+(?:\\.\\d+)?)\\s*mm\\s+(?:for|of)\\s+\\b{ef}\\b",
            f"\\b{ef}\\b\\s+(?:has|with)\\s+(?:a\\s+)?(?:displayed\\s+)?diameter\\s+of\\s*([+-]?\\d+(?:\\.\\d+)?)\\s*mm\\b",
            f"\\b{ef}\\b\\s+(?:is|=)\\s*([+-]?\\d+(?:\\.\\d+)?)\\s*mm\\b",
            f"\\b{ef}\\b['’]s\\s+([+-]?\\d+(?:\\.\\d+)?)\\s*mm\\s+(?:diameter|value)\\b",
        ]
        for pat in safe_patterns:
            vals += [float(x) for x in re.findall(pat, sent, flags=re.I)]
        if features_in_sent == {f}:
            broad = [
                f"\\b{ef}\\b[^.!?]{{0,60}}?(?:displayed\\s+)?diameter[^0-9]{{0,18}}([+-]?\\d+(?:\\.\\d+)?)\\s*mm\\b",
                f"(?:displayed\\s+)?diameter[^.!?]{{0,60}}?\\b{ef}\\b[^0-9]{{0,18}}([+-]?\\d+(?:\\.\\d+)?)\\s*mm\\b",
            ]
            for pat in broad:
                vals += [float(x) for x in re.findall(pat, sent, flags=re.I)]
        if i > 0 and re.search(
            "^\\s*(?:its|the\\s+feature['’]s)\\s+(?:displayed\\s+)?diameter\\s+(?:is|=)\\s*([+-]?\\d+(?:\\.\\d+)?)\\s*mm\\b",
            sent,
            flags=re.I,
        ):
            prev = sents[i - 1]
            if re.search(f"\\b{ef}\\b", prev, flags=re.I):
                m = re.search(
                    "(?:displayed\\s+)?diameter\\s+(?:is|=)\\s*([+-]?\\d+(?:\\.\\d+)?)\\s*mm\\b",
                    sent,
                    flags=re.I,
                )
                if m:
                    vals.append(float(m.group(1)))
    if component:
        ec = re.escape(str(component))
        for sent in sents:
            cpats = [
                f"(?:displayed\\s+)?diameter\\s+of\\s+{ec}\\s+(?:is|=)\\s*([+-]?\\d+(?:\\.\\d+)?)\\s*mm\\b",
                f"\\b{ec}\\b[^.!?]{{0,45}}?(?:displayed\\s+)?diameter[^0-9]{{0,15}}([+-]?\\d+(?:\\.\\d+)?)\\s*mm\\b",
            ]
            for pat in cpats:
                vals += [float(x) for x in re.findall(pat, sent, flags=re.I)]
    return list(dict.fromkeys(vals))


def _quantity_values(text: str, feature: str, component: str | None, bom_item: int | None) -> list[int]:
    vals: list[int] = []
    f = _feature(feature)
    ef = re.escape(f)
    entities = [f"\\b{ef}\\b"]
    if component:
        entities.append(re.escape(str(component)))
    if bom_item is not None:
        entities.append(f"\\bBOM\\s*(?:item)?\\s*#?\\s*{int(bom_item)}\\b")
    sents = _sents(text)
    for i, sent in enumerate(sents):
        linked = any((re.search(e, sent, flags=re.I) for e in entities))
        if (
            not linked
            and i > 0
            and re.search(
                "\\b(?:its|whose|that\\s+component(?:['’]s)?|the\\s+component(?:['’]s)?)\\b", sent, flags=re.I
            )
        ):
            linked = any((re.search(e, sents[i - 1], flags=re.I) for e in entities))
        if not linked:
            continue
        pats = [
            "\\bquantity(?:\\s+of|\\s+is|\\s*=|\\s+as)?\\s*(?:also\\s+)?(\\d+)\\b",
            "\\bquantity\\s+(?:associated\\s+with|for|of)\\s+[^.!?]{0,50}?\\s+(?:is|=)?\\s*(?:also\\s+)?(\\d+)\\b",
            "\\b(?:listed\\s+)?quantity\\s+(?:is|as|of|=)\\s*(?:also\\s+)?(\\d+)\\b",
            "\\bBOM\\s+quantity\\s+(?:is|of|=)?\\s*(?:also\\s+)?(\\d+)\\b",
            "\\b(?:has|with)\\s+(?:a\\s+)?(?:listed\\s+)?quantity\\s+(?:of\\s+)?(\\d+)\\b",
            "\\b(\\d+)\\s+instances?\\b",
            "\\bused\\s+(\\d+)\\s+(?:time|times)\\b",
        ]
        for p in pats:
            vals += [int(x) for x in re.findall(p, sent, flags=re.I)]
        vals += [
            int(x)
            for x in re.findall(
                "\\bBOM\\s+(?:lists|gives)\\s+(?:[^.!?]{0,35}?\\s+)?(?:a\\s+)?quantity\\s+(?:of|as)\\s+(?:also\\s+)?(\\d+)\\b",
                sent,
                flags=re.I,
            )
        ]
    return list(dict.fromkeys(vals))


def _state_numeric(vals: list[float | int], gold: float | int) -> str:
    if not vals:
        return "missing"
    matches = [_num_eq(v, gold) for v in vals]
    return "correct" if any(matches) else "contradiction"


def _state_text(vals: list[str], gold: str) -> str:
    if not vals:
        return "missing"
    g = _clean(gold)
    matches = []
    for v in vals:
        x = _clean(v)
        matches.append(g == x or g in x or x in g)
    return "correct" if any(matches) else "contradiction"


def _feature_states(text: str, graph: dict[str, Any]) -> dict[tuple[str, str], str]:
    out: dict[tuple[str, str], str] = {}
    facts = _facts(graph)
    for f, rec in facts.items():
        bom = rec.get("bom_item")
        comp = rec.get("component")
        qty = rec.get("quantity")
        dia = rec.get("diameter_mm")
        if bom is not None:
            out[f, "bom_item"] = _state_numeric(_feature_bom_values(text, f), int(bom))
        if comp:
            vals = []
            if bom is not None:
                vals += _bom_component_assertions(text, int(bom))
                eb = f"\\bBOM\\s*(?:item)?\\s*#?\\s*{int(bom)}\\b"
                ec = re.escape(str(comp))
                if any(
                    (
                        re.search(eb, sent, flags=re.I) and re.search(ec, sent, flags=re.I)
                        for sent in _sents(text)
                    )
                ):
                    vals.append(str(comp))
            vals += _feature_component_assertions(text, f)
            out[f, "component"] = _state_text(vals, str(comp))
        if qty is not None:
            out[f, "quantity"] = _state_numeric(
                _quantity_values(text, f, str(comp) if comp else None, int(bom) if bom is not None else None),
                int(qty),
            )
        if dia is not None:
            out[f, "diameter_mm"] = _state_numeric(
                _feature_diameter_values(text, f, str(comp) if comp else None), float(dia)
            )
    return out


def _correct(states: dict[tuple[str, str], str], f: str, slot: str) -> bool:
    return states.get((_feature(f), slot)) == "correct"


def _diameter_winner(text: str, op: dict[str, Any], facts: dict[str, dict[str, Any]]) -> str | None:
    candidates = [_feature(x) for x in op.get("candidates") or []]
    kw = "(?:larger|largest|greater|greatest|bigger|biggest|higher|highest)"
    sents = _sents(text)
    for f in candidates:
        ef = re.escape(f)
        for sent in sents:
            if (
                re.search(
                    f"\\b{ef}\\b[^.!?]{{0,90}}?{kw}[^.!?]{{0,40}}?(?:displayed\\s+)?diameter",
                    sent,
                    flags=re.I,
                )
                or re.search(
                    f"{kw}[^.!?]{{0,40}}?(?:displayed\\s+)?diameter[^.!?]{{0,90}}?\\b{ef}\\b",
                    sent,
                    flags=re.I,
                )
                or re.search(
                    f"\\b{ef}\\b[^.!?]{{0,90}}?(?:contains|has)[^.!?]{{0,45}}?(?:larger|largest)[^.!?]{{0,35}}?diameter",
                    sent,
                    flags=re.I,
                )
                or re.search(
                    f"(?:compar(?:e|ing)[^.!?]{{0,45}}(?:diameters?|values)[^.!?]{{0,55}})?\\b{ef}\\b\\s+is\\s+{kw}\\b",
                    sent,
                    flags=re.I,
                )
                or re.search(
                    f"\\b{ef}\\b\\s+has\\s+the\\s+{kw}\\s+(?:displayed\\s+)?diameter", sent, flags=re.I
                )
            ):
                return f
    for f in candidates:
        ef = re.escape(f)
        if re.search(
            f"\\bselect(?:ed)?\\s+(?:feature\\s+)?\\b{ef}\\b|\\b{ef}\\b\\s+(?:is\\s+)?(?:the\\s+)?selected(?:\\s+feature)?\\b|\\bselected\\s+(?:marked\\s+)?feature\\s+(?:is\\s+)?\\b{ef}\\b",
            text,
            flags=re.I,
        ):
            if re.search("\\b(?:larger|largest|greater|greatest)\\b", text, flags=re.I):
                return f
    for f in candidates:
        comp = facts.get(f, {}).get("component")
        if not comp:
            continue
        ec = re.escape(str(comp))
        for sent in sents:
            if re.search(ec, sent, flags=re.I) and re.search(
                f"{kw}[^.!?]{{0,40}}?(?:displayed\\s+)?diameter|(?:displayed\\s+)?diameter[^.!?]{{0,40}}?{kw}",
                sent,
                flags=re.I,
            ):
                return f
    for f in candidates:
        ef = re.escape(f)
        if re.search(
            f"\\b{ef}\\b(?:['’]s)?[^.!?]{{0,45}}?(?:is\\s+)?(?:larger|largest|greater|greatest)\\b|\\b{ef}\\b[^.!?]{{0,35}}?\\bselected\\b|\\bselect(?:ed)?\\s+\\b{ef}\\b",
            text,
            flags=re.I,
        ) and re.search("\\b(?:diameter|diameters|qualifying\\s+features)\\b", text, flags=re.I):
            return f
    return None


def _quantity_winner(text: str, op: dict[str, Any], facts: dict[str, dict[str, Any]]) -> str | None:
    candidates = [_feature(x) for x in op.get("candidates") or []]
    sents = _sents(text)
    for f in candidates:
        ef = re.escape(f)
        for sent in sents:
            if re.search(
                f"\\b{ef}\\b[^.!?]{{0,100}}?(?:used\\s+more\\s+times|more\\s+frequently\\s+used|greater\\s+quantity|higher\\s+quantity|larger\\s+quantity)",
                sent,
                flags=re.I,
            ):
                return f
            if re.search(
                f"(?:select(?:ed)?|selected\\s+(?:marked\\s+)?feature\\s+(?:is\\s+)?)\\b{ef}\\b",
                sent,
                flags=re.I,
            ):
                return f
    for f in candidates:
        comp = facts.get(f, {}).get("component")
        if not comp:
            continue
        ec = re.escape(str(comp))
        for sent in sents:
            if re.search(
                f"\\b{ec}\\b[^.!?]{{0,80}}?(?:is\\s+)?used\\s+more\\s+times\\s+than|\\b{ec}\\b[^.!?]{{0,80}}?(?:is\\s+)?more\\s+frequently\\s+used\\s+than|\\b{ec}\\b[^.!?]{{0,80}}?quantity[^.!?]{{0,35}}?(?:greater|higher|larger)\\s+than|(?:component\\s+used\\s+more\\s+times|more\\s+frequently\\s+used\\s+component)[^.!?]{{0,35}}?\\bis\\s+\\b{ec}\\b",
                sent,
                flags=re.I,
            ):
                return f
    for f in candidates:
        ef = re.escape(f)
        if re.search(
            f"(?:component\\s+associated\\s+with|component\\s+for)\\s+\\b{ef}\\b[^.!?]{{0,60}}?(?:used\\s+more\\s+times|more\\s+frequently\\s+used)",
            text,
            flags=re.I,
        ):
            return f
        if re.search(f"(?:therefore|so)[^.!?]{{0,40}}?(?:select(?:ed)?\\s+)?\\b{ef}\\b", text, flags=re.I):
            return f
    for f in candidates:
        ef = re.escape(f)
        if re.search(
            f"(?:feature|marked\\s+feature)[^.!?]{{0,80}}?(?:more\\s+frequently\\s+used|more-used|used\\s+more\\s+times)[^.!?]{{0,50}}?\\bis\\s+\\b{ef}\\b|(?:therefore|so)[^.!?]{{0,60}}?\\b{ef}\\b[^.!?]{{0,40}}?(?:selected|belongs)",
            text,
            flags=re.I,
        ):
            return f
    return None


def _filter_evidence(text: str, op: dict[str, Any], states: dict[tuple[str, str], str]) -> tuple[bool, bool]:
    candidates = [_feature(x) for x in op.get("candidates") or []]
    qualified = {_feature(x) for x in op.get("qualified") or []}
    filter_terms = "qualif(?:y|ies|ied|ying)|pass(?:es|ed)?\\s+(?:the\\s+)?(?:more-than-once\\s+)?filter|meet(?:s|ing)?\\s+(?:the\\s+)?(?:more-than-once\\s+)?(?:condition|filter)|used\\s+more\\s+than\\s+once|quantity\\s+(?:greater\\s+than|>)\\s*1|more\\s+than\\s+one"
    has_filter_language = bool(re.search(f"\\b(?:{filter_terms})\\b", text, flags=re.I))
    if not has_filter_language:
        return (False, False)
    qty_grounded = all((_correct(states, f, "quantity") for f in candidates))
    if not qty_grounded:
        return (False, False)
    explicit_qualified: set[str] = set()
    for f in candidates:
        ef = re.escape(f)
        if re.search(
            f"\\b{ef}\\b[^.!?]{{0,100}}?(?:{filter_terms})|(?:{filter_terms})[^.!?]{{0,100}}?\\b{ef}\\b",
            text,
            flags=re.I,
        ):
            explicit_qualified.add(f)
            continue
        for sent in _sents(text):
            if re.search(f"\\b{ef}\\b", sent, flags=re.I) and re.search(
                f"\\b(?:it|this\\s+feature)\\b[^.!?]{{0,25}}?(?:{filter_terms})", sent, flags=re.I
            ):
                explicit_qualified.add(f)
                break
    set_ok = bool(qualified) and qualified.issubset(explicit_qualified)
    return (True, set_ok)


def _required_keys(payload: dict[str, Any]) -> set[tuple[str, str]]:
    graph = _graph(payload)
    facts = _facts(graph)
    fam = _family(payload)
    keys: set[tuple[str, str]] = set()
    if fam in {HARD_C1, HARD_C2}:
        op = _op(graph, "compare_diameter") or {}
        cand = [_feature(x) for x in op.get("candidates") or []]
        winner = _feature(op.get("winner"))
        keys |= {(f, "diameter_mm") for f in cand if facts.get(f, {}).get("diameter_mm") is not None}
        for slot in ("bom_item", "component"):
            if facts.get(winner, {}).get(slot) is not None:
                keys.add((winner, slot))
        if fam == HARD_C2 and facts.get(winner, {}).get("quantity") is not None:
            keys.add((winner, "quantity"))
    elif fam == HARD_C3:
        op = _op(graph, "compare_quantity") or {}
        for f in [_feature(x) for x in op.get("candidates") or []]:
            for slot in ("bom_item", "component", "quantity"):
                if facts.get(f, {}).get(slot) is not None:
                    keys.add((f, slot))
    elif fam == HARD_C4:
        op = _op(graph, "filter_quantity_gt") or {}
        cand = [_feature(x) for x in op.get("candidates") or []]
        qual = [_feature(x) for x in op.get("qualified") or []]
        for f in cand:
            for slot in ("bom_item", "component", "quantity"):
                if facts.get(f, {}).get(slot) is not None:
                    keys.add((f, slot))
        for f in qual:
            if facts.get(f, {}).get("diameter_mm") is not None:
                keys.add((f, "diameter_mm"))
    return keys


def _contradiction(payload: dict[str, Any], states: dict[tuple[str, str], str]) -> tuple[float, int, int]:
    keys = _required_keys(payload)
    if not keys:
        return (0.0, 0, 0)
    bad = sum((states.get(k) == "contradiction" for k in keys))
    return (bad / len(keys), bad, len(keys))


def _path_diag(text: str, payload: dict[str, Any]) -> dict[str, float]:
    graph = _graph(payload)
    facts = _facts(graph)
    states = _feature_states(text, graph)
    fam = _family(payload)
    stage: dict[str, float] = {}
    terminal = False
    if fam == HARD_C1:
        op = _op(graph, "compare_diameter") or {}
        cand = [_feature(x) for x in op.get("candidates") or []]
        winner = _feature(op.get("winner"))
        dia = bool(cand) and all((_correct(states, f, "diameter_mm") for f in cand))
        cmp_ok = dia and _diameter_winner(text, op, facts) == winner
        bom = _correct(states, winner, "bom_item")
        comp = _correct(states, winner, "component")
        terminal = bool(dia and cmp_ok and bom and comp)
        stage = {
            "stage_diameter_grounding": 0.15 * dia,
            "stage_compare": 0.15 * cmp_ok,
            "stage_bom_traversal": 0.2 * bom,
            "stage_component_traversal": 0.2 * comp,
            "stage_terminal": 0.3 * terminal,
        }
    elif fam == HARD_C2:
        op = _op(graph, "compare_diameter") or {}
        cand = [_feature(x) for x in op.get("candidates") or []]
        winner = _feature(op.get("winner"))
        dia = bool(cand) and all((_correct(states, f, "diameter_mm") for f in cand))
        cmp_ok = dia and _diameter_winner(text, op, facts) == winner
        bom = _correct(states, winner, "bom_item")
        comp = _correct(states, winner, "component")
        qty = _correct(states, winner, "quantity")
        terminal = bool(dia and cmp_ok and bom and comp and qty)
        stage = {
            "stage_diameter_grounding": 0.1 * dia,
            "stage_compare": 0.15 * cmp_ok,
            "stage_bom_traversal": 0.15 * bom,
            "stage_component_traversal": 0.15 * comp,
            "stage_quantity_traversal": 0.2 * qty,
            "stage_terminal": 0.25 * terminal,
        }
    elif fam == HARD_C3:
        op = _op(graph, "compare_quantity") or {}
        cand = [_feature(x) for x in op.get("candidates") or []]
        winner = _feature(op.get("winner"))

        def branch(f: str) -> bool:
            return all((_correct(states, f, x) for x in ("bom_item", "component", "quantity")))

        a = len(cand) > 0 and branch(cand[0])
        b = len(cand) > 1 and branch(cand[1])
        cmp_ok = bool(a and b and (_quantity_winner(text, op, facts) == winner))
        winner_explicit = _quantity_winner(text, op, facts) == winner
        terminal = bool(a and b and cmp_ok)
        stage = {
            "stage_branch_a": 0.25 * a,
            "stage_branch_b": 0.25 * b,
            "stage_quantity_compare": 0.2 * cmp_ok,
            "stage_winner": 0.1 * winner_explicit,
            "stage_terminal": 0.2 * terminal,
        }
    elif fam == HARD_C4:
        filt = _op(graph, "filter_quantity_gt") or {}
        cmpop = _op(graph, "compare_diameter") or {}
        cand = [_feature(x) for x in filt.get("candidates") or []]
        qual = [_feature(x) for x in filt.get("qualified") or []]
        winner = _feature(cmpop.get("winner"))
        branch_qty = bool(cand) and all(
            (all((_correct(states, f, x) for x in ("bom_item", "component", "quantity"))) for f in cand)
        )
        filter_ok, set_ok = _filter_evidence(text, filt, states)
        dia = bool(qual) and all((_correct(states, f, "diameter_mm") for f in qual))
        cmp_ok = bool(filter_ok and set_ok and dia and (_diameter_winner(text, cmpop, facts) == winner))
        terminal = bool(branch_qty and filter_ok and set_ok and dia and cmp_ok)
        stage = {
            "stage_quantity_grounding": 0.2 * branch_qty,
            "stage_filter_operation": 0.25 * filter_ok,
            "stage_qualified_set": 0.15 * set_ok,
            "stage_diameter_grounding": 0.15 * dia,
            "stage_diameter_compare": 0.1 * cmp_ok,
            "stage_terminal": 0.15 * terminal,
        }
    raw = float(sum(stage.values()))
    path = raw if terminal else min(raw, 0.5)
    contra, contra_n, req_n = _contradiction(payload, states)
    base_diag = {
        "path": path,
        "raw_stage": raw,
        "terminal": float(terminal),
        "contradiction": contra,
        "contradiction_count": float(contra_n),
        "required_fact_count": float(req_n),
        "fact_correct_count": float(sum((v == "correct" for v in states.values()))),
        "fact_contradiction_count": float(sum((v == "contradiction" for v in states.values()))),
        "fact_missing_count": float(sum((v == "missing" for v in states.values()))),
    }
    base_diag.update({k: float(v) for k, v in stage.items()})
    return base_diag
