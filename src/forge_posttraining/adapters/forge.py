"""FORGE graph/text verifier, isolated from inference."""

from __future__ import annotations
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Callable, Optional
from ..ontology import Node, OntologyGraph, OntologyGraphError, PrefixState, Verification
from . import _text_grammar as _grammar
from ._answer import answer_correct, format_reward, normalize_component, normalize_feature, parse_quantity

RELATION_SLOT = {
    "has_diameter": "diameter_mm",
    "owner_bom_item": "bom_item",
    "identifies_component": "component",
    "has_quantity": "quantity",
}
_OP_CONSUMES = {
    "compare_diameter": "diameter_mm",
    "compare_quantity": "quantity",
    "filter_quantity_gt": "quantity",
}
_SUPPORTED_COMPARE_MODES = {"max"}


def _parse_gold(slot: str, value: Any) -> Any:
    s = str(value)
    if slot == "diameter_mm":
        m = re.fullmatch("\\s*([+-]?\\d+(?:\\.\\d+)?)\\s*mm\\s*", s)
        if not m:
            raise OntologyGraphError(f"unparseable diameter value {value!r}")
        return float(m.group(1))
    if slot == "bom_item":
        m = re.fullmatch("\\s*BOM item\\s+(\\d+)\\s*", s)
        if not m:
            raise OntologyGraphError(f"unparseable bom_item value {value!r}")
        return int(m.group(1))
    if slot == "quantity":
        if not re.fullmatch("\\s*\\d+\\s*", s):
            raise OntologyGraphError(f"unparseable quantity value {value!r}")
        return int(s)
    if not s.strip():
        raise OntologyGraphError("empty component value")
    return s


def _fact_id(feature: str, slot: str) -> str:
    return f"{feature}.{slot}"


def _matches_answer(node: Node, canonical: Any, answer_type: str) -> bool:
    if node.kind == "op":
        if not isinstance(node.gold, str):
            return False
        return answer_type == "feature" and normalize_feature(canonical) == normalize_feature(node.gold)
    if answer_type == "component":
        return node.slot == "component" and normalize_component(node.gold) == normalize_component(canonical)
    if answer_type == "quantity":
        return node.slot == "quantity" and parse_quantity(canonical) == node.gold
    if answer_type == "feature":
        return False
    raise OntologyGraphError(f"unsupported answer_type {answer_type!r}")


def build_graph(payload: dict[str, Any]) -> OntologyGraph:
    """Construct strict relation/operation prerequisites and the unique requested terminal."""
    g = payload.get("reward_graph")
    if not isinstance(g, dict):
        raise OntologyGraphError("missing reward_graph")
    raw_facts = g.get("facts_by_feature") or {}
    facts = {_grammar._feature(k): v for k, v in raw_facts.items() if isinstance(v, dict)}
    paths = g.get("paths") or []
    ops = g.get("operations") or []
    if not paths:
        raise OntologyGraphError("no gold paths")
    spec: dict[str, dict[str, Any]] = {}
    path_nodes: list[list[str]] = []
    for p in paths:
        if not p or p[0]["src"]["kind"] != "feature":
            raise OntologyGraphError("path does not start at a feature")
        root = _grammar._feature(p[0]["src"]["value"])
        ids: list[str] = []
        for j, e in enumerate(p):
            if j and e["src"] != p[j - 1]["dst"]:
                raise OntologyGraphError("discontinuous gold path")
            rel = e["relation"]
            if rel not in RELATION_SLOT:
                raise OntologyGraphError(f"unsupported relation {rel!r}")
            slot = RELATION_SLOT[rel]
            gold = _parse_gold(slot, e["dst"]["value"])
            fact_value = facts.get(root, {}).get(slot)
            if (
                fact_value is None
                or (slot == "component" and normalize_component(fact_value) != normalize_component(gold))
                or (slot != "component" and (not _grammar._num_eq(fact_value, gold)))
            ):
                raise OntologyGraphError(f"path edge disagrees with facts_by_feature: {root}.{slot}")
            nid = _fact_id(root, slot)
            prereqs = {ids[-1]} if ids else set()
            if nid in spec:
                if spec[nid]["gold"] != gold or spec[nid]["label"] != rel:
                    raise OntologyGraphError(f"conflicting duplicate gold edge {nid}")
                spec[nid]["prereqs"] |= prereqs
            else:
                spec[nid] = {
                    "kind": "fact",
                    "label": rel,
                    "feature": root,
                    "slot": slot,
                    "gold": gold,
                    "prereqs": set(prereqs),
                }
            ids.append(nid)
        path_nodes.append(ids)
    consumed: set[str] = set()
    for i, op in enumerate(ops):
        typ = op.get("type")
        if typ not in _OP_CONSUMES:
            raise OntologyGraphError(f"unsupported operation {typ!r}")
        cands = [_grammar._feature(x) for x in op.get("candidates") or []]
        if len(cands) < 2:
            raise OntologyGraphError(f"operation {typ} has fewer than two candidates")
        slot = _OP_CONSUMES[typ]
        prereqs = set()
        for c in cands:
            nid = _fact_id(c, slot)
            if nid not in spec:
                raise OntologyGraphError(f"operation {typ} lacks gold prerequisite {nid}")
            prereqs.add(nid)
            consumed.add(nid)
        if typ.startswith("compare_"):
            if op.get("mode") not in _SUPPORTED_COMPARE_MODES:
                raise OntologyGraphError(f"unsupported compare mode {op.get('mode')!r}")
            winner = _grammar._feature(op.get("winner"))
            if winner not in cands:
                raise OntologyGraphError("winner not among candidates")
            gold: Any = winner
            for k in range(i):
                prev = ops[k]
                if prev.get("type") == "filter_quantity_gt" and {
                    _grammar._feature(x) for x in prev.get("qualified") or []
                } == set(cands):
                    prereqs.add(f"op{k}:{prev['type']}")
        else:
            qualified = frozenset((_grammar._feature(x) for x in op.get("qualified") or []))
            if not qualified or not qualified <= set(cands):
                raise OntologyGraphError("filter qualified set invalid")
            if not isinstance(op.get("threshold"), (int, float)):
                raise OntologyGraphError("filter threshold missing")
            gold = qualified
        spec[f"op{i}:{typ}"] = {
            "kind": "op",
            "label": typ,
            "feature": None,
            "slot": None,
            "gold": gold,
            "prereqs": prereqs,
            "cands": tuple(cands),
        }
    for ids in path_nodes:
        if any((n in consumed for n in ids)):
            continue
        root = spec[ids[0]]["feature"]
        producers = [
            oid
            for oid, s in spec.items()
            if s["kind"] == "op" and (s["gold"] == root if isinstance(s["gold"], str) else root in s["gold"])
        ]
        if len(producers) == 1:
            spec[ids[0]]["prereqs"].add(producers[0])
        elif len(producers) > 1:
            raise OntologyGraphError(f"ambiguous producer for downstream path at {root}")
    depth: dict[str, int] = {}

    def _depth(nid: str, stack: tuple[str, ...] = ()) -> int:
        if nid in stack:
            raise OntologyGraphError("prerequisite cycle")
        if nid not in depth:
            depth[nid] = 1 + max((_depth(p, stack + (nid,)) for p in spec[nid]["prereqs"]), default=0)
        return depth[nid]

    for nid in spec:
        _depth(nid)
    nodes = {
        nid: Node(
            nid,
            s["kind"],
            s["label"],
            s["feature"],
            s["slot"],
            s["gold"],
            tuple(sorted(s["prereqs"])),
            depth[nid],
        )
        for nid, s in spec.items()
    }
    has_dependents = {p for n in nodes.values() for p in n.prereqs}
    sinks = [n for n in nodes.values() if n.id not in has_dependents]
    canonical = payload.get("canonical_answer")
    answer_type = payload.get("answer_type")
    hits = [n for n in sinks if _matches_answer(n, canonical, answer_type)]
    if len(hits) != 1:
        raise OntologyGraphError(f"terminal not unique: {len(hits)} sink(s) match canonical answer")
    terminal = hits[0].id
    required: set[str] = set()
    todo = [terminal]
    while todo:
        n = todo.pop()
        if n not in required:
            required.add(n)
            todo.extend(nodes[n].prereqs)
    return OntologyGraph(
        nodes,
        terminal,
        frozenset(required),
        facts,
        list(ops),
        {nid: s["cands"] for nid, s in spec.items() if s["kind"] == "op"},
    )


VERIFIED, MISSING, CONTRADICTION = ("verified", "missing", "contradiction")
_FILTER_TERMS = "qualif(?:y|ies|ied|ying)|pass(?:es|ed)?\\s+(?:the\\s+)?(?:more-than-once\\s+)?filter|meet(?:s|ing)?\\s+(?:the\\s+)?(?:more-than-once\\s+)?(?:condition|filter)|used\\s+more\\s+than\\s+once|quantity\\s+(?:greater\\s+than|>)\\s*1|more\\s+than\\s+one"
_NEGATION = "\\b(?:not|never|n't|fails?|failed|exclud\\w*|disqualif\\w*|only\\s+once|exactly\\s+once)\\b"


def _numeric_status(values: list, gold: Any) -> str:
    if not values:
        return MISSING
    return VERIFIED if all((_grammar._num_eq(v, gold) for v in values)) else CONTRADICTION


_X = "(?P<x>[^.;!?]+?)(?=\\s*(?:[.;!?](?:\\s|$)|$|,\\s*(?:which|so|and|but|therefore)\\b))"
_BOM_CLAIM = (
    "\\bBOM\\s*(?:item)?\\s*#?\\s*{b}\\b\\s*,?\\s*(?:identifies|corresponds\\s+to|refers\\s+to|is\\s+(?:labeled|named|listed\\s+as)|names)\\s+(?:the\\s+)?(?:component\\s+)?"
    + _X
)
_FEATURE_CLAIMS = (
    "\\bcomponent\\s+(?:containing|associated\\s+with|for|that\\s+contains)\\s+(?:the\\s+(?:marked\\s+)?feature\\s+)?\\b{f}\\b\\s+(?:is|=)\\s+(?:the\\s+)?(?:component\\s+)?"
    + _X,
    "\\b{f}\\b\\s+belongs\\s+to\\s+(?:the\\s+)?component\\s+" + _X,
)
_NON_ENTITY = re.compile(
    "\\b(?:used|uses|use|listed|appears?|has|have|having|is|are|was|were|be|quantit(?:y|ies)|times?|instances?|same|more|less|than|which|that|with|associated|corresponding|containing|contains|number|feature|larger|largest|smaller|smallest|greater|displayed|diameter|bom|item|marker|owner)\\b",
    re.I,
)


def _component_norm(x: str) -> str:
    s = normalize_component(str(x).strip(" \"'`*"))
    s = re.sub("^(?:the|a|an)\\s+", "", s)
    s = re.sub("\\s+(?:component|part)$", "", s).strip()
    while s.endswith(")") and s.count(")") > s.count("("):
        s = s[:-1].rstrip()
    while s.startswith("(") and s.count("(") > s.count(")"):
        s = s[1:].lstrip()
    return s


def _component_claims(
    text: str, feature: str, bom: Optional[int], gold: str, other_components: set[str]
) -> tuple[list[str], list[str]]:
    g = _component_norm(gold)
    others = {_component_norm(c) for c in other_components} - {g}
    support, conflict = ([], [])
    pats = [p.format(f=re.escape(feature)) for p in _FEATURE_CLAIMS]
    if bom is not None:
        pats.insert(0, _BOM_CLAIM.format(b=int(bom)))
    for sent in _grammar._sents(text):
        for pat in pats:
            for m in re.finditer(pat, sent, flags=re.I):
                x = m.group("x").strip()
                nx = _component_norm(x)
                if not nx or re.fullmatch("F\\d{1,3}", x.strip(), flags=re.I):
                    continue
                if nx == g:
                    support.append(x)
                elif nx in others or (len(x) <= 70 and len(x.split()) <= 12 and (not _NON_ENTITY.search(x))):
                    conflict.append(x)
        if (
            bom is not None
            and re.search(f"\\bBOM\\s*(?:item)?\\s*#?\\s*{int(bom)}\\b", sent, flags=re.I)
            and re.search(f"(?<![\\w(]){re.escape(str(gold))}(?![\\w(])", sent, flags=re.I)
        ):
            support.append(str(gold))
    return (support, conflict)


_PAREN_MM = re.compile("\\(\\s*[øØ⌀Φφ]?\\s*([+-]?\\d+(?:\\.\\d+)?)\\s*mm\\s*\\)", re.I)


def _paren_diameter_values(text: str, feature: str) -> list[float]:
    vals = []
    for sent in _grammar._sents(text):
        feats = [
            (m.end(), _grammar._feature(m.group(0)))
            for m in re.finditer("\\bF\\s*\\d{1,3}\\b", sent, flags=re.I)
        ]
        for m in _PAREN_MM.finditer(sent):
            before = [(end, f) for end, f in feats if end <= m.start()]
            if (
                before
                and before[-1][1] == feature
                and (m.start() - before[-1][0] <= 80)
                and (not _PAREN_MM.search(sent[before[-1][0] : m.start()]))
            ):
                vals.append(float(m.group(1)))
    return vals


def _fact_assertions(text: str, node: Node, facts: dict[str, dict[str, Any]]) -> list:
    rec = facts.get(node.feature, {})
    comp, bom = (rec.get("component"), rec.get("bom_item"))
    if node.slot == "diameter_mm":
        return _grammar._feature_diameter_values(
            text, node.feature, str(comp) if comp else None
        ) + _paren_diameter_values(text, node.feature)
    if node.slot == "bom_item":
        return _grammar._feature_bom_values(text, node.feature)
    if node.slot == "quantity":
        return _grammar._quantity_values(
            text, node.feature, str(comp) if comp else None, int(bom) if bom is not None else None
        )
    if node.slot == "component":
        others = {str(r.get("component")) for f, r in facts.items() if r.get("component")}
        support, conflict = _component_claims(text, node.feature, bom, str(node.gold), others)
        return [("support", x) for x in support] + [("conflict", x) for x in conflict]
    raise OntologyGraphError(f"no extractor for slot {node.slot!r}")


def _winner_claims(text: str, graph: OntologyGraph, node: Node) -> set[str]:
    detector = _grammar._diameter_winner if node.label == "compare_diameter" else _grammar._quantity_winner
    cands = graph.op_candidates[node.id]
    return {c for c in cands if detector(text, {"candidates": [c]}, graph.facts) == c}


def _filter_claims(text: str, cands: tuple[str, ...]) -> tuple[set[str], set[str], bool]:
    positive, negative = (set(), set())
    language = bool(re.search(f"\\b(?:{_FILTER_TERMS})\\b", text, flags=re.I))
    for sent in _grammar._sents(text):
        if not re.search(f"\\b(?:{_FILTER_TERMS})\\b", sent, flags=re.I):
            continue
        mentioned = [c for c in cands if re.search(f"\\b{re.escape(c)}\\b", sent, flags=re.I)]
        (negative if re.search(_NEGATION, sent, flags=re.I) else positive).update(mentioned)
    return (positive, negative, language)


def node_status(text: str, graph: OntologyGraph, nid: str) -> tuple[str, list]:
    """Check support and conflicting claims for one supplied entity-slot or operation."""
    node = graph.nodes[nid]
    if node.kind == "fact":
        vals = _fact_assertions(text, node, graph.facts)
        if node.slot == "component":
            if any((kind == "conflict" for kind, _ in vals)):
                return (CONTRADICTION, vals)
            return (VERIFIED if vals else MISSING, vals)
        return (_numeric_status(vals, node.gold), vals)
    if node.label.startswith("compare_"):
        claims = sorted(_winner_claims(text, graph, node))
        if not claims:
            return (MISSING, claims)
        return (VERIFIED if claims == [node.gold] else CONTRADICTION, claims)
    cands = graph.op_candidates[nid]
    pos, neg, language = _filter_claims(text, cands)
    claims = [sorted(pos), sorted(neg)]
    qualified = set(node.gold)
    if neg & qualified or pos - qualified or set(cands) - qualified & pos:
        return (CONTRADICTION, claims)
    if language and qualified <= pos:
        return (VERIFIED, claims)
    return (MISSING, claims)


def verify_text(text: str, graph: OntologyGraph) -> Verification:
    """Normalize text, check required facts, and compute prerequisite-closed evidence."""
    text = unicodedata.normalize("NFKC", text)
    status, evidence = ({}, {})
    for nid in graph.required:
        status[nid], evidence[nid] = node_status(text, graph, nid)
    closed = graph.closure({n for n, s in status.items() if s == VERIFIED})
    return Verification(status, evidence, closed, graph.frontier(closed), graph.rank(closed))


_FINAL_LINE = re.compile("(?im)^\\s*FINAL\\s*:")


def reasoning_end(response: str) -> int:
    """Locate the first closing-think marker or final-answer line."""
    k = response.find("</think>")
    if k >= 0:
        return k
    m = _FINAL_LINE.search(response)
    return m.start() if m else len(response)


def sentence_cuts(text: str, start: int, end: int) -> list[int]:
    """Enumerate nonempty sentence cuts, including the supplied reasoning end."""
    cuts = {m.start() + start for m in re.finditer("(?<=[.!?])\\s+|\\n+", text[start:end])}
    cuts.add(end)
    return sorted((c for c in cuts if c > start and text[start:c].strip()))


def best_prefix_state(
    response: str, graph: OntologyGraph, token_len: Optional[Callable[[str], int]] = None
) -> Optional[PrefixState]:
    """Keep the first token-aligned prefix at the greatest verified causal rank."""
    token_len = token_len or len
    best: Optional[PrefixState] = None
    for cut in sentence_cuts(response, 0, reasoning_end(response)):
        v = verify_text(response[:cut], graph)
        if not v.closed or (best is not None and v.rank <= best.rank):
            continue
        n_tok = token_len(response[:cut])
        if n_tok is None:
            continue
        best = PrefixState(cut, v.closed, v.frontier, v.rank, n_tok)
    return best


def assert_reset_prefix_safe(prefix: str) -> None:
    """Reject answer/termination text from a reasoning-state student context."""
    if "</think>" in prefix or _FINAL_LINE.search(prefix):
        raise OntologyGraphError("reset prefix contains </think> or a FINAL line")


@dataclass
class ResetChoice:
    rollout_index: Optional[int]
    state: Optional[PrefixState]
    skip_reason: Optional[str]


def select_reset_state(
    responses: list[str],
    graph: OntologyGraph,
    token_len: Optional[Callable[[str], Optional[int]] | list] = None,
) -> ResetChoice:
    """Select deepest non-terminal closure, then shortest tokens and earliest rollout."""
    candidates, any_verified = ([], False)
    for i, r in enumerate(responses):
        s = best_prefix_state(r, graph, token_len[i] if isinstance(token_len, list) else token_len)
        if s is None:
            continue
        any_verified = True
        if not s.frontier:
            continue
        assert_reset_prefix_safe(r[: s.cut])
        candidates.append((-s.rank[0], -s.rank[1], s.token_len, i, s))
    if not candidates:
        return ResetChoice(None, None, "no_unresolved_transition" if any_verified else "no_verified_state")
    *_, i, s = min(candidates, key=lambda x: x[:4])
    return ResetChoice(i, s, None)


@dataclass
class SuffixEval:
    A: float
    E: float
    F: float
    first_transition_cut: Optional[int]
    new_closed: Optional[frozenset[str]]
    newly_verified: tuple[str, ...]
    e_reason: str
    prefix_state_broken: bool


def evaluate_suffix(
    prefix: str, suffix: str, graph: OntologyGraph, state: PrefixState, payload: dict[str, Any]
) -> SuffixEval:
    """Credit the first frontier transition only if prior and credited facts remain sound."""
    full = prefix + suffix
    A = float(answer_correct(full, payload))
    F = float(format_reward(full))
    r_end = reasoning_end(full)
    E, cut, new_closed, newly, reason, broken = (0.0, None, None, (), "no_transition", False)
    if r_end > len(prefix):
        whole = unicodedata.normalize("NFKC", full[:r_end])
        broken = any((node_status(whole, graph, n)[0] == CONTRADICTION for n in state.closed))
        if broken:
            reason = "state_contradicted"
        else:
            for c in sentence_cuts(full, len(prefix), r_end):
                v = verify_text(full[:c], graph)
                if state.closed <= v.closed and v.closed & state.frontier:
                    cut, new_closed = (c - len(prefix), v.closed)
                    newly = tuple(sorted(v.closed - state.closed))
                    break
            if cut is not None:
                if any((node_status(whole, graph, n)[0] == CONTRADICTION for n in newly)):
                    reason, cut, new_closed, newly = ("credited_contradicted", None, None, ())
                else:
                    E, reason = (1.0, "credited")
    return SuffixEval(A, E, F, cut, new_closed, newly, reason, broken)


def depth_advance(original: PrefixState, closed: frozenset[str], graph: OntologyGraph) -> int:
    """Measure progressive search depth relative to the original verified state."""
    return graph.rank(closed)[0] - original.rank[0]
