"""Verified terminal closure self-distillation; targets always come from model sampling."""

from __future__ import annotations
import re
from typing import Optional
from .utils import TokenAligner
from .schemas import SafetyError
from .prompts import FORBIDDEN_CONTEXT

_FINAL_LINE_RE = re.compile(r"(?im)^\s*FINAL\s*:[^\n]*")


def terminal_prefix(
    ids: list[int], tokenizer, skip: bool, graph, verifier
) -> Optional[tuple[list[int], str, frozenset]]:
    """Find the earliest aligned prefix closing the requested terminal dependency graph."""
    al = TokenAligner(tokenizer, ids, skip)
    text = al.text
    end = verifier.reasoning_end(text)
    cut = next(
        (
            c
            for c in verifier.sentence_cuts(text, 0, end)
            if graph.terminal in verifier.verify_text(text[:c], graph).closed
        ),
        None,
    )
    if cut is None:
        return None
    t = al.tokens_for_prefix(text[:cut])
    if t is None:
        return None
    pre_text = al.decode(t)
    v = verifier.verify_text(pre_text, graph)
    if graph.terminal not in v.closed or "</think>" in pre_text or _FINAL_LINE_RE.search(pre_text):
        return None
    return (list(ids[:t]), pre_text, v.closed)


def evaluate_closure(
    prefix_text: str, closed, cand_ids: list[int], tokenizer, skip: bool, payload: dict, graph, verifier
) -> dict:
    """Apply the frozen Z verifier and cut a sampled closure through its FINAL line."""
    cand = tokenizer.decode(cand_ids, skip_special_tokens=skip)
    full = prefix_text + cand
    end = verifier.reasoning_end(full)
    m = _FINAL_LINE_RE.search(full, max(end, len(prefix_text)))
    A = bool(verifier.answer_correct(full, payload))
    F = float(verifier.format_reward(full))
    whole = full[:end]
    contra = any((verifier.node_status(whole, graph, n)[0] == verifier.CONTRADICTION for n in closed))
    Z = int(m is not None and A and (F == 1.0) and (not contra))
    out = {
        "ids": list(cand_ids),
        "text": cand,
        "Z": Z,
        "A": A,
        "F": F,
        "has_final": m is not None,
        "contradiction": contra,
        "tokens": len(cand_ids),
        "closure_tokens": None,
    }
    if Z:
        fin_end = m.end() - len(prefix_text)
        for t in range(1, len(cand_ids) + 1):
            d = tokenizer.decode(cand_ids[:t], skip_special_tokens=skip)
            if len(d) >= fin_end and d[:fin_end] == cand[:fin_end]:
                out["closure_tokens"] = t
                break
        if out["closure_tokens"] is None:
            out["Z"] = 0
    return out


def select_closure(cands: list[dict]) -> Optional[int]:
    """Choose the shortest valid closure, preserving child order on ties."""
    ok = [(i, c) for i, c in enumerate(cands) if c["Z"] == 1 and (not c["contradiction"])]
    return min(ok, key=lambda x: (x[1]["closure_tokens"], x[0]))[0] if ok else None


def terminal_trigger(closed: frozenset[str], graph) -> bool:
    """Only a verified requested terminal node can initiate closure search."""
    return graph.terminal in closed


def require_signal_free(context: str) -> None:
    """Closure requests have no operation signal and no termination logit-bias override."""
    if any(marker in context for marker in FORBIDDEN_CONTEXT):
        raise SafetyError("Closure context contains a reserved operation control")
