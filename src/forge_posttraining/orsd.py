"""Ontology-routed selective distillation of verified non-terminal transitions."""

from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import Any, Optional, Sequence
import re
from .ontology import OntologyGraphError, PrefixState
from .schemas import (
    AugmentedBatch,
    DirectGroup,
    DistillationTarget,
    GenerationRequest,
    Sampler,
    SamplingConfig,
    SafetyError,
)
from .utils import TokenAligner
from .prompts import SIGNAL_FORMAT, SIGNAL_MARKER, FORBIDDEN_CONTEXT
from .verifier import Verifier
from . import vtc

RELATION_BUDGET = {
    "owner_bom_item": 44,
    "has_diameter": 127,
    "compare_diameter": 30,
    "compare_quantity": 39,
    "has_quantity": 21,
    "identifies_component": 17,
}


@dataclass(frozen=True)
class ORSDConfig:
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    relation_budgets: dict[str, int] = field(default_factory=lambda: dict(RELATION_BUDGET))
    unsupported_relations: tuple[str, ...] = ("filter_quantity_gt",)
    max_depth_advance: int = 3
    response_length: int = 2048
    max_prompt_tokens: int = 16384
    closure_k: int = 58
    enable_vtc: bool = True
    skip_special_tokens: bool = True
    lambda_sd: float = 1.0

    def __post_init__(self):
        if self.sampling.first_n != 8 or self.sampling.retry_n != 16:
            raise SafetyError("The method requires first n=8 and retry n=16")
        if min(self.max_depth_advance, self.response_length, self.max_prompt_tokens, self.closure_k) <= 0:
            raise SafetyError("Search and context budgets must be positive")
        if any(not isinstance(v, int) or v <= 0 for v in self.relation_budgets.values()):
            raise SafetyError("Invalid relation budget")
        if set(self.relation_budgets) & set(self.unsupported_relations):
            raise SafetyError("Unsupported relations cannot be silently searched")

    def budget(self, relation: str) -> int | None:
        if relation in self.relation_budgets:
            return self.relation_budgets[relation]
        if relation in self.unsupported_relations:
            return None
        raise SafetyError("Unknown schema relation; fail closed")


@dataclass
class Branch:
    group: DirectGroup
    graph: Any
    original_state: Any
    state: Any
    model_ids: list[int]
    prefix_text: str
    level: int = 1

    @property
    def payload(self):
        return self.group.payload


def all_wrong(answer_correct: Sequence[float], n: int = 8) -> bool:
    """All-wrong direct groups leave GRPO; mixed and all-correct groups remain unchanged."""
    if len(answer_correct) != n or any(a not in (0, 1) for a in answer_correct):
        raise SafetyError("Expected eight binary direct correctness outcomes")
    return sum(answer_correct) == 0


def signal_text(prefix: str, relation: str, config: ORSDConfig | None = None) -> str:
    """The sole privileged teacher hint is a schema relation name."""
    if (config or ORSDConfig()).budget(relation) is None:
        raise SafetyError("Unsupported search relation")
    return ("" if prefix.endswith("\n") else "\n") + SIGNAL_FORMAT.format(relation=relation) + "\n"


def leakage_audit(signal: str, payload: dict, graph) -> list[str]:
    words = {str(payload.get("canonical_answer", ""))}
    for f, rec in graph.facts.items():
        words.add(f)
        if rec.get("component"):
            words.add(str(rec["component"]))
    for n in graph.nodes.values():
        if n.kind == "op" and isinstance(n.gold, str):
            words.add(n.gold)
    leaks = [
        w
        for w in sorted(words)
        if w.strip() and re.search(f"(?<![\\w]){re.escape(w)}(?![\\w])", signal, flags=re.I)
    ]
    if re.search("\\d", signal):
        leaks.append("digit")
    return leaks


def candidate_contradictions(text: str, graph, verifier) -> int:
    return sum((s == verifier.CONTRADICTION for s in verifier.verify_text(text, graph).status.values()))


def evaluate_teacher_chunk(
    br: Branch, signal_ids: list[int], chunk_ids: list[int], tokenizer, relation: str, skip, verifier, blocked
) -> dict:
    head = tokenizer.decode(br.model_ids + signal_ids, skip_special_tokens=skip)
    full = tokenizer.decode(br.model_ids + signal_ids + chunk_ids, skip_special_tokens=skip)
    blocked = [t for t in chunk_ids if t in blocked]
    if blocked:
        raise SafetyError(f"Search mask violated: blocked tokens sampled {blocked[:4]}")
    if not head.startswith(br.prefix_text) or not full.startswith(head):
        return {"valid": False, "E": 0.0, "A": 0.0, "reason": "decode_mismatch", "chunk_ids": list(chunk_ids)}
    chunk_text = full[len(head) :]
    ev = verifier.evaluate_suffix(br.prefix_text, chunk_text, br.graph, br.state, br.payload)
    has_final_line = verifier._FINAL_LINE.search(chunk_text) is not None
    A = float(verifier.answer_correct(br.prefix_text + chunk_text, br.payload)) if has_final_line else 0.0
    out = {
        "valid": True,
        "chunk_ids": list(chunk_ids),
        "chunk_text": chunk_text,
        "E": ev.E,
        "A": A,
        "e_reason": ev.e_reason,
        "newly": list(ev.newly_verified),
        "contradiction": ev.e_reason in ("state_contradicted", "credited_contradicted"),
        "copies_relation_literal": relation in chunk_text,
        "cut_tokens": None,
        "advance": None,
        "new_closed": None,
        "contradiction_count": None,
    }
    if ev.E == 1:
        al = TokenAligner(tokenizer, chunk_ids, skip)
        t = al.tokens_for_prefix(chunk_text[: ev.first_transition_cut]) if al.text == chunk_text else None
        if t is None:
            out.update(E=0.0, e_reason="cut_unaligned")
        elif SIGNAL_MARKER.lower() in al.decode(t).lower():
            out.update(E=0.0, e_reason="copied_control_line")
        else:
            cut_text = br.prefix_text + al.decode(t)
            out.update(
                cut_tokens=t,
                new_closed=sorted(ev.new_closed),
                advance=verifier.depth_advance(br.original_state, ev.new_closed, br.graph),
                contradiction_count=candidate_contradictions(cut_text, br.graph, verifier),
            )
    return out


def select_candidate(cands: list[dict]) -> Optional[int]:
    ok = [(i, c) for i, c in enumerate(cands) if c.get("valid") and c["E"] == 1 and c["cut_tokens"]]
    if not ok:
        return None
    return min(ok, key=lambda x: (-x[1]["advance"], x[1]["cut_tokens"], x[1]["contradiction_count"], x[0]))[0]


class ORSDController:
    """Generate verified SD targets without constructing answers or altering the actor."""

    def __init__(
        self,
        tokenizer,
        sampler: Sampler,
        verifier: Verifier,
        actor_revision: str,
        config: ORSDConfig | None = None,
    ):
        self.tokenizer, self.sampler, self.verifier = tokenizer, sampler, verifier
        self.actor_revision = actor_revision
        self.config = config or ORSDConfig()
        self._requests = 0
        controls = ["</think>", "FINAL", " FINAL", "<|im_end|>", "<|endoftext|>", "<|image_pad|>"]
        ids = []
        for text in controls:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            if len(tokens) != 1:
                raise SafetyError("Search controls must be single tokens for this tokenizer")
            ids.append(tokens[0])
        self.blocked = tuple(ids[:5])
        self.logit_bias = {i: -100.0 for i in ids}

    def decode(self, ids):
        return self.tokenizer.decode(list(ids), skip_special_tokens=self.config.skip_special_tokens)

    def request(self, group, context, n, budget, bias):
        self._requests += 1
        if len(group.prompt_ids) + len(context) > self.config.max_prompt_tokens:
            raise SafetyError("Student/search prompt exceeds configured token width")
        return GenerationRequest(
            str(self._requests),
            group.prompt_id,
            group.prompt_ids,
            tuple(context),
            group.multimodal,
            n,
            min(budget, self.config.response_length),
            self.config.sampling,
            self.actor_revision,
            bias,
        )

    def draw(self, requests, retain_from=0):
        """All children are attributed to current actor/context; retry keeps only 9–16."""
        outputs = self.sampler.sample(requests)
        if len(outputs) != len(requests):
            raise SafetyError("Sampler request count differs")
        retained = []
        for request, children in zip(requests, outputs):
            if len(children) != request.n:
                raise SafetyError("Sampler child count differs")
            for child in children:
                if (
                    child.request_id != request.request_id
                    or child.prompt_ids != request.prompt_ids
                    or child.context_ids != request.context_ids
                    or child.actor_revision != request.actor_revision
                ):
                    raise SafetyError("Search child does not match request or current actor revision")
                if request.logit_bias is not None and set(child.token_ids[: request.max_tokens]) & set(
                    self.blocked
                ):
                    raise SafetyError("Blocked termination control sampled")
            retained.append(
                [replace(c, token_ids=c.token_ids[: request.max_tokens]) for c in children[retain_from:]]
            )
        return retained

    def augment(self, groups: Sequence[DirectGroup]) -> AugmentedBatch:
        """Build routing and one preferred terminal closure target per original prompt."""
        if len({g.prompt_id for g in groups}) != len(groups):
            raise SafetyError("Duplicate prompt identity")
        batch = AugmentedBatch()
        active = []
        direct_terminal = {}
        closure_states = {}
        for group in groups:
            if len(group.rollout_ids) != 8:
                raise SafetyError("Direct rollout count differs")
            if not all_wrong(group.answer_correct):
                batch.retained_direct.append(group)
                continue
            try:
                graph = self.verifier.build_graph(group.payload)
            except OntologyGraphError:
                batch.skipped.append({"prompt_id": group.prompt_id, "reason": "graph_error"})
                continue
            aligners = [
                TokenAligner(self.tokenizer, list(ids), self.config.skip_special_tokens)
                for ids in group.rollout_ids
            ]
            if self.config.enable_vtc:
                terminal = []
                for index, ids in enumerate(group.rollout_ids):
                    p = vtc.terminal_prefix(
                        list(ids), self.tokenizer, self.config.skip_special_tokens, graph, self.verifier
                    )
                    if p is not None:
                        terminal.append((len(p[0]), index, p))
                if terminal:
                    _, _, p = min(terminal, key=lambda x: x[:2])
                    direct_terminal[group.prompt_id] = (group, graph, p[0], p[1], p[2], 0)
            choice = self.verifier.select_reset_state(
                [a.text for a in aligners], graph, [a.tokens_for_prefix for a in aligners]
            )
            if choice.skip_reason:
                batch.skipped.append({"prompt_id": group.prompt_id, "reason": choice.skip_reason})
                continue
            state = choice.state
            ids = list(group.rollout_ids[choice.rollout_index][: state.token_len])
            text = self.decode(ids)
            self.require_prefix(text)
            if self.verifier.verify_text(text, graph).closed != state.closed:
                raise SafetyError("Prefix did not reverify")
            active.append(Branch(group, graph, state, state, ids, text))
        while active:
            work = []
            requests = []
            for branch in active:
                for relation in sorted({branch.graph.nodes[n].label for n in branch.state.frontier}):
                    budget = self.config.budget(relation)
                    if budget is None:
                        batch.skipped.append(
                            {
                                "prompt_id": branch.group.prompt_id,
                                "reason": "unsupported_search_relation",
                                "relation": relation,
                            }
                        )
                        continue
                    signal = signal_text(branch.prefix_text, relation, self.config)
                    if leakage_audit(signal, branch.payload, branch.graph):
                        raise SafetyError("Operation hint leaks privileged values")
                    signal_ids = self.tokenizer.encode(signal, add_special_tokens=False)
                    work.append((branch, relation, signal_ids))
                    requests.append(
                        self.request(branch.group, branch.model_ids + signal_ids, 8, budget, self.logit_bias)
                    )
            if not work:
                break
            batch.search_requests.extend(requests)
            first = self.draw(requests)
            evaluated = []
            sources = []
            for (branch, relation, signal), children in zip(work, first):
                evaluated.append(
                    [
                        evaluate_teacher_chunk(
                            branch,
                            signal,
                            list(c.token_ids),
                            self.tokenizer,
                            relation,
                            self.config.skip_special_tokens,
                            self.verifier,
                            self.blocked,
                        )
                        for c in children
                    ]
                )
                sources.append([(c, i + 1) for i, c in enumerate(children)])
            retry = [k for k, items in enumerate(evaluated) if not any(c["E"] == 1 for c in items)]
            if retry:
                more = [replace(requests[k], n=16) for k in retry]
                batch.search_requests.extend(more)
                second = self.draw(more, 8)
                for k, children in zip(retry, second):
                    branch, relation, signal = work[k]
                    evaluated[k] += [
                        evaluate_teacher_chunk(
                            branch,
                            signal,
                            list(c.token_ids),
                            self.tokenizer,
                            relation,
                            self.config.skip_special_tokens,
                            self.verifier,
                            self.blocked,
                        )
                        for c in children
                    ]
                    sources[k] += [(c, i + 9) for i, c in enumerate(children)]
            advancing = defaultdict(list)
            for (branch, relation, signal), items, source in zip(work, evaluated, sources):
                pick = select_candidate(items)
                if pick is None:
                    continue
                c = items[pick]
                sample, child = source[pick]
                tokens = list(sample.token_ids[: c["cut_tokens"]])
                target = self.target(
                    branch.group,
                    branch.model_ids,
                    tokens,
                    "route",
                    relation,
                    sample,
                    child,
                    branch.level,
                    1,
                    tuple(signal),
                )
                self.require_student(target)
                batch.targets.append(target)
                advancing[id(branch)].append((c, tokens))
            next_active = []
            # Keep every accepted relation target; one best branch continues progressively.
            for branch in active:
                accepted = advancing.get(id(branch))
                if not accepted:
                    continue
                c, tokens = min(accepted, key=lambda x: (-x[0]["advance"], x[0]["cut_tokens"]))
                ids = branch.model_ids + tokens
                text = self.decode(ids)
                self.require_prefix(text)
                closed = frozenset(c["new_closed"])
                if self.verifier.verify_text(text, branch.graph).closed != closed:
                    raise SafetyError("Accepted transition did not reverify")
                advance = self.verifier.depth_advance(branch.original_state, closed, branch.graph)
                state = PrefixState(
                    len(text), closed, branch.graph.frontier(closed), branch.graph.rank(closed), len(ids)
                )
                if self.config.enable_vtc and vtc.terminal_trigger(closed, branch.graph):
                    closure_states.setdefault(
                        branch.group.prompt_id,
                        (branch.group, branch.graph, ids, text, closed, branch.level + 1),
                    )
                    continue
                if not state.frontier or advance >= self.config.max_depth_advance:
                    continue
                budgets = [self.config.budget(branch.graph.nodes[n].label) for n in state.frontier]
                budget = max((b for b in budgets if b is not None), default=0)
                if len(ids) + budget > self.config.response_length:
                    continue
                next_active.append(
                    Branch(
                        branch.group, branch.graph, branch.original_state, state, ids, text, branch.level + 1
                    )
                )
            active = next_active
        if self.config.enable_vtc:
            for key, state in direct_terminal.items():
                closure_states.setdefault(key, state)
            self.closure_search(list(closure_states.values()), batch)
        if not batch.retained_direct:
            raise SafetyError("No retained direct group; no SD-only update or denominator fallback")
        return batch

    def require_prefix(self, text):
        if "</think>" in text or self.verifier._FINAL_LINE.search(text):
            raise SafetyError("Terminal marker in routing prefix")

    def require_student(self, target):
        text = self.decode(target.context_ids + target.target_ids)
        if SIGNAL_MARKER.lower() in text.lower() or FORBIDDEN_CONTEXT[1] in text:
            raise SafetyError("Teacher control in student text")
        if target.channel == "route" and set(target.target_ids) & set(self.blocked):
            raise SafetyError("Termination control in routing target")

    def target(self, group, context, tokens, channel, relation, sample, child, level, evidence, signal=()):
        text = self.decode(context)
        target_text = (
            self.decode(list(context) + list(tokens))[len(text) :]
            if channel == "route"
            else self.decode(tokens)
        )
        return DistillationTarget(
            group.prompt_id,
            tuple(context),
            tuple(tokens),
            channel,
            relation,
            self.actor_revision,
            sample.request_id,
            child,
            level,
            evidence,
            text,
            target_text,
            signal,
        )

    def closure_search(self, states, batch):
        if not states:
            return
        requests = []
        for group, graph, context, text, closed, level in states:
            vtc.require_signal_free(self.tokenizer.decode(context, skip_special_tokens=False))
            if not vtc.terminal_trigger(closed, graph):
                raise SafetyError("Non-terminal closure request")
            requests.append(self.request(group, context, 8, self.config.closure_k, None))
        batch.search_requests.extend(requests)
        first = self.draw(requests)
        items = []
        sources = []
        for state, children in zip(states, first):
            group, graph, context, text, closed, level = state
            items.append(
                [
                    vtc.evaluate_closure(
                        text,
                        closed,
                        list(c.token_ids),
                        self.tokenizer,
                        self.config.skip_special_tokens,
                        group.payload,
                        graph,
                        self.verifier,
                    )
                    for c in children
                ]
            )
            sources.append([(c, i + 1) for i, c in enumerate(children)])
        retry = [k for k, v in enumerate(items) if not any(c["Z"] for c in v)]
        if retry:
            more = [replace(requests[k], n=16) for k in retry]
            batch.search_requests.extend(more)
            second = self.draw(more, 8)
            for k, children in zip(retry, second):
                group, graph, context, text, closed, level = states[k]
                items[k] += [
                    vtc.evaluate_closure(
                        text,
                        closed,
                        list(c.token_ids),
                        self.tokenizer,
                        self.config.skip_special_tokens,
                        group.payload,
                        graph,
                        self.verifier,
                    )
                    for c in children
                ]
                sources[k] += [(c, i + 9) for i, c in enumerate(children)]
        for state, candidates, source in zip(states, items, sources):
            pick = vtc.select_closure(candidates)
            if pick is None:
                continue
            group, graph, context, text, closed, level = state
            c = candidates[pick]
            sample, child = source[pick]
            target = self.target(
                group,
                context,
                list(sample.token_ids[: c["closure_tokens"]]),
                "close",
                "terminal_closure",
                sample,
                child,
                level,
                1,
            )
            self.require_student(target)
            batch.targets.append(target)
