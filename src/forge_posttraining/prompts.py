"""Frozen direct instruction and operation-only teacher signal."""

TASK_INSTRUCTION = (
    "Reason about the mechanical drawing before answering.\n"
    "Your reasoning MUST be enclosed within <think> and </think> tags.\n"
    "After </think>, output exactly one final-answer line in this form:\n"
    "FINAL: <answer>\nUse only information supported by the drawing."
)
SIGNAL_FORMAT = "Next reasoning operation: {relation}. Continue reasoning."
SIGNAL_MARKER = "Next reasoning operation"
# Reserved historical input control name: rejected, never generated or routed.
FORBIDDEN_CONTEXT = (SIGNAL_MARKER, "finalize_answer")


def task_prompt(question: str) -> str:
    """Render the same instruction as the inspected trimmed Jinja question template."""
    return question.strip() + "\n\n" + TASK_INSTRUCTION
