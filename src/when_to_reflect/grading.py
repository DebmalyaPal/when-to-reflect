"""Conservative answer extraction and grading for local MATH smoke tests.

This initial grader compares normalized surface forms only. Final experiments
may replace it with a stronger symbolic or verifier-based grader.
"""

import re


def _extract_braced(text: str, opening_brace_index: int) -> str | None:
    """Extract a balanced braced expression beginning at ``opening_brace_index``."""
    depth = 0
    for index in range(opening_brace_index, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[opening_brace_index + 1 : index]
    return None


def extract_final_boxed_answer(text: str) -> str | None:
    """Return the final balanced ``\\boxed{...}`` expression in text."""
    answers = []
    search_start = 0
    marker = r"\boxed{"
    while (marker_index := text.find(marker, search_start)) != -1:
        opening_brace_index = marker_index + len(marker) - 1
        answer = _extract_braced(text, opening_brace_index)
        if answer is not None:
            answers.append(answer)
        search_start = opening_brace_index + 1
    return answers[-1] if answers else None


def extract_reference_answer(reference_solution: str) -> str | None:
    """Extract the final boxed MATH reference answer when present."""
    return extract_final_boxed_answer(reference_solution)


def extract_generated_answer(generated_text: str) -> str | None:
    """Extract a generated final answer without guessing from arbitrary text."""
    boxed_answer = extract_final_boxed_answer(generated_text)
    if boxed_answer is not None:
        return boxed_answer

    matches = list(re.finditer(r"\bfinal\s+answer\s*:\s*([^\n]+)", generated_text, re.IGNORECASE))
    return matches[-1].group(1).strip() if matches else None


def normalize_answer(answer: str) -> str:
    """Normalize harmless presentation differences for straightforward answers."""
    normalized = answer.strip()
    for opening, closing in (("$", "$"), (r"\(", r"\)"), (r"\[", r"\]")):
        if normalized.startswith(opening) and normalized.endswith(closing):
            normalized = normalized[len(opening) : -len(closing)].strip()
    normalized = normalized.replace(r"\left", "").replace(r"\right", "")
    normalized = normalized.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    normalized = re.sub(r"\s+", "", normalized).rstrip(".,;")
    return normalized


def answers_match(reference_answer: str | None, generated_answer: str | None) -> bool:
    """Compare extracted answers without symbolic simplification."""
    return (
        reference_answer is not None
        and generated_answer is not None
        and normalize_answer(reference_answer) == normalize_answer(generated_answer)
    )


def score_answer(
    reference_answer: str | None,
    generated_answer: str | None,
    generation_complete: bool,
) -> bool | None:
    """Return no score for incomplete generations; otherwise compare answers."""
    if not generation_complete:
        return None
    return answers_match(reference_answer, generated_answer)
