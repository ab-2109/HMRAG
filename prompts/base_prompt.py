"""
Prompt construction for the HM-RAG Decision Agent (Layer 3).

Builds few-shot prompts from ScienceQA problems for the consistency-voting
and expert-refinement stages described in Section 3.4 of the paper.

Supported prompt format codes (set via ``--prompt_format``):

    Input side          Output side
    ──────────          ───────────
    CQM  = Context → Question → Options       A   = Answer only
    QCM  = Question → Context → Options       AL  = Answer + Solution
    QCML = QCM + Lecture in input              AE  = Answer + Lecture
    QCME = QCM + Solution in input             ALE = Answer + Lecture + Solution
    QCMLE = QCM + Lecture + Solution           AEL = Answer + Solution + Lecture
    QCLM  = Q → Context → Lecture → Options   LA  = Lecture → Answer
    QCEM  = Q → Context → Solution → Options  EA  = Solution → Answer
    QCLEM = Q → C → Lecture + Solution → Opt  LEA = Lecture + Solution → Answer
                                               ELA = Solution + Lecture → Answer

Default is ``CQM-A`` (Context-Question-Options → Answer).
"""

from typing import Any, Dict, List, Optional

# ------------------------------------------------------------------
# System prompt prepended to every few-shot prompt
# ------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert science question answering assistant. "
    "You are given a multiple-choice science question along with optional "
    "context (hints, image captions, or background knowledge). "
    "Select the single best answer from the provided options. "
    "Always respond in the format: 'The answer is X.' where X is one of "
    "the option letters (A, B, C, D, or E)."
)


# ------------------------------------------------------------------
# Field extractors
# ------------------------------------------------------------------

def get_question_text(problem: Dict[str, Any]) -> str:
    """Return the question string from a problem dict."""
    return problem['question']


def get_context_text(problem: Dict[str, Any], use_caption: bool = False) -> str:
    """Combine hint and (optionally) caption into a context string.

    Returns ``"N/A"`` when no context is available so the LLM sees a
    consistent placeholder rather than an empty field.
    """
    hint = problem.get('hint', '') or ''
    caption = problem.get('caption', '') if use_caption else ''
    context = ' '.join([hint, caption]).strip()
    return context if context else 'N/A'


def get_choice_text(problem: Dict[str, Any], options: List[str]) -> str:
    """Format choices as ``(A) … (B) … (C) …``."""
    choices = problem['choices']
    parts = [f"({options[i]}) {c}" for i, c in enumerate(choices)]
    return ' '.join(parts)


def get_answer(problem: Dict[str, Any], options: List[str]) -> str:
    """Return the letter label of the correct answer."""
    return options[problem['answer']]


def get_lecture_text(problem: Dict[str, Any]) -> str:
    """Return the lecture (background knowledge) text."""
    return (problem.get('lecture', '') or '').replace('\n', ' ').strip()


def get_solution_text(problem: Dict[str, Any]) -> str:
    """Return the solution / explanation text."""
    return (problem.get('solution', '') or '').replace('\n', ' ').strip()


# ------------------------------------------------------------------
# Single-example formatter
# ------------------------------------------------------------------

def _format_input(
    fmt: str,
    question: str,
    context: str,
    choice: str,
    lecture: str,
    solution: str,
) -> str:
    """Build the *input* portion of one example according to *fmt*."""
    templates = {
        'CQM':   f"Context: {context}\nQuestion: {question}\nOptions: {choice}\n",
        'QCM':   f"Question: {question}\nContext: {context}\nOptions: {choice}\n",
        'QCML':  f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {lecture}\n",
        'QCME':  f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {solution}\n",
        'QCMLE': f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {lecture} {solution}\n",
        'QCLM':  f"Question: {question}\nContext: {context}\nBECAUSE: {lecture}\nOptions: {choice}\n",
        'QCEM':  f"Question: {question}\nContext: {context}\nBECAUSE: {solution}\nOptions: {choice}\n",
        'QCLEM': f"Question: {question}\nContext: {context}\nBECAUSE: {lecture} {solution}\nOptions: {choice}\n",
    }
    return templates.get(fmt, templates['CQM'])


def _format_output(
    fmt: str,
    answer: str,
    lecture: str,
    solution: str,
    is_test: bool,
) -> str:
    """Build the *output* portion of one example according to *fmt*.

    For the test example the output is just ``"Answer:"`` so the LLM
    completes it.
    """
    if is_test:
        return "Answer:"

    templates = {
        'A':   f"Answer: The answer is {answer}.",
        'AL':  f"Answer: The answer is {answer}. BECAUSE: {solution}",
        'AE':  f"Answer: The answer is {answer}. BECAUSE: {lecture}",
        'ALE': f"Answer: The answer is {answer}. BECAUSE: {lecture} {solution}",
        'AEL': f"Answer: The answer is {answer}. BECAUSE: {solution} {lecture}",
        'LA':  f"Answer: {lecture} The answer is {answer}.",
        'EA':  f"Answer: {solution} The answer is {answer}.",
        'LEA': f"Answer: {lecture} {solution} The answer is {answer}.",
        'ELA': f"Answer: {solution} {lecture} The answer is {answer}.",
    }
    return templates.get(fmt, templates['A'])


def create_one_example(
    prompt_format: str,
    question: str,
    context: str,
    choice: str,
    answer: str,
    lecture: str,
    solution: str,
    test_example: bool = True,
) -> str:
    """Create a single formatted example (train or test).

    Args:
        prompt_format: A string like ``"CQM-A"`` where the left side
            controls input layout and the right side controls output.
        question, context, choice, answer, lecture, solution:
            Pre-extracted fields from a ScienceQA problem.
        test_example: If ``True``, the output is left open for the
            model to complete.

    Returns:
        A single formatted example string.
    """
    input_fmt, output_fmt = prompt_format.split('-', 1)

    text_in = _format_input(input_fmt, question, context, choice, lecture, solution)
    text_out = _format_output(output_fmt, answer, lecture, solution, is_test=test_example)

    text = (text_in + text_out).replace('  ', ' ').strip()
    # Remove dangling "BECAUSE:" when no explanation follows
    if text.endswith('BECAUSE:'):
        text = text.replace('BECAUSE:', '').strip()
    return text


# ------------------------------------------------------------------
# Helper: extract all fields from one problem
# ------------------------------------------------------------------

def _extract_fields(
    problem: Dict[str, Any],
    options: List[str],
    use_caption: bool,
) -> Dict[str, str]:
    """Pull every prompt-relevant field out of a problem dict."""
    return {
        'question': get_question_text(problem),
        'context':  get_context_text(problem, use_caption),
        'choice':   get_choice_text(problem, options),
        'answer':   get_answer(problem, options),
        'lecture':  get_lecture_text(problem),
        'solution': get_solution_text(problem),
    }


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------

def build_prompt(
    problems: Dict[str, Any],
    shot_qids: List[str],
    test_qid: str,
    args: Any,
) -> str:
    """Build a complete few-shot prompt for the Decision Agent.

    Structure::

        <system instruction>

        <train example 1>

        <train example 2>

        ...

        <test example (answer left open)>

    Args:
        problems:  Full dataset dict  ``{qid: problem_dict, …}``.
        shot_qids: Question IDs to use as few-shot demonstrations.
        test_qid:  The question ID being answered.
        args:      Config namespace with ``use_caption``, ``options``,
                   ``prompt_format``.

    Returns:
        The assembled prompt string ready to be sent to the LLM.
    """
    use_caption = getattr(args, 'use_caption', False)
    options = getattr(args, 'options', ['A', 'B', 'C', 'D', 'E'])
    prompt_format = getattr(args, 'prompt_format', 'CQM-A')

    parts: List[str] = [SYSTEM_PROMPT]

    # ── Few-shot training examples ──────────────────────────────────
    for qid in shot_qids:
        fields = _extract_fields(problems[qid], options, use_caption)
        example = create_one_example(prompt_format, **fields, test_example=False)
        parts.append(example)

    # ── Test example (answer to be completed by the LLM) ────────────
    fields = _extract_fields(problems[test_qid], options, use_caption)
    test_example = create_one_example(prompt_format, **fields, test_example=True)
    parts.append(test_example)

    return '\n\n'.join(parts)