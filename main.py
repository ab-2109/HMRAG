"""
HM-RAG — Hierarchical Multi-Agent Multimodal RAG.

Entry point for evaluating the HM-RAG framework on ScienceQA.

Usage:
    python main.py \\
        --data_root ./dataset/ScienceQA/data/scienceqa \\
        --image_root ./dataset/ScienceQA/data/scienceqa \\
        --llm_model_name qwen2.5:1.5b \\
        --serper_api_key YOUR_KEY \\
        --test_number 100
"""

import json
import logging
import os
import random
import time
import argparse
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

from agents.multi_retrieval_agents import MRetrievalAgent
from preprocessing.build_knowledge_base import KnowledgeBaseBuilder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-28s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ======================================================================
# Argument parsing
# ======================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="HM-RAG: Hierarchical Multi-Agent Multimodal RAG for ScienceQA",
    )

    # ── Data paths ───────────────────────────────────────────────────
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to root data directory containing problems.json and pid_splits.json')
    parser.add_argument('--image_root', type=str, default='',
                        help='Path to image root directory')
    parser.add_argument('--output_root', type=str, default='outputs',
                        help='Path to output directory')
    parser.add_argument('--caption_file', type=str, default='',
                        help='Path to captions JSON file')

    # ── Evaluation settings ──────────────────────────────────────────
    parser.add_argument('--options', type=str, nargs='+',
                        default=['A', 'B', 'C', 'D', 'E'],
                        help='Answer option letters')
    parser.add_argument('--test_split', type=str, default='test',
                        choices=['test', 'val', 'minival'],
                        help='Which data split to evaluate on')
    parser.add_argument('--test_number', type=int, default=-1,
                        help='Number of test problems to run (-1 = all)')
    parser.add_argument('--shot_number', type=int, default=0,
                        help='Number of n-shot training examples for prompting')
    parser.add_argument('--shot_qids', type=int, nargs='+', default=None,
                        help='Specific question IDs for shot examples')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--label', type=str, default='exp',
                        help='Experiment label for output file naming')
    parser.add_argument('--debug', action='store_true',
                        help='Debug mode: run only first 10 examples')
    parser.add_argument('--save_every', type=int, default=50,
                        help='Save results every N examples')
    parser.add_argument('--use_caption', action='store_true',
                        help='Whether to use image captions as context')

    # ── Prompt format ────────────────────────────────────────────────
    parser.add_argument('--prompt_format', type=str, default='CQM-A',
                        choices=[
                            'CQM-A', 'CQM-LA', 'CQM-EA', 'CQM-LEA', 'CQM-ELA',
                            'CQM-AL', 'CQM-AE', 'CQM-ALE',
                            'QCM-A', 'QCM-LA', 'QCM-EA', 'QCM-LEA', 'QCM-ELA',
                            'QCM-AL', 'QCM-AE', 'QCM-ALE',
                            'QCML-A', 'QCME-A', 'QCMLE-A',
                            'QCLM-A', 'QCEM-A', 'QCLEM-A', 'QCML-AE',
                        ],
                        help='Prompt format template (see prompts/base_prompt.py)')

    # ── Model / retrieval settings ───────────────────────────────────
    parser.add_argument('--working_dir', type=str, default='./lightrag_workdir',
                        help='Working directory for LightRAG')
    parser.add_argument('--llm_model_name', type=str, default='qwen2.5:1.5b',
                        help='Ollama LLM model name for text generation')
    parser.add_argument('--web_llm_model_name', type=str, default='qwen2.5:1.5b',
                        help='Ollama LLM model name for web retrieval answer synthesis')
    parser.add_argument('--mode', type=str, default='hybrid',
                        help='LightRAG retrieval mode')
    parser.add_argument('--top_k', type=int, default=4,
                        help='Number of retrieval results per agent')
    parser.add_argument('--serper_api_key', type=str, default='',
                        help='API key for Serper web search (https://serper.dev)')
    parser.add_argument('--ollama_base_url', type=str,
                        default='http://localhost:11434',
                        help='Base URL for Ollama server')
    parser.add_argument('--hf_token', type=str, default='',
                        help='Hugging Face access token for downloading gated models')

    return parser.parse_args()


# ======================================================================
# Data loading
# ======================================================================

def load_data(args: argparse.Namespace) -> Tuple[Dict, List[str], List[str]]:
    """Load ScienceQA problems, splits, and optional captions.

    Returns:
        (problems_dict, test_qids, shot_qids)
    """
    problems_path = os.path.join(args.data_root, 'problems.json')
    splits_path = os.path.join(args.data_root, 'pid_splits.json')

    with open(problems_path, 'r') as f:
        problems = json.load(f)
    with open(splits_path, 'r') as f:
        pid_splits = json.load(f)

    # Merge captions into problems
    captions: Dict[str, str] = {}
    if args.caption_file and os.path.exists(args.caption_file):
        with open(args.caption_file, 'r') as f:
            captions = json.load(f).get('captions', {})
    for qid in problems:
        problems[qid]['caption'] = captions.get(qid, '')

    # Select test question IDs
    qids = pid_splits[args.test_split]
    if args.test_number > 0:
        qids = qids[:args.test_number]
    if args.debug:
        qids = qids[:10]
    logger.info("Test problems: %d (split=%s)", len(qids), args.test_split)

    # Select few-shot example IDs
    train_qids = pid_splits.get('train', [])
    shot_qids = _select_shot_qids(args, train_qids)
    logger.info("Shot example QIDs: %s", shot_qids)

    return problems, qids, shot_qids


def _select_shot_qids(
    args: argparse.Namespace,
    train_qids: List[str],
) -> List[str]:
    """Pick few-shot example question IDs from the training set."""
    if args.shot_qids is not None:
        # User specified exact IDs
        shot_qids = [str(qid) for qid in args.shot_qids]
        for qid in shot_qids:
            if qid not in train_qids:
                raise ValueError(f"Shot QID {qid} not found in training set")
        return shot_qids

    if args.shot_number > 0 and train_qids:
        n = min(args.shot_number, len(train_qids))
        return random.sample(train_qids, n)

    return []


def _get_train_qids(args: argparse.Namespace) -> List[str]:
    """Load training-split QIDs for knowledge-base construction.

    These are the problems whose textual content (hint, lecture,
    solution, image captions) gets indexed into LightRAG during
    the Phase 1 preprocessing step.
    """
    splits_path = os.path.join(args.data_root, 'pid_splits.json')
    with open(splits_path, 'r') as f:
        pid_splits = json.load(f)
    return pid_splits.get('train', [])


# ======================================================================
# HuggingFace token setup
# ======================================================================

def _setup_hf_token(token: str) -> None:
    """Set HF token in environment and login to the Hub."""
    if not token:
        return
    os.environ['HF_TOKEN'] = token
    os.environ['HUGGING_FACE_HUB_TOKEN'] = token
    try:
        from huggingface_hub import login
        login(token=token)
        logger.info("✓ Logged in to Hugging Face Hub")
    except Exception as e:
        logger.warning("Could not login to HF Hub: %s", e)


# ======================================================================
# Result saving
# ======================================================================

def _save_results(
    result_file: str,
    results: Dict[str, Any],
    outputs: Dict[str, Any],
) -> None:
    """Write results and outputs to JSON files."""
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Also save the raw agent outputs for analysis
    output_file = result_file.replace('.json', '_outputs.json')
    try:
        with open(output_file, 'w') as f:
            json.dump(outputs, f, indent=2, default=str)
    except Exception as e:
        logger.warning("Could not save outputs file: %s", e)


# ======================================================================
# Main evaluation loop
# ======================================================================

def main() -> None:
    """Run HM-RAG evaluation on ScienceQA."""
    args = parse_args()

    logger.info("=" * 60)
    logger.info("HM-RAG Evaluation")
    logger.info("=" * 60)
    logger.info("Arguments:\n%s", json.dumps(vars(args), indent=2, sort_keys=False))

    random.seed(args.seed)

    # ── Setup ────────────────────────────────────────────────────────
    _setup_hf_token(args.hf_token)
    problems, qids, shot_qids = load_data(args)

    os.makedirs(args.output_root, exist_ok=True)
    result_file = os.path.join(
        args.output_root,
        f"{args.label}_{args.test_split}.json",
    )

    # ── Phase 1: Build Knowledge Base (Section 3.1) ──────────────────
    # Index training-split problems into LightRAG so vector/graph
    # retrieval agents have a populated database to query.
    # Uses the same working_dir that the retrieval agents will read from.
    train_qids = _get_train_qids(args)
    builder = KnowledgeBaseBuilder(args)
    builder.build(problems, train_qids)

    # ── Initialise agent ─────────────────────────────────────────────
    agent = MRetrievalAgent(args)

    # ── Evaluation loop ──────────────────────────────────────────────
    correct = 0
    results: Dict[str, int] = {}
    outputs: Dict[str, Any] = {}
    failed: List[str] = []
    total_time = 0.0

    for i, qid in enumerate(tqdm(qids, desc="Evaluating")):
        answer = problems[qid]['answer']

        t0 = time.time()
        try:
            final_ans, all_messages = agent.predict(problems, shot_qids, qid)
        except Exception as e:
            logger.error("Error on qid %s: %s", qid, e)
            final_ans = -1
            all_messages = [str(e)]
        elapsed = time.time() - t0
        total_time += elapsed

        results[qid] = final_ans
        outputs[qid] = all_messages

        if final_ans == answer:
            correct += 1
        else:
            failed.append(qid)

        # Periodic logging
        if (i + 1) % 10 == 0:
            acc_so_far = correct / (i + 1)
            logger.info(
                "Progress %d/%d | accuracy=%.4f | last_qid=%s (%.1fs)",
                i + 1, len(qids), acc_so_far, qid, elapsed,
            )

        # Periodic save
        if (i + 1) % args.save_every == 0:
            _save_results(result_file, results, outputs)
            logger.info("Checkpoint saved after %d examples", i + 1)

    # ── Final save and report ────────────────────────────────────────
    _save_results(result_file, results, outputs)

    total = len(qids)
    accuracy = correct / total if total > 0 else 0.0
    avg_time = total_time / total if total > 0 else 0.0

    logger.info("=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info("Results saved to:  %s", result_file)
    logger.info("Correct:           %d / %d", correct, total)
    logger.info("Accuracy:          %.4f", accuracy)
    logger.info("Total time:        %.1fs", total_time)
    logger.info("Avg time/question: %.2fs", avg_time)
    logger.info("Failed QIDs (%d):  %s", len(failed), failed[:20])
    if len(failed) > 20:
        logger.info("  ... and %d more", len(failed) - 20)


if __name__ == "__main__":
    main()


