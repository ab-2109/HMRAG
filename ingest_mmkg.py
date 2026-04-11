"""LightRAG ingestion for HM-RAG (OpenAI + Neo4j + Qdrant + SmolVLM).

Why this script exists:
- HM-RAG retrieval calls LightRAG.query(...).
- Before query-time retrieval can work, LightRAG storage must be populated.
- This script builds multimodal fused documents from ScienceQA problems and inserts
  them into LightRAG, which then handles entity extraction + graph/vector writes.

Stack assumptions:
- LLM for extraction/indexing: OpenAI (default: gpt-4o-mini)
- Embeddings: text-embedding-3-small
- Graph storage: Neo4j
- Vector storage: Qdrant
- VLM: SmolVLM (default) or OpenAI vision
"""

import argparse
import asyncio
import inspect
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from openai import OpenAI
from tqdm import tqdm

from retrieval.lightrag_factory import create_lightrag_client


_SMOLVLM_CACHE = {
    "model_name": None,
    "processor": None,
    "model": None,
}


def _run_maybe_async(value):
    if inspect.isawaitable(value):
        return asyncio.run(value)
    return value


def _initialize_lightrag(rag) -> None:
    init_fn = getattr(rag, "initialize_storages", None)
    if callable(init_fn):
        _run_maybe_async(init_fn())

    # Some versions expose an explicit pipeline status init hook.
    pipeline_fn = getattr(rag, "initialize_pipeline_status", None)
    if callable(pipeline_fn):
        _run_maybe_async(pipeline_fn())


def _resolve_image_path(image_root: str, split: str, qid: str, image_name: str) -> str:
    if not image_name:
        return ""

    candidate = Path(image_root) / split / qid / image_name
    if candidate.exists():
        return str(candidate)

    candidate = Path(image_root) / qid / image_name
    if candidate.exists():
        return str(candidate)

    for s in ["train", "val", "test", "minival"]:
        candidate = Path(image_root) / s / qid / image_name
        if candidate.exists():
            return str(candidate)

    return ""


def _build_fused_text(problem: Dict, options: List[str], vision_text: str) -> str:
    q = problem.get("question", "")
    hint = problem.get("hint", "")
    cap = problem.get("caption", "")
    choices = problem.get("choices", [])
    choice_txt = " ".join([f"({options[i]}) {c}" for i, c in enumerate(choices) if i < len(options)])

    parts = [
        f"Question: {q}",
        f"Hint: {hint}" if hint else "",
        f"Caption: {cap}" if cap else "",
        f"Image-derived evidence: {vision_text}" if vision_text else "",
        f"Options: {choice_txt}" if choice_txt else "",
    ]
    return "\n".join([p for p in parts if p]).strip()


def _describe_image_openai(
    image_path: str,
    question: str,
    hint: str,
    caption: str,
    api_key: str,
    model: str,
    base_url: str = "",
) -> str:
    client = OpenAI(api_key=api_key, base_url=base_url or None)
    prompt = (
        "You are preparing evidence for multimodal QA retrieval. "
        "Describe only useful visual facts, objects, relations, text in image, "
        "and measurements if visible. Keep it concise and factual.\n\n"
        f"Question: {question}\n"
        f"Hint: {hint}\n"
        f"Caption: {caption}\n"
    )

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    import base64

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    resp = client.chat.completions.create(
        model=model,
        temperature=0.1,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                ],
            }
        ],
    )
    return (resp.choices[0].message.content or "").strip()


def _describe_image_smallvlm(image_path: str, question: str, hint: str, caption: str, model_name: str) -> str:
    from PIL import Image
    import torch
    from transformers import AutoProcessor

    global _SMOLVLM_CACHE
    if _SMOLVLM_CACHE["model"] is None or _SMOLVLM_CACHE["model_name"] != model_name:
        try:
            from transformers import AutoModelForVision2Seq as _AutoVisionModel
        except Exception:
            try:
                from transformers import AutoModelForImageTextToText as _AutoVisionModel
            except Exception as e:
                raise ImportError(
                    "SmolVLM auto model class not found in current transformers version. "
                    "Please upgrade transformers or use --vlm-provider openai."
                ) from e

        _SMOLVLM_CACHE["processor"] = AutoProcessor.from_pretrained(model_name, use_fast=True)
        _SMOLVLM_CACHE["model"] = _AutoVisionModel.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
        )
        _SMOLVLM_CACHE["model_name"] = model_name

    processor = _SMOLVLM_CACHE["processor"]
    model = _SMOLVLM_CACHE["model"]

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": (
                        "Describe the image in the best possibly way in 20 words. "
                        "Mention only facts visible in the image.\n\n"
                        f"Question: {question}\n"
                        f"Hint: {hint}\n"
                        f"Caption: {caption}\n"
                    ),
                },
            ],
        }
    ]

    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=[chat_text], images=[image], padding=True, return_tensors="pt")

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    generated_ids = model.generate(**inputs, max_new_tokens=256)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return (output_text[0] if output_text else "").strip()


def _select_qids(
    problems: Dict,
    split: str,
    dataset_path: str,
    pid_splits_path: str = "",
    start_index: int = 0,
    max_items: Optional[int] = None,
) -> List[str]:
    # 1) Explicit pid_splits path from user
    candidate_paths: List[Path] = []
    if pid_splits_path:
        candidate_paths.append(Path(pid_splits_path))

    # 2) Auto-discover sibling pid_splits.json next to problems.json
    candidate_paths.append(Path(dataset_path).parent / "pid_splits.json")

    for p in candidate_paths:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                pid_splits = json.load(f)
            raw = pid_splits.get(split, [])
            qids = [str(qid) for qid in raw if str(qid) in problems]
            if start_index > 0:
                qids = qids[start_index:]
            if max_items:
                qids = qids[:max_items]
            print(f"[ingest] Using pid_splits from {p} ({len(qids)} qids for split='{split}')")
            return qids

    # 3) Fallback to per-problem split field if present
    any_split_field = any("split" in p for p in problems.values())
    if any_split_field:
        qids = [qid for qid, p in problems.items() if p.get("split") == split]
        if start_index > 0:
            qids = qids[start_index:]
        if max_items:
            qids = qids[:max_items]
        print(f"[ingest] Using problem['split'] field ({len(qids)} qids for split='{split}')")
        return qids

    # 4) Last resort: ingest everything
    qids = list(problems.keys())
    if start_index > 0:
        qids = qids[start_index:]
    if max_items:
        qids = qids[:max_items]
    print(
        "[ingest] Warning: no pid_splits.json or problem['split'] found. "
        f"Falling back to all problems ({len(qids)} qids)."
    )
    return qids


def ingest_dataset(
    dataset_path: str,
    image_root: str,
    split: str,
    pid_splits_path: str,
    start_index: int,
    max_items: Optional[int],
    options: List[str],
    use_vlm: bool,
    vlm_provider: str,
    vlm_model_name: str,
    log_vlm: bool,
    vlm_log_max_chars: int,
    openai_api_key: str,
    openai_base_url: str,
    lightrag_client,
) -> None:
    with open(dataset_path, "r", encoding="utf-8") as f:
        problems = json.load(f)

    qids = _select_qids(
        problems=problems,
        split=split,
        dataset_path=dataset_path,
        pid_splits_path=pid_splits_path,
        start_index=start_index,
        max_items=max_items,
    )

    print(
        f"[ingest] {len(qids)} items selected for split='{split}'"
        + (f" starting from index {start_index}" if start_index else "")
        + (" (capped by --max-items)" if max_items else "")
    )

    stats = {
        "total": 0,
        "with_image": 0,
        "vlm_success": 0,
        "inserted": 0,
        "failed": 0,
    }

    for qid in tqdm(qids, desc=f"Ingest {split}"):
        p = problems[qid]
        stats["total"] += 1

        question = p.get("question", "")
        hint = p.get("hint", "")
        caption = p.get("caption", "")
        image_name = p.get("image", "")
        answer_idx = int(p.get("answer", 0))

        img_path = _resolve_image_path(image_root, split, qid, image_name)
        if img_path:
            stats["with_image"] += 1

        vision_text = ""
        if use_vlm and img_path:
            try:
                if vlm_provider == "smallvlm":
                    vision_text = _describe_image_smallvlm(
                        image_path=img_path,
                        question=question,
                        hint=hint,
                        caption=caption,
                        model_name=vlm_model_name,
                    )
                elif vlm_provider == "openai":
                    vision_text = _describe_image_openai(
                        image_path=img_path,
                        question=question,
                        hint=hint,
                        caption=caption,
                        api_key=openai_api_key,
                        base_url=openai_base_url,
                        model=vlm_model_name,
                    )

                if vision_text:
                    stats["vlm_success"] += 1
                    if log_vlm:
                        snippet = vision_text[:vlm_log_max_chars]
                        if len(vision_text) > vlm_log_max_chars:
                            snippet += "..."
                        tqdm.write(f"[VLM OUTPUT] qid={qid}: {snippet}")
            except Exception as e:
                tqdm.write(f"[VLM ERROR] qid={qid}: {e}")

        fused_text = _build_fused_text(p, options, vision_text)
        if not fused_text:
            continue

        # Keep per-problem metadata in-text for reproducible retrieval traces.
        doc = (
            f"QID: {qid}\n"
            f"Split: {split}\n"
            f"AnswerIndex: {answer_idx}\n"
            f"ImagePath: {img_path}\n"
            f"{fused_text}"
        )

        try:
            _run_maybe_async(lightrag_client.insert(doc))
            stats["inserted"] += 1
        except Exception as e:
            stats["failed"] += 1
            tqdm.write(f"[INSERT ERROR] qid={qid}: {e}")

    print(f"\n[ingest] Done. Stats: {stats}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest ScienceQA into LightRAG using OpenAI+Neo4j+Qdrant with SmolVLM."
    )

    parser.add_argument("--dataset", required=True, help="Path to ScienceQA problems.json")
    parser.add_argument(
        "--pid-splits",
        default="",
        help="Optional path to pid_splits.json. If omitted, script auto-discovers sibling file.",
    )
    parser.add_argument("--image-root", required=True, help="ScienceQA image root")
    parser.add_argument("--split", default="train", choices=["train", "val", "test", "minival"])
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-items", type=int, default=None)

    parser.add_argument("--working-dir", default="./lightrag_workdir")

    parser.add_argument("--openai-api-key", default=os.getenv("OPENAI_API_KEY", ""))
    parser.add_argument("--openai-base-url", default=os.getenv("OPENAI_BASE_URL", ""))
    parser.add_argument("--llm-model-name", default="gpt-4o-mini")
    parser.add_argument("--embedding-model-name", default="text-embedding-3-small")

    parser.add_argument("--neo4j-uri", default=os.getenv("NEO4J_URI", ""))
    parser.add_argument("--neo4j-username", default=os.getenv("NEO4J_USERNAME", ""))
    parser.add_argument("--neo4j-password", default=os.getenv("NEO4J_PASSWORD", ""))
    parser.add_argument("--neo4j-database", default=os.getenv("NEO4J_DATABASE", "neo4j"))

    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", ""))
    parser.add_argument("--qdrant-api-key", default=os.getenv("QDRANT_API_KEY", ""))
    parser.add_argument("--qdrant-collection", default="hmrag_chunks")

    parser.add_argument("--use-vlm", action="store_true")
    parser.add_argument(
        "--vlm-provider",
        choices=["smallvlm", "openai"],
        default="smallvlm",
        help="Use SmolVLM locally on GPU or OpenAI vision.",
    )
    parser.add_argument(
        "--vlm-model-name",
        default="HuggingFaceTB/SmolVLM-Instruct",
        help="If provider=smallvlm, Hugging Face model ID. If provider=openai, OpenAI vision model name.",
    )
    parser.add_argument(
        "--log-vlm",
        action="store_true",
        help="Log VLM output snippets during ingestion.",
    )
    parser.add_argument(
        "--vlm-log-max-chars",
        type=int,
        default=240,
        help="Max chars to print per VLM output when --log-vlm is enabled.",
    )

    args = parser.parse_args()

    if not args.openai_api_key:
        raise ValueError("Missing OpenAI API key. Set --openai-api-key or OPENAI_API_KEY.")

    config = argparse.Namespace(
        working_dir=args.working_dir,
        openai_api_key=args.openai_api_key,
        openai_base_url=args.openai_base_url,
        llm_model_name=args.llm_model_name,
        embedding_model_name=args.embedding_model_name,
        neo4j_uri=args.neo4j_uri,
        neo4j_username=args.neo4j_username,
        neo4j_password=args.neo4j_password,
        neo4j_database=args.neo4j_database,
        qdrant_url=args.qdrant_url,
        qdrant_api_key=args.qdrant_api_key,
        qdrant_collection=args.qdrant_collection,
    )

    lightrag_client = create_lightrag_client(config)
    _initialize_lightrag(lightrag_client)

    options = ["A", "B", "C", "D", "E"]
    ingest_dataset(
        dataset_path=args.dataset,
        image_root=args.image_root,
        split=args.split,
        pid_splits_path=args.pid_splits,
        start_index=args.start_index,
        max_items=args.max_items,
        options=options,
        use_vlm=args.use_vlm,
        vlm_provider=args.vlm_provider,
        vlm_model_name=args.vlm_model_name,
        log_vlm=args.log_vlm,
        vlm_log_max_chars=args.vlm_log_max_chars,
        openai_api_key=args.openai_api_key,
        openai_base_url=args.openai_base_url,
        lightrag_client=lightrag_client,
    )


if __name__ == "__main__":
    main()
