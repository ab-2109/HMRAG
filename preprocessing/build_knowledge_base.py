"""
Knowledge-Base Builder — HM-RAG Phase 1 (Section 3.1).

Implements the three-step pre-processing pipeline from the paper:

    Eq 1:  D_img  = VLM(image)              — image-to-text via Qwen VLM
    Eq 2:  D_comb = concat(D_text, D_img)   — merge textual + visual info
    Eq 3:  KB     = LightRAG.insert(D_comb) — index into vector + graph DB

The builder checks a marker file inside ``working_dir`` so the (expensive)
indexing step runs only once.  Delete ``.kb_built`` to force a rebuild.
"""

import asyncio
import gc
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)

# Marker file written after a successful build
_MARKER = ".kb_built"

# Batch size for LightRAG insertion (keeps memory bounded)
_INSERT_BATCH = 50

# Max new tokens for image captioning
_VLM_MAX_TOKENS = 256


# ======================================================================
# Image-to-text via Qwen VLM (Equation 1)
# ======================================================================

class ImageCaptioner:
    """Generate textual descriptions of images using a small Qwen VLM.

    Uses ``Qwen/Qwen2-VL-2B-Instruct`` which has ~2 B params on paper
    but only ~1.5 B *non-embedding* params — the smallest Qwen VL model
    publicly available as of 2024-12.

    The model is loaded lazily and released after captioning is done
    (``unload()``) so that GPU memory is available for the LLM during
    the evaluation phase.
    """

    MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

    def __init__(self, hf_token: str = "", device: Optional[str] = None):
        self.hf_token = hf_token
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._processor = None

    # ----- lazy load --------------------------------------------------

    def _load(self):
        if self._model is not None:
            return

        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        token_kw = {"token": self.hf_token} if self.hf_token else {}

        logger.info("Loading VLM %s …", self.MODEL_ID)
        self._processor = AutoProcessor.from_pretrained(
            self.MODEL_ID, use_fast=True, **token_kw,
        )
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.MODEL_ID,
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None,
            **token_kw,
        )
        if self.device != "cuda":
            self._model.to(self.device)
        logger.info("VLM loaded on %s", self.device)

    def unload(self):
        """Release model from GPU/CPU memory."""
        del self._model, self._processor
        self._model = None
        self._processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("VLM unloaded")

    # ----- captioning -------------------------------------------------

    def caption(self, image_path: str) -> str:
        """Return a textual description of the image at *image_path*.

        Falls back to an empty string on any error so that the pipeline
        never crashes because of a single bad image.
        """
        if not image_path or not os.path.isfile(image_path):
            return ""

        self._load()

        try:
            from qwen_vl_utils import process_vision_info

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"file://{image_path}"},
                        {
                            "type": "text",
                            "text": (
                                "Describe this image in detail for a science "
                                "question-answering system. Include all visible "
                                "text, labels, diagrams, charts, and any "
                                "relevant scientific information."
                            ),
                        },
                    ],
                }
            ]

            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self._processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            device = next(self._model.parameters()).device
            inputs = {
                k: v.to(device) if hasattr(v, "to") else v
                for k, v in inputs.items()
            }

            with torch.inference_mode():
                gen_ids = self._model.generate(**inputs, max_new_tokens=_VLM_MAX_TOKENS)
            # trim the input tokens from the generated sequence
            trimmed = [
                out[len(inp):]
                for inp, out in zip(inputs["input_ids"], gen_ids)
            ]
            caption = self._processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
            )[0].strip()
            return caption
        except Exception as e:
            logger.warning("Caption failed for %s: %s", image_path, e)
            return ""


# ======================================================================
# Document assembly (Equation 2)
# ======================================================================

def _build_document(
    problem: Dict[str, Any],
    qid: str,
    image_caption: str,
) -> str:
    """Combine every textual signal from a ScienceQA problem into one
    document string suitable for LightRAG indexing.

    Fields used (when available):
        - question + choices  (always present)
        - hint               (contextual clue, sometimes empty)
        - lecture             (background knowledge, sometimes empty)
        - solution            (step-by-step explanation, sometimes empty)
        - caption from dataset (human-written, sometimes empty)
        - image_caption       (VLM-generated, may be empty)
        - subject / topic / category metadata
    """
    parts: List[str] = []

    # Metadata
    subject = problem.get("subject", "")
    topic = problem.get("topic", "")
    category = problem.get("category", "")
    if subject or topic or category:
        parts.append(f"Subject: {subject}  Topic: {topic}  Category: {category}")

    # Core question
    q = problem.get("question", "")
    choices = problem.get("choices", [])
    if q:
        opts = " | ".join(
            f"({chr(65 + i)}) {c}" for i, c in enumerate(choices)
        )
        parts.append(f"Question: {q}\nOptions: {opts}")

    # Hint / context
    hint = (problem.get("hint") or "").strip()
    if hint:
        parts.append(f"Hint: {hint}")

    # Lecture (background knowledge)
    lecture = (problem.get("lecture") or "").strip()
    if lecture:
        parts.append(f"Lecture: {lecture}")

    # Solution / explanation
    solution = (problem.get("solution") or "").strip()
    if solution:
        parts.append(f"Solution: {solution}")

    # Dataset-provided caption
    ds_caption = (problem.get("caption") or "").strip()
    if ds_caption:
        parts.append(f"Image caption: {ds_caption}")

    # VLM-generated image description (Eq 1)
    if image_caption:
        parts.append(f"Image description (VLM): {image_caption}")

    # Correct answer (so the graph can extract answer-entity relations)
    answer_idx = problem.get("answer")
    if answer_idx is not None and answer_idx < len(choices):
        parts.append(f"Correct answer: ({chr(65 + answer_idx)}) {choices[answer_idx]}")

    return "\n".join(parts)


# ======================================================================
# Knowledge-base builder (Equation 3)
# ======================================================================

class KnowledgeBaseBuilder:
    """Build the LightRAG knowledge base from ScienceQA training data.

    Usage::

        builder = KnowledgeBaseBuilder(args)
        builder.build(problems, train_qids)
    """

    def __init__(self, config: Any):
        self.config = config
        self.working_dir: str = getattr(config, "working_dir", "./lightrag_workdir")
        self.image_root: str = getattr(config, "image_root", "")
        self.hf_token: str = (
            getattr(config, "hf_token", "") or os.environ.get("HF_TOKEN", "")
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        problems: Dict[str, Any],
        qids: List[str],
    ) -> None:
        """Run the full preprocessing pipeline (Equations 1-3).

        Args:
            problems: Full ScienceQA problems dict ``{qid: problem, …}``.
            qids:     List of question IDs to index (typically the
                      **training** split so the test set is unseen).
        """
        marker_path = os.path.join(self.working_dir, _MARKER)
        if os.path.exists(marker_path):
            logger.info(
                "Knowledge base already built (%s exists) — skipping. "
                "Delete that file to force a rebuild.",
                marker_path,
            )
            return

        os.makedirs(self.working_dir, exist_ok=True)

        logger.info("=" * 60)
        logger.info("Phase 1: Building Knowledge Base (%d problems)", len(qids))
        logger.info("=" * 60)

        # ── Step 1: Image captioning (Eq 1) ─────────────────────────
        captions = self._caption_images(problems, qids)

        # ── Step 2: Assemble documents (Eq 2) ───────────────────────
        documents = self._assemble_documents(problems, qids, captions)

        # ── Step 3: Index into LightRAG (Eq 3) ──────────────────────
        self._index_documents(documents)

        # Write marker
        with open(marker_path, "w") as f:
            f.write(f"Built from {len(documents)} documents\n")
        logger.info("Phase 1 complete — marker written to %s", marker_path)

    # ------------------------------------------------------------------
    # Step 1: Image captioning
    # ------------------------------------------------------------------

    def _caption_images(
        self,
        problems: Dict[str, Any],
        qids: List[str],
    ) -> Dict[str, str]:
        """Generate VLM captions for every image in *qids*.

        Returns:
            ``{qid: caption_string}`` — empty string if no image.
        """
        # Check if a cached captions file exists
        cache_path = os.path.join(self.working_dir, "vlm_captions.json")
        if os.path.exists(cache_path):
            logger.info("Loading cached VLM captions from %s", cache_path)
            with open(cache_path, "r") as f:
                return json.load(f)

        # Find which qids have images
        image_qids = []
        for qid in qids:
            img = problems[qid].get("image", "")
            if img:
                split = problems[qid].get("split", "train")
                img_path = os.path.join(self.image_root, split, qid, img)
                if os.path.isfile(img_path):
                    image_qids.append((qid, img_path))

        captions: Dict[str, str] = {qid: "" for qid in qids}

        if not image_qids:
            logger.info("No images found — skipping VLM captioning")
            return captions

        logger.info("Captioning %d images with VLM …", len(image_qids))
        captioner = ImageCaptioner(hf_token=self.hf_token)

        try:
            for i, (qid, img_path) in enumerate(image_qids):
                cap = captioner.caption(img_path)
                captions[qid] = cap
                if (i + 1) % 50 == 0 or (i + 1) == len(image_qids):
                    logger.info(
                        "  captioned %d / %d images", i + 1, len(image_qids),
                    )
        finally:
            captioner.unload()

        # Cache captions to disk
        with open(cache_path, "w") as f:
            json.dump(captions, f)
        logger.info("VLM captions cached to %s", cache_path)

        return captions

    # ------------------------------------------------------------------
    # Step 2: Assemble documents
    # ------------------------------------------------------------------

    def _assemble_documents(
        self,
        problems: Dict[str, Any],
        qids: List[str],
        captions: Dict[str, str],
    ) -> List[str]:
        """Build combined documents (Equation 2) for each problem."""
        documents: List[str] = []
        for qid in qids:
            doc = _build_document(problems[qid], qid, captions.get(qid, ""))
            if doc.strip():
                documents.append(doc)
        logger.info("Assembled %d documents", len(documents))
        return documents

    # ------------------------------------------------------------------
    # Step 3: Index into LightRAG
    # ------------------------------------------------------------------

    def _index_documents(self, documents: List[str]) -> None:
        """Insert all documents into LightRAG (vector + graph DB).

        Uses ``nest_asyncio`` so this works inside Colab / Jupyter
        event loops.
        """
        import nest_asyncio
        nest_asyncio.apply()

        from lightrag import LightRAG
        from lightrag.llm.ollama import ollama_model_complete, ollama_embed
        from lightrag.utils import EmbeddingFunc

        ollama_host = getattr(self.config, "ollama_base_url", "http://localhost:11434")
        model_name = getattr(self.config, "llm_model_name", "qwen2.5:1.5b")

        rag = LightRAG(
            working_dir=self.working_dir,
            llm_model_func=ollama_model_complete,
            llm_model_name=model_name,
            llm_model_max_async=4,
            llm_model_kwargs={
                "host": ollama_host,
                "options": {"num_ctx": 4096},
            },
            embedding_func=EmbeddingFunc(
                embedding_dim=768,
                max_token_size=8192,
                func=lambda texts: ollama_embed.func(
                    texts, embed_model="nomic-embed-text", host=ollama_host,
                ),
            ),
        )

        total = len(documents)
        logger.info("Indexing %d documents into LightRAG …", total)

        loop = asyncio.get_event_loop()
        for start in range(0, total, _INSERT_BATCH):
            batch = documents[start : start + _INSERT_BATCH]
            combined = "\n\n---\n\n".join(batch)
            try:
                loop.run_until_complete(rag.ainsert(combined))
            except Exception as e:
                logger.error(
                    "Insert failed for batch %d–%d: %s",
                    start, start + len(batch), e,
                )
            done = min(start + _INSERT_BATCH, total)
            logger.info("  indexed %d / %d documents", done, total)

        logger.info("LightRAG indexing complete")
