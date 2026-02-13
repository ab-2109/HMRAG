"""
Base model utilities for HM-RAG.

Provides a lightweight base class and shared helpers used across the
framework's model interactions (Ollama text LLM, Qwen VL vision model).

In the current HM-RAG pipeline the actual model calls happen in:
    - agents/decompose_agent.py   (OllamaLLM for query decomposition)
    - agents/summary_agent.py     (OllamaLLM for voting, Qwen-VL for vision)
    - retrieval/web_retrieval.py  (OllamaLLM for answer generation)
    - retrieval/vector_retrieval.py & graph_retrieval.py (LightRAG internals)

This module centralises device management, memory cleanup, and message
formatting so that individual agents don't duplicate boilerplate.
"""

import logging
import os
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


class BaseModel:
    """Shared base for any model wrapper used in HM-RAG.

    Subclasses or consumers can use the class methods directly without
    instantiation — they are intentionally kept stateless where possible.
    """

    def __init__(self, config: Any = None):
        self.config = config

    # ------------------------------------------------------------------
    # Device / memory helpers
    # ------------------------------------------------------------------

    @staticmethod
    def get_device() -> torch.device:
        """Return the best available torch device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @staticmethod
    def clean_up() -> None:
        """Free GPU memory.  Safe to call even when no GPU is present."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.debug("GPU cache cleared")

    # ------------------------------------------------------------------
    # Chat-message formatting (OpenAI / Qwen chat-template style)
    # ------------------------------------------------------------------

    @staticmethod
    def create_text_message(text: str, question: str) -> Dict[str, Any]:
        """Build a chat message containing text context + question."""
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Context: {text}\n\nQuestion: {question}"},
            ],
        }

    @staticmethod
    def create_image_message(image_path: str, question: str) -> Dict[str, Any]:
        """Build a chat message containing an image + question.

        The *image_path* is converted to a ``file://`` URI if it is a
        local filesystem path (required by Qwen-VL's ``process_vision_info``).
        """
        if image_path and not image_path.startswith(("http://", "https://", "file://")):
            image_path = f"file://{os.path.abspath(image_path)}"

        return {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": question},
            ],
        }

    @staticmethod
    def create_ask_message(question: str) -> Dict[str, Any]:
        """Build a simple text-only user message."""
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
            ],
        }

    @staticmethod
    def build_chat_messages(
        question: str,
        text: Optional[str] = None,
        image_path: Optional[str] = None,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Assemble a complete chat-message list from the available inputs.

        Priority: reuse *history* if provided, then append text and/or
        image messages, falling back to a plain question message if
        neither text nor image is given.
        """
        messages: List[Dict[str, Any]] = list(history) if history else []

        if text:
            messages.append(BaseModel.create_text_message(text, question))
        if image_path:
            messages.append(BaseModel.create_image_message(image_path, question))
        if not text and not image_path:
            messages.append(BaseModel.create_ask_message(question))

        return messages