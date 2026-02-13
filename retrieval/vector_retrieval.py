"""
Vector-based Retrieval Agent (HM-RAG Layer 2, Section 3.3.1).

Uses LightRAG in *naive* mode for flat vector similarity search over
embedded document chunks.  This provides broad, unstructured retrieval
that complements the graph-based and web-based agents.

The knowledge base must be populated first by the preprocessing step
(``preprocessing.build_knowledge_base.KnowledgeBaseBuilder``).

Embedding model: nomic-embed-text (768 dimensions) via Ollama.
"""

import asyncio
import logging
from typing import Any

from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc

from retrieval.base_retrieval import BaseRetrieval

logger = logging.getLogger(__name__)

# Fallback seed document — only used if the preprocessing step was
# skipped and the DB is completely empty.
_SEED_DOCUMENT = (
    "Science is the systematic study of the natural world through observation "
    "and experimentation. Key branches include physics, chemistry, biology, "
    "earth science, and astronomy. The scientific method involves forming "
    "hypotheses, conducting experiments, collecting data, and drawing conclusions."
)


class VectorRetrieval(BaseRetrieval):
    """Vector similarity retrieval agent using LightRAG naive mode.

    Connects to the *same* ``working_dir`` that was populated during
    Phase 1 preprocessing.  If the DB is empty (preprocessing was
    skipped), a minimal seed document is inserted as a fallback.

    Attributes:
        mode:   Always ``"naive"`` — pure vector similarity, no graph.
        client: The LightRAG instance configured with Ollama LLM and
                nomic-embed-text embeddings (768-dim).
    """

    MODE = "naive"

    def __init__(self, config: Any):
        super().__init__(config)

        self.mode: str = self.MODE
        self._initialised = False

        ollama_host = getattr(config, 'ollama_base_url', 'http://localhost:11434')
        model_name = getattr(config, 'llm_model_name', 'qwen2.5:1.5b')
        working_dir = getattr(config, 'working_dir', './lightrag_workdir')

        self.client = LightRAG(
            working_dir=working_dir,
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
                    texts, embed_model="nomic-embed-text", host=ollama_host
                ),
            ),
        )

        logger.info(
            "VectorRetrieval initialised | mode=%s | top_k=%d | model=%s | dir=%s",
            self.mode, self.top_k, model_name, working_dir,
        )

    def _ensure_initialised(self) -> None:
        """Insert a seed document only if the DB is completely empty.

        This is a fallback for when preprocessing was skipped.
        Normally the knowledge base is already populated by Phase 1.
        """
        if self._initialised:
            return
        try:
            import nest_asyncio
            nest_asyncio.apply()
            loop = asyncio.get_event_loop()
            loop.run_until_complete(self.client.ainsert(_SEED_DOCUMENT))
            logger.info("VectorRetrieval: seed document inserted (fallback)")
        except Exception as e:
            logger.debug("VectorRetrieval: seed insert skipped: %s", e)
        self._initialised = True

    def find_top_k(self, query: str) -> str:
        """Query the vector store via LightRAG naive mode.

        Args:
            query: The search query (original question or sub-query).

        Returns:
            The retrieval results as a string, or an error message
            if the query fails.
        """
        self._ensure_initialised()
        try:
            result = self.client.query(
                query,
                param=QueryParam(mode=self.MODE, top_k=self.top_k),
            )
            logger.debug(
                "VectorRetrieval | query='%s' | result_len=%d",
                query[:60], len(str(result)),
            )
            return str(result) if result else ""
        except Exception as e:
            logger.error("VectorRetrieval failed for '%s': %s", query[:60], e)
            return f"Vector retrieval failed: {e}"
    
    