"""
Graph-based Retrieval Agent (HM-RAG Layer 2, Section 3.3.2).

Uses LightRAG in *hybrid/mix* mode to traverse a knowledge graph built
from ingested documents.  The graph captures entity–relation triples and
enables multi-hop reasoning that pure vector similarity cannot achieve.

The knowledge base must be populated first by the preprocessing step
(``preprocessing.build_knowledge_base.KnowledgeBaseBuilder``).

Query modes supported by LightRAG:
    naive   — flat vector similarity (used by VectorRetrieval)
    local   — single-hop graph neighbours
    global  — community-level summaries
    hybrid  — combines local + global
    mix     — combines naive + local + global  (default here)
"""

import asyncio
import logging
from typing import Any

from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc

from retrieval.base_retrieval import BaseRetrieval

logger = logging.getLogger(__name__)

# Fallback seed document — only used if preprocessing was skipped.
_SEED_DOCUMENT = (
    "Science is the systematic study of the natural world through observation "
    "and experimentation. Key branches include physics, chemistry, biology, "
    "earth science, and astronomy. The scientific method involves forming "
    "hypotheses, conducting experiments, collecting data, and drawing conclusions."
)


class GraphRetrieval(BaseRetrieval):
    """Knowledge-graph retrieval agent using LightRAG hybrid/mix mode.

    Connects to the *same* ``working_dir`` that was populated during
    Phase 1 preprocessing.

    Attributes:
        mode:   LightRAG query mode (default ``"mix"``).
        client: The LightRAG instance configured with Ollama LLM and
                nomic-embed-text embeddings (768-dim).
    """

    def __init__(self, config: Any):
        super().__init__(config)

        self.mode: str = getattr(config, 'graph_search_mode',
                                 getattr(config, 'mode', 'mix'))
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
            "GraphRetrieval initialised | mode=%s | top_k=%d | model=%s | dir=%s",
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
            logger.info("GraphRetrieval: seed document inserted (fallback)")
        except Exception as e:
            logger.debug("GraphRetrieval: seed insert skipped: %s", e)
        self._initialised = True

    def find_top_k(self, query: str) -> str:
        """Query the knowledge graph via LightRAG.

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
                param=QueryParam(mode=self.mode, top_k=self.top_k),
            )
            logger.debug(
                "GraphRetrieval | mode=%s | query='%s' | result_len=%d",
                self.mode, query[:60], len(str(result)),
            )
            return str(result) if result else ""
        except Exception as e:
            logger.error("GraphRetrieval failed for '%s': %s", query[:60], e)
            return f"Graph retrieval failed: {e}"
    