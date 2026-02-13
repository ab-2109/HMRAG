"""
HM-RAG retrieval layer (Layer 2).

Provides the three parallel retrieval agents:
    - :class:`VectorRetrieval` — LightRAG naive mode (vector similarity)
    - :class:`GraphRetrieval`  — LightRAG hybrid mode (knowledge graph)
    - :class:`WebRetrieval`    — Serper API + Ollama LLM (live web search)

All inherit from :class:`BaseRetrieval`.
"""

from retrieval.base_retrieval import BaseRetrieval
from retrieval.vector_retrieval import VectorRetrieval
from retrieval.graph_retrieval import GraphRetrieval
from retrieval.web_retrieval import WebRetrieval

__all__ = ["BaseRetrieval", "VectorRetrieval", "GraphRetrieval", "WebRetrieval"]
