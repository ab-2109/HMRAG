"""
Base retrieval interface for HM-RAG Layer 2 retrieval agents.

All retrieval agents (Vector, Graph, Web) inherit from :class:`BaseRetrieval`
so the orchestrator (:class:`MRetrievalAgent`) can treat them uniformly.

Section 3.3 of the paper describes three parallel retrieval agents:
    - VectorRetrieval  — LightRAG naive mode  (unstructured similarity search)
    - GraphRetrieval   — LightRAG hybrid mode (knowledge-graph traversal)
    - WebRetrieval     — SerpAPI + LLM answer generation (live web search)
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class BaseRetrieval(ABC):
    """Abstract base class for all HM-RAG retrieval agents.

    Every retrieval agent must implement :meth:`find_top_k` which accepts
    a query string and returns a response string (or structured data
    convertible to string).

    Attributes:
        config: Configuration namespace with model names, API keys,
                working directories, and hyperparameters.
        top_k:  Number of results to retrieve (read from config,
                default 4).
    """

    def __init__(self, config: Any):
        self.config = config
        self.top_k: int = int(getattr(config, 'top_k', 4))

    @abstractmethod
    def find_top_k(self, query: str) -> str:
        """Retrieve the most relevant results for *query*.

        Args:
            query: The search query string (may be an original question
                   or a sub-query produced by the Decomposition Agent).

        Returns:
            A string containing the retrieval results.  The Decision
            Agent will receive this string as one of its inputs.

        Raises:
            NotImplementedError: If the subclass has not implemented
                this method.
        """
        raise NotImplementedError("Subclasses must implement find_top_k()")

    def prepare(self, dataset: Any = None) -> None:
        """Optional hook to pre-load or index a dataset.

        Subclasses may override this to ingest documents into a vector
        store, build a knowledge graph, etc.  The default implementation
        is a no-op.

        Args:
            dataset: An optional dataset object or path to prepare from.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(top_k={self.top_k})"