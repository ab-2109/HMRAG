"""
HM-RAG agent layer.

Provides the three core agents of the framework:
    - :class:`DecomposeAgent`  — Layer 1: query decomposition
    - :class:`MRetrievalAgent` — Orchestrator: coordinates all three layers
    - :class:`SummaryAgent`    — Layer 3: consistency voting + refinement
"""

from agents.decompose_agent import DecomposeAgent
from agents.multi_retrieval_agents import MRetrievalAgent
from agents.summary_agent import SummaryAgent

__all__ = ["DecomposeAgent", "MRetrievalAgent", "SummaryAgent"]
