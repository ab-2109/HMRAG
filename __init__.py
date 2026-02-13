"""
HM-RAG — Hierarchical Multi-Agent Multimodal RAG.

A three-layer retrieval-augmented generation framework for multimodal
science question answering (ACM MM 2025).

Architecture:
    Layer 1 — Decomposition Agent:  query analysis and splitting
    Layer 2 — Retrieval Agents:     parallel vector / graph / web retrieval
    Layer 3 — Decision Agent:       consistency voting + expert refinement
"""
