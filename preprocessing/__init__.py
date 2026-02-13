"""
Preprocessing module for HM-RAG (Phase 1, Section 3.1).

Handles knowledge-base construction:
    1. Image → text via Qwen VLM  (Equation 1)
    2. Combine textual fields + image description  (Equation 2)
    3. Index combined documents into LightRAG  (Equation 3)
"""

from preprocessing.build_knowledge_base import KnowledgeBaseBuilder

__all__ = ["KnowledgeBaseBuilder"]
