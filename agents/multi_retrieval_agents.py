"""
Multi-Retrieval Agent Orchestrator (HM-RAG, Section 3).

Coordinates the three-layer HM-RAG pipeline:
    Layer 1  – Decomposition Agent:  split complex queries into sub-queries
    Layer 2  – Retrieval Agents:     parallel vector / graph / web retrieval
    Layer 3  – Decision Agent:       consistency voting + expert refinement
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

from retrieval.vector_retrieval import VectorRetrieval
from retrieval.graph_retrieval import GraphRetrieval
from retrieval.web_retrieval import WebRetrieval
from agents.summary_agent import SummaryAgent
from agents.decompose_agent import DecomposeAgent

logger = logging.getLogger(__name__)


class MRetrievalAgent:
    """Hierarchical Multi-Agent orchestrator for the HM-RAG framework.

    Connects decomposition → parallel multi-source retrieval → decision
    for each question in the dataset.
    """

    # Labels used when formatting messages for the Decision Agent
    AGENT_LABELS = {
        "vector": "Vector Retrieval Agent",
        "graph":  "Graph Retrieval Agent",
        "web":    "Web Retrieval Agent",
    }

    def __init__(self, config):
        self.config = config

        # Layer 1 – Decomposition
        self.dec_agent = DecomposeAgent(config)

        # Layer 2 – Retrieval (vector, graph, web)
        self.vector_retrieval = VectorRetrieval(config)
        self.graph_retrieval  = GraphRetrieval(config)
        self.web_retrieval    = WebRetrieval(config)

        # Layer 3 – Decision
        self.sum_agent = SummaryAgent(config)

        # Response length cap to avoid blowing up the LLM context window
        self._max_response_len = int(getattr(config, 'max_response_length', 4096))

        logger.info("MRetrievalAgent initialised")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        problems: Dict[str, Any],
        shot_qids: List[str],
        qid: str,
    ) -> Tuple[int, List[str]]:
        """Run the full HM-RAG pipeline for one question.

        Args:
            problems:  Full dataset dict  {qid: problem_dict, …}.
            shot_qids: Question IDs used as few-shot examples.
            qid:       ID of the question to answer.

        Returns:
            (predicted_answer_index, list_of_agent_response_strings)
        """
        question = problems[qid]['question']
        logger.info("qid=%s | question='%s'", qid, question[:80])

        # ── Layer 1: Decomposition ──────────────────────────────────────
        t0 = time.time()
        sub_queries = self._decompose(question)
        logger.info(
            "Layer 1 | %.2fs | %d sub-queries: %s",
            time.time() - t0, len(sub_queries),
            [q[:60] for q in sub_queries],
        )

        # ── Layer 2: Multi-source Retrieval ─────────────────────────────
        t0 = time.time()
        agent_responses = self._retrieve_all(sub_queries)
        logger.info("Layer 2 | %.2fs", time.time() - t0)

        # ── Format for Decision Agent ───────────────────────────────────
        all_messages = self._format_messages(agent_responses)

        # ── Layer 3: Decision (Voting + Refinement) ─────────────────────
        t0 = time.time()
        final_ans, final_messages = self.sum_agent.summarize(
            problems, shot_qids, qid, all_messages
        )
        logger.info("Layer 3 | %.2fs | answer_idx=%s", time.time() - t0, final_ans)

        return final_ans, final_messages

    # ------------------------------------------------------------------
    # Layer 1 helper
    # ------------------------------------------------------------------

    def _decompose(self, question: str) -> List[str]:
        """Decompose *question* into sub-queries, with safe fallback."""
        try:
            sub_queries = self.dec_agent.decompose(question)
        except Exception as e:
            logger.warning("Decomposition failed (%s) — using original query", e)
            return [question]

        # Normalise to a non-empty list of strings
        if isinstance(sub_queries, str):
            sub_queries = [sub_queries]
        sub_queries = [q.strip() for q in (sub_queries or []) if q and q.strip()]
        return sub_queries if sub_queries else [question]

    # ------------------------------------------------------------------
    # Layer 2 helpers
    # ------------------------------------------------------------------

    def _retrieve_all(self, sub_queries: List[str]) -> Dict[str, str]:
        """Retrieve from all three sources for every sub-query.

        Per the paper (Section 3.3.3) the three retrieval agents run in
        parallel.  We use a ThreadPoolExecutor because all three agents
        are I/O-bound (HTTP calls to Ollama / SerpAPI / disk reads).

        Args:
            sub_queries: List of sub-query strings from Layer 1.

        Returns:
            {"vector": combined_str, "graph": combined_str, "web": combined_str}
        """
        all_responses: Dict[str, List[str]] = {
            "vector": [], "graph": [], "web": [],
        }

        for i, sub_q in enumerate(sub_queries, 1):
            logger.debug("Sub-query %d/%d: '%s'", i, len(sub_queries), sub_q[:80])
            v, g, w = self._retrieve_one_query(sub_q)
            all_responses["vector"].append(v)
            all_responses["graph"].append(g)
            all_responses["web"].append(w)

        # Join multiple sub-query results with a visual separator
        sep = "\n---\n" if len(sub_queries) > 1 else ""
        return {k: sep.join(v) for k, v in all_responses.items()}

    def _retrieve_one_query(self, query: str) -> Tuple[str, str, str]:
        """Run all three retrieval agents in parallel for a single query."""
        agents = {
            "vector": self.vector_retrieval,
            "graph":  self.graph_retrieval,
            "web":    self.web_retrieval,
        }
        results: Dict[str, str] = {}

        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {
                pool.submit(self._safe_retrieve, name, agent, query): name
                for name, agent in agents.items()
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    results[name] = future.result(timeout=120)
                except Exception as e:
                    logger.error("%s timed out / errored: %s", name, e)
                    results[name] = f"Retrieval timed out: {e}"

        return results.get("vector", ""), results.get("graph", ""), results.get("web", "")

    @staticmethod
    def _safe_retrieve(name: str, agent, query: str) -> str:
        """Call *agent.find_top_k* with error isolation."""
        try:
            response = agent.find_top_k(query)
            return str(response) if response else ""
        except Exception as e:
            logger.error("%s failed for '%s': %s", name, query[:60], e)
            return f"Retrieval failed: {e}"

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def _format_messages(self, agent_responses: Dict[str, str]) -> List[str]:
        """Build the three labelled messages expected by the Decision Agent."""
        messages = []
        for key in ("vector", "graph", "web"):
            label = self.AGENT_LABELS[key]
            text  = agent_responses.get(key, "No response available.")

            # Truncate excessively long responses
            if len(text) > self._max_response_len:
                text = text[: self._max_response_len] + "\n... [truncated]"
                logger.warning("%s response truncated to %d chars", label, self._max_response_len)

            messages.append(f"{label}:\n{text}\n")
        return messages
