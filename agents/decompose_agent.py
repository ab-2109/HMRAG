import re
import logging
from typing import List
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates aligned with the HM-RAG paper (Section 3.2):
#   "The Decomposition Agent judges whether decomposition is necessary
#    using a binary decision prompt and, when needed, decomposes the
#    reasoning steps of the original question into 2–3 simply and
#    logically connected sub-questions while retaining keywords from the
#    original question."
# ---------------------------------------------------------------------------

NEED_DECOMPOSE_PROMPT = PromptTemplate.from_template(
    "You are a question analysis assistant for science questions.\n"
    "Your task is to decide whether the following question requires "
    "decomposition into simpler sub-questions in order to be answered.\n\n"
    "A question should be decomposed when it:\n"
    "  - Involves multiple reasoning steps (e.g. compare, then conclude)\n"
    "  - Asks about more than one concept or entity\n"
    "  - Requires combining information from different knowledge areas\n\n"
    "A question should NOT be decomposed when it:\n"
    "  - Can be answered with a single fact or definition\n"
    "  - Asks about only one clear concept\n\n"
    "Question: {query}\n\n"
    "Does this question need to be decomposed? "
    "Answer with exactly YES or NO."
)

DECOMPOSE_PROMPT = PromptTemplate.from_template(
    "You are a question decomposition assistant for science questions.\n"
    "Break the following question into 2 to 3 simpler sub-questions that, "
    "when answered together, fully address the original question.\n\n"
    "Rules:\n"
    "1. Each sub-question must be self-contained and answerable independently.\n"
    "2. Retain important keywords and scientific terms from the original question.\n"
    "3. The sub-questions should be logically connected so that answering them "
    "in order leads to the answer of the original question.\n"
    "4. Output ONLY the sub-questions separated by '||'. Do not add numbering, "
    "explanations, or any other text.\n\n"
    "Original question: {query}\n\n"
    "Sub-questions:"
)

QUERY_REWRITE_PROMPT = PromptTemplate.from_template(
    "You are a query rewriting assistant for science questions.\n"
    "Rewrite the following question so that it is clearer and more specific, "
    "making it easier to search for relevant information. Keep all scientific "
    "terms and key concepts. Output ONLY the rewritten question.\n\n"
    "Original question: {query}\n\n"
    "Rewritten question:"
)


class DecomposeAgent:
    """Layer-1 Decomposition Agent (HM-RAG paper, Section 3.2).

    Responsibilities:
        1. Judge whether a query needs decomposition (binary decision).
        2. If yes, decompose into 2–3 logically connected sub-questions
           that retain keywords from the original question.
        3. If no, optionally rewrite the query for clarity before passing
           it to the retrieval layer.
    """

    def __init__(self, config):
        self.config = config
        self.max_intents = int(getattr(config, 'max_intents', 3))
        self.max_retries = 3

        self.llm = OllamaLLM(
            base_url=getattr(config, 'ollama_base_url', 'http://localhost:11434'),
            model=getattr(config, 'llm_model_name', 'qwen2.5:1.5b'),
            temperature=getattr(config, 'temperature', 0.0),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decompose(self, query: str) -> List[str]:
        """Decide whether to decompose *query* and return a list of sub-queries.

        Returns a list of 1-3 sub-queries.  If decomposition is not needed
        the list contains either the original query or a clarified rewrite.
        """
        needs_split = self._needs_decomposition(query)

        if needs_split:
            sub_queries = self._split_query(query)
            # Validate: if splitting produced only 1 or too many, fall back
            if len(sub_queries) < 2:
                logger.info("Decomposition produced <2 sub-queries; using original query.")
                return [self._rewrite_query(query)]
            return sub_queries[: self.max_intents]

        # Single-intent query – rewrite for better retrieval
        return [self._rewrite_query(query)]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _needs_decomposition(self, query: str) -> bool:
        """Binary decision: does the query need decomposition?

        Asks the LLM for a YES/NO answer.  Retries up to *max_retries* times
        on ambiguous responses, defaulting to False (no decomposition).
        """
        for attempt in range(self.max_retries):
            try:
                response = self.llm.invoke(
                    NEED_DECOMPOSE_PROMPT.format(query=query)
                ).strip().upper()

                # Accept first clear YES / NO token
                if "YES" in response.split()[0] if response else "":
                    logger.debug("Decomposition decision: YES (attempt %d)", attempt + 1)
                    return True
                if "NO" in response.split()[0] if response else "":
                    logger.debug("Decomposition decision: NO (attempt %d)", attempt + 1)
                    return False

                # Fallback: scan for YES/NO anywhere in the response
                if re.search(r'\bYES\b', response):
                    return True
                if re.search(r'\bNO\b', response):
                    return False
            except Exception as e:
                logger.warning("Decomposition decision attempt %d failed: %s", attempt + 1, e)

        logger.info("Could not determine decomposition need; defaulting to NO.")
        return False

    def _split_query(self, query: str) -> List[str]:
        """Decompose *query* into 2–3 sub-questions via the LLM.

        Retries on failure and validates that each sub-question is non-trivial.
        """
        for attempt in range(self.max_retries):
            try:
                response = self.llm.invoke(
                    DECOMPOSE_PROMPT.format(query=query)
                )
                sub_queries = self._parse_sub_queries(response, query)
                if sub_queries:
                    logger.info(
                        "Decomposed into %d sub-queries (attempt %d): %s",
                        len(sub_queries), attempt + 1, sub_queries,
                    )
                    return sub_queries
            except Exception as e:
                logger.warning("Decomposition attempt %d failed: %s", attempt + 1, e)

        logger.info("All decomposition attempts failed; returning original query.")
        return [query]

    def _rewrite_query(self, query: str) -> str:
        """Rewrite a single-intent query for clearer retrieval.

        On failure, returns the original query unchanged.
        """
        try:
            response = self.llm.invoke(
                QUERY_REWRITE_PROMPT.format(query=query)
            ).strip()
            # Basic sanity: rewrite should not be empty or absurdly short
            if response and len(response) >= 10:
                logger.debug("Query rewritten: %s -> %s", query, response)
                return response
        except Exception as e:
            logger.warning("Query rewrite failed: %s", e)

        return query

    # ------------------------------------------------------------------
    # Parsing utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_sub_queries(response: str, original_query: str) -> List[str]:
        """Extract sub-queries from LLM response.

        Tries '||' splitting first, then numbered-list patterns, then
        newline splitting as a last resort.
        """
        if not response or not response.strip():
            return []

        text = response.strip()

        # Strategy 1: split on '||' (the format we explicitly requested)
        if "||" in text:
            parts = [q.strip() for q in text.split("||") if q.strip()]
            parts = DecomposeAgent._clean_sub_queries(parts)
            if len(parts) >= 2:
                return parts

        # Strategy 2: numbered list  (e.g. "1. …\n2. …")
        numbered = re.findall(r'(?:^|\n)\s*\d+[.)]\s*(.+)', text)
        if len(numbered) >= 2:
            return DecomposeAgent._clean_sub_queries(numbered)

        # Strategy 3: bullet list  (e.g. "- …\n- …")
        bullets = re.findall(r'(?:^|\n)\s*[-•*]\s*(.+)', text)
        if len(bullets) >= 2:
            return DecomposeAgent._clean_sub_queries(bullets)

        # Strategy 4: newline-separated (each line is a sub-question)
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if len(lines) >= 2:
            return DecomposeAgent._clean_sub_queries(lines)

        return []

    @staticmethod
    def _clean_sub_queries(queries: List[str]) -> List[str]:
        """Remove numbering prefixes, quotes, and trivially short entries."""
        cleaned = []
        for q in queries:
            # Strip leading numbering / bullets / quotes
            q = re.sub(r'^[\d]+[.)]\s*', '', q)
            q = re.sub(r'^[-•*]\s*', '', q)
            q = q.strip().strip('"').strip("'").strip()
            # Keep only non-trivial entries (at least a few words)
            if len(q) >= 10:
                cleaned.append(q)
        return cleaned