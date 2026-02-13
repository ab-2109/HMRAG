"""
Web-based Retrieval Agent (HM-RAG Layer 2, Section 3.3.3).

Performs live web search via SerpAPI (Google Search) and then uses a
local Ollama LLM to synthesise an answer from the retrieved snippets.

This agent provides real-time, up-to-date knowledge that complements
the static vector and graph retrieval agents.

Pipeline:
    1. query  →  SerpAPI (Google)  →  raw search results
    2. raw results  →  format_results()  →  readable snippets
    3. snippets + query  →  Ollama LLM  →  synthesised answer string
"""

import logging
from typing import Any, Dict, List, Optional, Union

from langchain_community.utilities import SerpAPIWrapper
from langchain_ollama import OllamaLLM

from retrieval.base_retrieval import BaseRetrieval

logger = logging.getLogger(__name__)

# Prompt template used to instruct the LLM to synthesise an answer
# from web search results.  Keeps the model focused and prevents
# hallucination beyond what the snippets support.
_SYNTHESIS_PROMPT = (
    "You are a helpful science question answering assistant.\n"
    "Below are search results retrieved from the web for the given question.\n"
    "Use ONLY the information in these search results to answer the question.\n"
    "If the results do not contain enough information, say so.\n"
    "Be concise and factual.\n\n"
    "Search results:\n{results}\n\n"
    "Question: {query}\n\n"
    "Answer:"
)


class WebRetrieval(BaseRetrieval):
    """Live web search retrieval agent using SerpAPI + Ollama LLM.

    Attributes:
        client: SerpAPIWrapper instance for Google search.
        llm:    OllamaLLM instance for answer synthesis from snippets.
    """

    def __init__(self, config: Any):
        super().__init__(config)

        serpapi_api_key = getattr(config, 'serpapi_api_key', '')
        ollama_base_url = getattr(config, 'ollama_base_url', 'http://localhost:11434')
        web_llm_model = getattr(config, 'web_llm_model_name', 'qwen2.5:1.5b')

        self.client = SerpAPIWrapper(serpapi_api_key=serpapi_api_key)

        self.llm = OllamaLLM(
            base_url=ollama_base_url,
            model=web_llm_model,
            temperature=0.35,
        )

        logger.info(
            "WebRetrieval initialised | top_k=%d | model=%s",
            self.top_k, web_llm_model,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find_top_k(self, query: str) -> str:
        """Search the web and synthesise an answer.

        Args:
            query: The search query (original question or sub-query).

        Returns:
            A string containing the LLM-synthesised answer, or an
            error message if the search or generation fails.
        """
        try:
            raw_results = self.client.results(query)
            formatted = self.format_results(raw_results)
            answer = self._generate(formatted, query)
            logger.debug(
                "WebRetrieval | query='%s' | snippets=%d chars | answer=%d chars",
                query[:60], len(formatted), len(answer),
            )
            return answer
        except Exception as e:
            logger.error("WebRetrieval failed for '%s': %s", query[:60], e)
            return f"Web retrieval failed: {e}"

    # ------------------------------------------------------------------
    # Result formatting
    # ------------------------------------------------------------------

    def format_results(self, results: Union[Dict, str, Any]) -> str:
        """Convert raw SerpAPI results into readable text snippets.

        Handles three response shapes:
            - ``dict`` with ``answerBox`` and/or ``organic`` keys
            - plain ``str`` (SerpAPIWrapper sometimes returns this)
            - anything else → ``str()`` fallback

        Args:
            results: Raw output from ``SerpAPIWrapper.results()``.

        Returns:
            A formatted string of search snippets.
        """
        # SerpAPIWrapper sometimes returns a plain string directly
        if isinstance(results, str):
            return results if results.strip() else "No relevant results found."

        if not isinstance(results, dict):
            return str(results) if results else "No relevant results found."

        snippets: List[str] = []

        # Direct answer box (featured snippet)
        answer_box = results.get('answerBox')
        if answer_box and isinstance(answer_box, dict):
            answer_text = (
                answer_box.get('answer')
                or answer_box.get('snippet')
                or answer_box.get('snippetHighlighted', [''])[0]
            )
            if answer_text:
                snippets.append(
                    f"Direct answer: {answer_text}\n"
                    f"Source: {answer_box.get('link', 'N/A')}"
                )

        # Organic search results
        for item in results.get('organic', [])[:self.top_k]:
            title = item.get('title', 'No title')
            snippet = item.get('snippet', 'No snippet')
            link = item.get('link', '')
            snippets.append(f"[{title}]\n{snippet}\nLink: {link}")

        # Knowledge graph panel
        knowledge = results.get('knowledgeGraph')
        if knowledge and isinstance(knowledge, dict):
            desc = knowledge.get('description', '')
            if desc:
                snippets.append(f"Knowledge Graph: {desc}")

        return "\n\n".join(snippets) if snippets else "No relevant results found."

    # ------------------------------------------------------------------
    # Answer synthesis
    # ------------------------------------------------------------------

    def _generate(self, search_results: str, query: str) -> str:
        """Use the LLM to synthesise an answer from search snippets.

        Args:
            search_results: Formatted search result text.
            query: The original query for context.

        Returns:
            The LLM-generated answer string.
        """
        prompt = _SYNTHESIS_PROMPT.format(results=search_results, query=query)
        try:
            answer = self.llm.invoke(prompt)
            return answer.strip() if answer else ""
        except Exception as e:
            logger.error("WebRetrieval generation failed: %s", e)
            return f"Web generation failed: {e}"
    
    