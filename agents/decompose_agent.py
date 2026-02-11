import os
from typing import List
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama


class DecomposeAgent:
    def __init__(self, config):
        self.config = config
        # Use Ollama to connect to the locally deployed model
        self.llm = Ollama(
            base_url=getattr(config, 'ollama_base_url', 'http://localhost:11434'),
            model=getattr(config, 'llm_model_name', 'qwen2.5:7b'),
            temperature=getattr(config, 'temperature', 0.35),
        )

    def count_intents(self, query: str) -> int:
        """
        Determine the number of intents in the input query.
        Uses LLM to analyze the number of intents contained in the input text.
        Args:
            query (str): The input query text.
        Returns:
            int: The number of intents.
        """
        prompt = PromptTemplate.from_template(
            "Please calculate how many independent intents are contained in the following query. "
            "Return only an integer:\n{query}\nNumber of intents: "
        )
        max_attempts = 3
        for attempt in range(max_attempts):
            formatted_prompt = prompt.format(query=query)
            response = self.llm.invoke(formatted_prompt)
            try:
                # Extract first integer from response
                import re
                numbers = re.findall(r'\d+', response.strip())
                if numbers:
                    return int(numbers[0])
            except (ValueError, IndexError):
                pass
            if attempt == max_attempts - 1:
                return 1  # Default to 1 intent if parsing fails
        return 1

    def decompose(self, query: str) -> List[str]:
        """
        Decompose the query. If the number of intents is greater than 1, perform intent decomposition.
        Args:
            query (str): The input query text.
        Returns:
            List[str]: A list of decomposed sub-queries, or a single-element list.
        """
        intent_count = self.count_intents(query)
        intent_count = min(intent_count, 3)  # Limit the number of intents to a maximum of 3
        if intent_count > 1:
            return self._split_query(query)
        return [query]

    def _split_query(self, query: str) -> List[str]:
        """
        The method that actually performs query decomposition.
        Args:
            query (str): The input query text.
        Returns:
            List[str]: A list of decomposed sub-queries.
        """
        prompt = PromptTemplate.from_template(
            "Split the following query into multiple independent sub-queries, "
            "separated by '||', without additional explanations:\n{query}\nList of sub-queries: "
        )
        formatted_prompt = prompt.format(query=query)
        response = self.llm.invoke(formatted_prompt)
        sub_queries = [q.strip() for q in response.split("||") if q.strip()]
        # If splitting failed, return the original query
        if not sub_queries:
            return [query]
        return sub_queries