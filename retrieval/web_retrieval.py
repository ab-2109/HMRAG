from langchain_community.utilities import SerpAPIWrapper
from langchain_openai import ChatOpenAI

from retrieval.base_retrieval import BaseRetrieval


class WebRetrieval(BaseRetrieval):
    def __init__(self, config):
        self.config = config
        self.search_engine = "Google"
        
        serpapi_api_key = getattr(config, 'serpapi_api_key', '')
        self.top_k = getattr(config, 'top_k', 4)
        web_llm_model = getattr(config, 'web_llm_model_name', 'gpt-4o-mini')
        openai_api_key = getattr(config, 'openai_api_key', '')
        openai_base_url = getattr(config, 'openai_base_url', '')

        self.client = SerpAPIWrapper(
            serpapi_api_key=serpapi_api_key
        )

        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            base_url=openai_base_url if openai_base_url else None,
            model=web_llm_model,
            temperature=0.35,
        )
        self.results = []

    def format_results(self, results):
        """Format the search results into readable text."""
        max_results = self.top_k
        processed = []
        
        if isinstance(results, dict):
            if 'answerBox' in results or 'answer_box' in results:
                answer = results.get('answerBox', results.get('answer_box', {}))
                processed.append(
                    f"Direct answer: {answer.get('answer', '')}\n"
                    f"Source: {answer.get('link', '')}\n"
                )
            
            organic = results.get('organic', results.get('organic_results', []))
            if organic:
                for item in organic[:max_results]:
                    processed.append(
                        f"[{item.get('title', 'No title')}]\n"
                        f"{item.get('snippet', 'No snippet')}\n"
                        f"Link: {item.get('link', '')}\n"
                    )
        
        return "\n".join(processed) if processed else "No relevant results found"
    
    def generation(self, results_with_query):
        """Use the configured chat model to generate an answer from search results."""
        try:
            answer = self.llm.invoke(results_with_query).content
        except Exception as e:
            print(f"WebRetrieval generation error: {e}")
            answer = f"Web generation failed: {str(e)}"
        return answer
        
    def find_top_k(self, query):
        """
        Search the web and generate an answer.
        Args:
            query (str): The search query.
        Returns:
            str: The generated answer based on web search results.
        """
        try:
            raw_results = self.client.results(query)
            formatted_results = self.format_results(raw_results)
            self.results = self.generation(formatted_results + "\n" + query)
        except Exception as e:
            print(f"WebRetrieval error: {e}")
            self.results = f"Web retrieval failed: {str(e)}"
        return self.results
    
    