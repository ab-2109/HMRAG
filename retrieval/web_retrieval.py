from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.llms import Ollama

from retrieval.base_retrieval import BaseRetrieval


class WebRetrieval(BaseRetrieval):
    def __init__(self, config):
        self.config = config
        self.search_engine = "Google"
        
        serper_api_key = getattr(config, 'serper_api_key', '')
        self.top_k = getattr(config, 'top_k', 4)
        ollama_base_url = getattr(config, 'ollama_base_url', 'http://localhost:11434')
        web_llm_model = getattr(config, 'web_llm_model_name', 'qwen2.5:7b')

        self.client = GoogleSerperAPIWrapper(
            serper_api_key=serper_api_key,
            gl="us",
            hl="en",
            k=self.top_k
        )

        self.llm = Ollama(
            base_url=ollama_base_url,
            model=web_llm_model,
            temperature=0.35,
        )
        self.results = []

    def format_results(self, results):
        """Format the search results into readable text."""
        max_results = self.top_k
        processed = []
        
        if isinstance(results, dict):
            if 'answerBox' in results:
                answer = results['answerBox']
                processed.append(
                    f"Direct answer: {answer.get('answer', '')}\n"
                    f"Source: {answer.get('link', '')}\n"
                )
            
            if 'organic' in results:
                for item in results['organic'][:max_results]:
                    processed.append(
                        f"[{item.get('title', 'No title')}]\n"
                        f"{item.get('snippet', 'No snippet')}\n"
                        f"Link: {item.get('link', '')}\n"
                    )
        
        return "\n".join(processed) if processed else "No relevant results found"
    
    def generation(self, results_with_query):
        """Use Ollama model to generate an answer from search results."""
        try:
            answer = self.llm.invoke(results_with_query)
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
    
    