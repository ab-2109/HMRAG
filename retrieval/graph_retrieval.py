from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc

from retrieval.base_retrieval import BaseRetrieval


class GraphRetrieval(BaseRetrieval):
    def __init__(self, config):
        self.config = config
        self.mode = getattr(config, 'mode', 'mix')
        self.top_k = getattr(config, 'top_k', 4)
        ollama_host = getattr(config, 'ollama_base_url', 'http://localhost:11434')
        model_name = getattr(config, 'llm_model_name', 'qwen2.5:1.5b')
        working_dir = getattr(config, 'working_dir', './lightrag_workdir')

        self.client = LightRAG(
            working_dir=working_dir,
            llm_model_func=ollama_model_complete,
            llm_model_name=model_name,
            llm_model_max_async=4,
            # llm_model_max_token_size=65536,
            llm_model_kwargs={"host": ollama_host, "options": {"num_ctx": 4096}},
            embedding_func=EmbeddingFunc(
                embedding_dim=768,
                max_token_size=8192,
                func=lambda texts: ollama_embed.func(
                    texts, embed_model="nomic-embed-text", host=ollama_host
                ),
            ),
        )
        self.results = []

    def find_top_k(self, query):
        """
        Query the graph-based knowledge using the 'mix' mode.
        Args:
            query (str): The search query.
        Returns:
            str: The retrieval results.
        """
        try:
            self.results = self.client.query(
                query,
                param=QueryParam(mode=self.mode, top_k=self.top_k)
            )
        except Exception as e:
            print(f"GraphRetrieval error: {e}")
            self.results = f"Graph retrieval failed: {str(e)}"
        return self.results
    