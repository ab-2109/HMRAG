from lightrag import QueryParam

from retrieval.base_retrieval import BaseRetrieval
from retrieval.lightrag_factory import create_lightrag_client


class VectorRetrieval(BaseRetrieval):
    def __init__(self, config):
        self.config = config
        self.mode = getattr(config, 'mode', 'naive')
        self.top_k = getattr(config, 'top_k', 4)
        self.client = create_lightrag_client(config)
        self.results = []

    def find_top_k(self, query):
        """
        Query the vector-based knowledge using the 'naive' mode.
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
            print(f"VectorRetrieval error: {e}")
            self.results = f"Vector retrieval failed: {str(e)}"
        return self.results
    
    