from lightrag import QueryParam
import inspect
import traceback

from retrieval.base_retrieval import BaseRetrieval
from retrieval.lightrag_factory import create_lightrag_client


class GraphRetrieval(BaseRetrieval):
    def __init__(self, config, client=None):
        self.config = config
        self.mode = getattr(config, 'mode', 'mix')
        self.top_k = getattr(config, 'top_k', 4)
        self.client = client if client is not None else create_lightrag_client(config)
        self.results = []

    def _build_query_param(self):
        kwargs = {
            "mode": self.mode,
            "top_k": self.top_k,
        }
        try:
            supported = set(inspect.signature(QueryParam).parameters.keys())
            if "enable_rerank" in supported:
                kwargs["enable_rerank"] = False
        except Exception:
            pass
        return QueryParam(**kwargs)

    def find_top_k(self, query):
        """
        Query the graph-based knowledge using the 'mix' mode.
        Args:
            query (str): The search query.
        Returns:
            str: The retrieval results.
        """
        debug_retrieval = getattr(self.config, 'debug_retrieval', False)
        param = self._build_query_param()
        if debug_retrieval:
            print(f"[LightRAG:graph] received query={query!r}")
            print(f"[LightRAG:graph] query param={param}")
        try:
            self.results = self.client.query(
                query,
                param=param
            )
            if debug_retrieval:
                preview = str(self.results)
                if len(preview) > 800:
                    preview = preview[:800] + "..."
                print(f"[LightRAG:graph] returned type={type(self.results).__name__} preview={preview}")
        except Exception as e:
            print(f"GraphRetrieval error: {e}")
            if debug_retrieval:
                print(traceback.format_exc())
            self.results = f"Graph retrieval failed: {str(e)}"
        return self.results
    