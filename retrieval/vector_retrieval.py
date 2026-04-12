from lightrag import QueryParam
import inspect
import traceback

from retrieval.base_retrieval import BaseRetrieval
from retrieval.lightrag_factory import create_initialized_lightrag_client


class VectorRetrieval(BaseRetrieval):
    def __init__(self, config, client=None, refresh_client_callback=None):
        self.config = config
        self.mode = getattr(config, 'mode', 'naive')
        self.top_k = getattr(config, 'top_k', 4)
        self.client = client if client is not None else create_initialized_lightrag_client(config)
        self.refresh_client_callback = refresh_client_callback
        self.results = []

    def _ensure_client_ready(self):
        if self.client is None:
            raise RuntimeError("LightRAG vector client is None")
        query_fn = getattr(self.client, "query", None)
        if not callable(query_fn):
            raise RuntimeError("LightRAG vector client has no callable 'query'")

    def _refresh_client(self):
        if callable(self.refresh_client_callback):
            self.client = self.refresh_client_callback()
        else:
            self.client = create_initialized_lightrag_client(self.config)
        return self.client

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

    def _should_recover_client(self, error: Exception) -> bool:
        msg = str(error).lower()
        recovery_markers = [
            "asynchronous context manager protocol",
            "attached to a different loop",
        ]
        return any(marker in msg for marker in recovery_markers)

    def find_top_k(self, query):
        """
        Query the vector-based knowledge using the 'naive' mode.
        Args:
            query (str): The search query.
        Returns:
            str: The retrieval results.
        """
        debug_retrieval = getattr(self.config, 'debug_retrieval', False)
        param = self._build_query_param()
        if debug_retrieval:
            print(f"[LightRAG:vector] received query={query!r}")
            print(f"[LightRAG:vector] query param={param}")
            print(f"[LightRAG:vector] client_type={type(self.client).__name__}")
        try:
            self._ensure_client_ready()
            self.results = self.client.query(
                query,
                param=param
            )
            if debug_retrieval:
                preview = str(self.results)
                if len(preview) > 800:
                    preview = preview[:800] + "..."
                print(f"[LightRAG:vector] returned type={type(self.results).__name__} preview={preview}")
        except Exception as e:
            print(f"VectorRetrieval error: {e}")
            if debug_retrieval:
                print(traceback.format_exc())
            if self._should_recover_client(e):
                print("[LightRAG:vector] Recovering client and retrying query once.")
                try:
                    self._refresh_client()
                    self._ensure_client_ready()
                    self.results = self.client.query(query, param=param)
                    if debug_retrieval:
                        preview = str(self.results)
                        if len(preview) > 800:
                            preview = preview[:800] + "..."
                        print(
                            "[LightRAG:vector] recovery retry success "
                            f"type={type(self.results).__name__} preview={preview}"
                        )
                    return self.results
                except Exception as retry_error:
                    print(f"VectorRetrieval recovery retry failed: {retry_error}")
                    if debug_retrieval:
                        print(traceback.format_exc())
            self.results = f"Vector retrieval failed: {str(e)}"
        return self.results
    
    