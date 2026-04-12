import inspect
import os
import asyncio
import threading
from typing import Any, Dict

from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc


def _get_openai_provider_functions():
    try:
        from lightrag.llm.openai import openai_complete_if_cache, openai_embed

        return openai_complete_if_cache, openai_embed
    except Exception as e:
        raise ImportError(
            "Could not import LightRAG OpenAI provider functions. "
            "Please ensure lightrag-hku is installed with OpenAI support."
        ) from e


def _require_config_value(config, attr_name: str, env_name: str = "") -> str:
    value = getattr(config, attr_name, "") or ""
    if value:
        return str(value)
    if env_name:
        env_value = os.getenv(env_name, "")
        if env_value:
            return env_value
    raise ValueError(
        f"Missing required config '{attr_name}'"
        + (f" (or environment variable '{env_name}')" if env_name else "")
    )


def _normalize_openai_base_url(value: str) -> str:
    """Return a valid OpenAI-compatible base URL or empty string.

    LightRAG/OpenAI client raises UnsupportedProtocol when base_url is an empty
    string or protocol-less host. We only forward base_url when it is valid.
    """
    raw = (value or "").strip()
    if not raw:
        return ""
    if raw.startswith("http://") or raw.startswith("https://"):
        return raw.rstrip("/")
    # Treat protocol-less values as invalid to avoid hard-to-debug runtime errors.
    return ""


def _build_openai_embedding_func(openai_embed, embedding_model_name: str):
    def _embed(texts):
        # Keep compatibility across LightRAG versions that renamed keyword args.
        try:
            return openai_embed(
                texts,
                model=embedding_model_name,
            )
        except TypeError:
            return openai_embed(
                texts,
                embed_model=embedding_model_name,
            )

    # Some LightRAG storage backends look for model_name on embedding func
    # to isolate collection/workspace names.
    setattr(_embed, "model_name", embedding_model_name)
    return _embed


def _build_openai_llm_func(openai_complete_if_cache, llm_model_name: str):
    """Build a version-tolerant OpenAI completion adapter for LightRAG.

    LightRAG/OpenAI provider signatures differ between versions:
    - (prompt, ...)
    - (model, prompt, ...)
    This adapter supports both.
    """

    provider_sig = inspect.signature(openai_complete_if_cache)
    params = list(provider_sig.parameters.keys())
    model_first = len(params) >= 2 and params[0] == "model" and params[1] == "prompt"

    if model_first:
        async def _llm(prompt, system_prompt=None, history_messages=None, **kwargs):
            return await openai_complete_if_cache(
                llm_model_name,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                **kwargs,
            )
    else:
        async def _llm(prompt, system_prompt=None, history_messages=None, **kwargs):
            return await openai_complete_if_cache(
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                **kwargs,
            )

    return _llm


def create_lightrag_client(config):
    """Create a LightRAG client configured for OpenAI + Neo4j + Qdrant."""
    openai_api_key = _require_config_value(config, "openai_api_key", "OPENAI_API_KEY")
    neo4j_uri = _require_config_value(config, "neo4j_uri", "NEO4J_URI")
    neo4j_username = _require_config_value(config, "neo4j_username", "NEO4J_USERNAME")
    neo4j_password = _require_config_value(config, "neo4j_password", "NEO4J_PASSWORD")
    qdrant_url = _require_config_value(config, "qdrant_url", "QDRANT_URL")

    openai_base_url = _normalize_openai_base_url(
        getattr(config, "openai_base_url", "") or os.getenv("OPENAI_BASE_URL", "")
    )
    llm_model_name = getattr(config, "llm_model_name", "gpt-4o-mini")
    embedding_model_name = getattr(config, "embedding_model_name", "text-embedding-3-small")
    working_dir = getattr(config, "working_dir", "./lightrag_workdir")
    qdrant_api_key = getattr(config, "qdrant_api_key", "") or os.getenv("QDRANT_API_KEY", "")
    qdrant_collection = getattr(config, "qdrant_collection", "hmrag_chunks")
    neo4j_database = getattr(config, "neo4j_database", "neo4j")

    # Make credentials discoverable for provider functions that read from env vars.
    os.environ["OPENAI_API_KEY"] = openai_api_key
    if openai_base_url:
        os.environ["OPENAI_BASE_URL"] = openai_base_url
    elif "OPENAI_BASE_URL" in os.environ:
        # Prevent downstream clients from accidentally reading an invalid leftover value.
        del os.environ["OPENAI_BASE_URL"]

    openai_complete_if_cache, openai_embed = _get_openai_provider_functions()
    llm_complete_func = _build_openai_llm_func(openai_complete_if_cache, llm_model_name)

    embedding_func = _build_openai_embedding_func(openai_embed, embedding_model_name)
    embedding_func_kwargs: Dict[str, Any] = {
        "embedding_dim": 1536,
        "max_token_size": 8192,
        "func": embedding_func,
    }
    # Newer LightRAG versions accept this and use it for data isolation naming.
    if "model_name" in inspect.signature(EmbeddingFunc).parameters:
        embedding_func_kwargs["model_name"] = embedding_model_name

    llm_model_kwargs: Dict[str, Any] = {
        "api_key": openai_api_key,
    }
    if openai_base_url:
        llm_model_kwargs["base_url"] = openai_base_url

    ctor_kwargs: Dict[str, Any] = {
        "working_dir": working_dir,
        "llm_model_func": llm_complete_func,
        "llm_model_name": llm_model_name,
        "llm_model_max_async": 4,
        "llm_model_kwargs": llm_model_kwargs,
        "graph_storage": "Neo4JStorage",
        "graph_storage_kwargs": {
            "uri": neo4j_uri,
            "username": neo4j_username,
            "password": neo4j_password,
            "database": neo4j_database,
        },
        "vector_storage": "QdrantVectorDBStorage",
        "vector_storage_kwargs": {
            "url": qdrant_url,
            "api_key": qdrant_api_key,
            "collection_name": qdrant_collection,
        },
        "embedding_func": EmbeddingFunc(**embedding_func_kwargs),
    }

    supported_params = set(inspect.signature(LightRAG.__init__).parameters.keys())
    missing_storage_hooks = [
        p for p in ("graph_storage", "vector_storage") if p not in supported_params
    ]
    if missing_storage_hooks:
        raise RuntimeError(
            "Installed LightRAG version does not expose storage backend parameters "
            f"({', '.join(missing_storage_hooks)}). Please upgrade lightrag-hku."
        )

    filtered_kwargs = {k: v for k, v in ctor_kwargs.items() if k in supported_params}
    return LightRAG(**filtered_kwargs)


def _run_maybe_async(value):
    if inspect.isawaitable(value):
        try:
            asyncio.get_running_loop()
            has_running_loop = True
        except RuntimeError:
            has_running_loop = False

        if not has_running_loop:
            return asyncio.run(value)

        # Jupyter/Colab may already have a running loop in this thread.
        # Run the coroutine in a fresh loop on a helper thread.
        result_holder = {"value": None, "error": None}

        def _runner():
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                result_holder["value"] = loop.run_until_complete(value)
            except Exception as e:
                result_holder["error"] = e
            finally:
                loop.close()

        t = threading.Thread(target=_runner, daemon=True)
        t.start()
        t.join()
        if result_holder["error"] is not None:
            raise result_holder["error"]
        return result_holder["value"]
    return value


def initialize_lightrag_client(client) -> None:
    """Initialize LightRAG storages/pipeline when hooks exist.

    Some LightRAG versions require explicit initialization before query/insert,
    otherwise internal async storage handles can remain unset.
    """
    init_fn = getattr(client, "initialize_storages", None)
    if callable(init_fn):
        _run_maybe_async(init_fn())

    pipeline_fn = getattr(client, "initialize_pipeline_status", None)
    if callable(pipeline_fn):
        _run_maybe_async(pipeline_fn())


def create_initialized_lightrag_client(config):
    client = create_lightrag_client(config)
    initialize_lightrag_client(client)
    return client
