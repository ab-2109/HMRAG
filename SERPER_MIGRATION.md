# Migration from SerpAPI to Serper API

## Changes Made

### 1. Code Changes

#### `retrieval/web_retrieval.py`
- Replaced `SerpAPIWrapper` (from `langchain_community`) with direct `requests.post()` to Serper API
- Updated API key parameter: `serpapi_api_key` → `serper_api_key`
- Endpoint: `https://google.serper.dev/search`
- Auth: `X-API-KEY` header
- Removed dependency on `google-search-results` package

#### `main.py`
- Updated argument: `--serpapi_api_key` → `--serper_api_key`
- Updated help text to reference "Serper web search"

#### `configs/multi_retrieval_agents.yaml`
- Updated key: `serpapi_api_key` → `serper_api_key`

### 2. Documentation Changes

#### `COLAB_SETUP.md`
- Updated all references from SerpAPI to Serper
- Changed API key variable: `SERPAPI_API_KEY` → `SERPER_API_KEY`
- Updated URL: https://serpapi.com/ → https://serper.dev/
- Updated all command examples with new argument name

#### `README.md`
- Updated command example with new argument name

#### `colab_setup.ipynb`
- Updated API key variables throughout notebook
- Updated all command cells with new argument name

### 3. Requirements
- Removed `google-search-results` from `requirements.txt`
- Added `requests` (used for direct HTTP calls to Serper API)

## How to Use

### Get Serper API Key
1. Visit https://serper.dev/
2. Sign up (free tier: 2,500 searches)
3. Get your API key from the dashboard

### Running the Code

**Old command:**
```bash
python3 main.py --serpapi_api_key YOUR_KEY ...
```

**New command:**
```bash
python3 main.py --serper_api_key YOUR_KEY ...
```

### Environment Variables

**Old:**
```python
SERPAPI_API_KEY = "your-key"
```

**New:**
```python
SERPER_API_KEY = "your-key"
```

## Why Serper API?

Serper API offers:
- 2,500 free searches (vs SerpAPI's 100/month)
- Faster response times
- Simpler API (single POST endpoint)
- No heavy Python package dependency (`google-search-results` not needed)
- Clean JSON responses with `answerBox`, `organic`, `knowledgeGraph`

## LightRAG LLM Configuration

LightRAG is configured to use the **locally hosted Ollama qwen2.5:1.5b** model
instead of the default GPT-4o-mini. This is done by passing:
- `llm_model_func=ollama_model_complete`
- `llm_model_name='qwen2.5:1.5b'`

to the `LightRAG()` constructor in both `vector_retrieval.py` and
`graph_retrieval.py`. No OpenAI API key is needed.

## Compatibility

This change is **NOT backward compatible**. Users must:
1. Get a Serper API key (instead of SerpAPI)
2. Update their scripts to use `--serper_api_key`
3. Update environment variables from `SERPAPI_API_KEY` to `SERPER_API_KEY`
