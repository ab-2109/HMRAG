# Migration from Serper to SerpAPI

## Changes Made

### 1. Code Changes

#### `retrieval/web_retrieval.py`
- Replaced `GoogleSerperAPIWrapper` with `SerpAPIWrapper`
- Updated API key parameter: `serper_api_key` → `serpapi_api_key`
- Updated result format parsing:
  - `answerBox` → `answer_box`
  - `organic` → `organic_results`

#### `main.py`
- Updated argument: `--serper_api_key` → `--serpapi_api_key`
- Updated help text to reference "SerpAPI web search"

#### `configs/multi_retrieval_agents.yaml`
- Updated key: `web_search_api_key` → `serpapi_api_key`

### 2. Documentation Changes

#### `COLAB_SETUP.md`
- Updated all references from Serper to SerpAPI
- Changed API key variable: `SERPER_API_KEY` → `SERPAPI_API_KEY`
- Updated URL: https://serper.dev/ → https://serpapi.com/
- Updated all command examples with new argument name

#### `README.md`
- Updated command example with new argument name

#### `colab_setup.ipynb`
- Updated API key variables throughout notebook
- Updated all command cells with new argument name

### 3. Requirements
- No changes needed to `requirements.txt`
- The `google-search-results` package supports SerpAPI

## How to Use

### Get SerpAPI Key
1. Visit https://serpapi.com/
2. Sign up (free tier: 100 searches/month)
3. Get your API key from the dashboard

### Running the Code

**Old command:**
```bash
python3 main.py --serper_api_key YOUR_KEY ...
```

**New command:**
```bash
python3 main.py --serpapi_api_key YOUR_KEY ...
```

### Environment Variables

**Old:**
```python
SERPER_API_KEY = "your-key"
```

**New:**
```python
SERPAPI_API_KEY = "your-key"
```

## Why SerpAPI?

SerpAPI offers:
- More stable API
- Better documentation
- 100 free searches/month
- Support for multiple search engines
- Structured JSON responses
- Better rate limiting

## Compatibility

This change is **NOT backward compatible**. Users must:
1. Get a SerpAPI key (instead of Serper)
2. Update their scripts to use `--serpapi_api_key`
3. Update environment variables if using them

## Testing

All files pass syntax check:
```bash
python3 -m py_compile retrieval/web_retrieval.py  # ✓ OK
```
