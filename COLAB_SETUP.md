# Running HM-RAG on Google Colab (OpenAI + Neo4j + Qdrant + SmolVLM)

This guide is aligned with the current codebase:
- LightRAG LLM and entity extraction: OpenAI API (default model: gpt-4o-mini)
- LightRAG embeddings: text-embedding-3-small
- Graph storage: Neo4j
- Vector storage: Qdrant
- VLM for image reasoning: SmolVLM

## Prerequisites

1. Google Colab runtime (GPU recommended for SmolVLM)
2. API keys and endpoints:
- SerpAPI key (required for web retrieval)
- OpenAI API key (required)
- Neo4j URI, username, password (required)
- Qdrant URL (required), API key (optional)

## Step-by-Step Setup

### 1. Clone Repository
```bash
!git clone https://github.com/ab-2109/HMRAG.git
%cd HMRAG
```

### 2. Install Dependencies
```bash
!pip install -q -r requirements.txt
```

### 3. Configure Secrets

In Colab Secrets, set:
- SERPAPI_API_KEY
- OPENAI_API_KEY
- NEO4J_URI
- NEO4J_USERNAME
- NEO4J_PASSWORD
- QDRANT_URL
- QDRANT_API_KEY (optional)

Then load them in a cell:
```python
from google.colab import userdata

SERPAPI_API_KEY = userdata.get("SERPAPI_API_KEY")
OPENAI_API_KEY = userdata.get("OPENAI_API_KEY")
NEO4J_URI = userdata.get("NEO4J_URI")
NEO4J_USERNAME = userdata.get("NEO4J_USERNAME")
NEO4J_PASSWORD = userdata.get("NEO4J_PASSWORD")
QDRANT_URL = userdata.get("QDRANT_URL")

try:
    QDRANT_API_KEY = userdata.get("QDRANT_API_KEY")
except Exception:
    QDRANT_API_KEY = ""
```

### 4. Download Dataset
```bash
!bash dataset/download_ScienceQA.sh
```

### 5. Create Working Directories
```bash
!mkdir -p outputs lightrag_workdir
```

### 6. Run Inference (Quick Test)
```bash
!python3 main.py \
  --data_root ./dataset/ScienceQA/data \
  --image_root ./dataset/ScienceQA/images \
  --output_root ./outputs \
  --caption_file ./dataset/ScienceQA/data/captions.json \
  --working_dir ./lightrag_workdir \
  --serpapi_api_key "$SERPAPI_API_KEY" \
  --openai_api_key "$OPENAI_API_KEY" \
  --llm_model_name gpt-4o-mini \
  --decompose_model_name gpt-4o-mini \
  --web_llm_model_name gpt-4o-mini \
  --embedding_model_name text-embedding-3-small \
  --vlm_model_name HuggingFaceTB/SmolVLM-Instruct \
  --neo4j_uri "$NEO4J_URI" \
  --neo4j_username "$NEO4J_USERNAME" \
  --neo4j_password "$NEO4J_PASSWORD" \
  --qdrant_url "$QDRANT_URL" \
  --qdrant_api_key "$QDRANT_API_KEY" \
  --test_split test \
  --test_number 5 \
  --shot_number 0 \
  --label test_run \
  --save_every 5
```

### 7. Full Run
```bash
!python3 main.py \
  --data_root ./dataset/ScienceQA/data \
  --image_root ./dataset/ScienceQA/images \
  --output_root ./outputs \
  --caption_file ./dataset/ScienceQA/data/captions.json \
  --working_dir ./lightrag_workdir \
  --serpapi_api_key "$SERPAPI_API_KEY" \
  --openai_api_key "$OPENAI_API_KEY" \
  --neo4j_uri "$NEO4J_URI" \
  --neo4j_username "$NEO4J_USERNAME" \
  --neo4j_password "$NEO4J_PASSWORD" \
  --qdrant_url "$QDRANT_URL" \
  --qdrant_api_key "$QDRANT_API_KEY" \
  --test_split test \
  --shot_number 2 \
  --label full_run \
  --save_every 50 \
  --use_caption
```

## Colab Coherency Notes

1. Neo4j and Qdrant must be externally reachable from Colab.
2. If Neo4j or Qdrant are local on your laptop, expose them with a secure tunnel before running.
3. GPU in Colab primarily helps SmolVLM. OpenAI and remote databases run off-Colab.
4. The existing colab_setup.ipynb in this repo still contains old Ollama-era cells and should be treated as outdated.

## Troubleshooting

### Missing lightrag
```bash
!pip install -q lightrag-hku
```

### Neo4j connection/auth errors
- Verify scheme and port in NEO4J_URI (example: bolt://host:7687)
- Verify username/password and database permissions
- Verify network ingress/firewall allows Colab egress IPs

### Qdrant connection errors
- Verify QDRANT_URL format (example: https://host:6333)
- Verify API key if cluster requires it

### Slow or OOM in VLM stage
- Use a GPU runtime
- Reduce test_number for quick checks
- Keep VLM model as SmolVLM (default)
