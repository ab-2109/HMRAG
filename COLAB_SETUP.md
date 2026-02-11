# Running HM-RAG on Google Colab

This guide provides step-by-step instructions for running HM-RAG on Google Colab.

## Quick Start

### Option 1: Use the Notebook
Upload `colab_setup.ipynb` to Google Colab and run the cells sequentially.

### Option 2: Manual Setup

## Prerequisites

1. **Google Colab Account** with GPU runtime (recommended)
2. **API Keys**:
   - SerpAPI Key (for web search, get from https://serpapi.com/) - **Required**
   - OpenAI API Key (optional - system uses Ollama by default)

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

### 3. Setup API Keys

Store your API keys in Colab Secrets (left sidebar → 🔑 Secrets):
- `SERPAPI_API_KEY` (Required)
- `OPENAI_API_KEY` (Optional - not used, kept for compatibility)

Then access them:
```python
from google.colab import userdata
SERPAPI_API_KEY = userdata.get('SERPAPI_API_KEY')
# OPENAI_API_KEY = userdata.get('OPENAI_API_KEY')  # Optional
```

### 4. Download Dataset
```bash
!bash dataset/download_ScienceQA.sh
```

### 5. Create Working Directories
```bash
!mkdir -p outputs lightrag_workdir
```

### 6. Run Inference (Test)

**Small test with 5 examples:**
```bash
!python3 main.py \
    --data_root ./dataset/ScienceQA/data \
    --image_root ./dataset/ScienceQA/images \
    --output_root ./outputs \
    --caption_file ./dataset/ScienceQA/captions.json \
    --working_dir ./lightrag_workdir \
    --serpapi_api_key "${SERPAPI_API_KEY}" \
    --test_split test \
    --test_number 5 \
    --shot_number 0 \
    --label test_run \
    --save_every 5
```

### 7. Run Full Inference

**Full dataset:**
```bash
!python3 main.py \
    --data_root ./dataset/ScienceQA/data \
    --image_root ./dataset/ScienceQA/images \
    --output_root ./outputs \
    --caption_file ./dataset/ScienceQA/captions.json \
    --working_dir ./lightrag_workdir \
    --serpapi_api_key "${SERPAPI_API_KEY}" \
    --test_split test \
    --shot_number 2 \
    --label full_run \
    --save_every 50 \
    --use_caption
```

### 8. View Results
```python
import json

with open('outputs/test_run_test.json', 'r') as f:
    results = json.load(f)

print(f"Total results: {len(results)}")
```

### 9. Download Results
```python
from google.colab import files

# Download single file
files.download('outputs/test_run_test.json')

# Or zip all outputs
!zip -r outputs.zip outputs/
files.download('outputs.zip')
```

## Important Notes for Colab

### 1. Ollama is Complex on Colab
The code uses Ollama for local LLM inference. On Colab, this is challenging because:
- Ollama needs to run as a background service
- Models are large and slow to download
- Colab has limited persistent storage

**Solution:** Follow the Ollama installation steps in the notebook or use a pre-configured environment.

### 2. GPU Runtime
For vision models (Qwen2.5-VL), enable GPU:
- Runtime → Change runtime type → GPU → T4 or better

### 3. Session Timeouts
Colab sessions timeout after inactivity. For long runs:
- Use `--save_every 10` to save frequently
- Keep the browser tab active
- Consider Colab Pro for longer sessions

### 4. Storage Limits
Free Colab has limited disk space. The ScienceQA dataset is large:
- Use `--test_number` to limit examples during testing
- Mount Google Drive for more storage:
```python
from google.colab import drive
drive.mount('/content/drive')
```

## Command-Line Arguments Reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_root` | Required | Path to problems.json and pid_splits.json |
| `--image_root` | '' | Path to images directory |
| `--output_root` | 'outputs' | Where to save results |
| `--caption_file` | '' | Path to captions.json |
| `--working_dir` | './lightrag_workdir' | LightRAG working directory |
| `--llm_model_name` | 'qwen2.5:7b' | Ollama model name |
| `--web_llm_model_name` | 'qwen2.5:7b' | Model for web retrieval |
| `--mode` | 'hybrid' | Retrieval mode: naive/hybrid/mix |
| `--serpapi_api_key` | '' | SerpAPI key for web search |
| `--ollama_base_url` | 'http://localhost:11434' | Ollama server URL |
| `--test_split` | 'test' | Dataset split: test/val/minival |
| `--test_number` | -1 | Number of examples (-1 = all) |
| `--shot_number` | 0 | Number of few-shot examples |
| `--label` | 'exp' | Experiment label for output file |
| `--debug` | False | Run only 10 examples |
| `--save_every` | 50 | Save results every N examples |
| `--use_caption` | False | Use image captions |
| `--seed` | 42 | Random seed |

## Troubleshooting

### "No module named 'lightrag'"
```bash
!pip install lightrag-hku
```

### "No module named 'qwen_vl_utils'"
```bash
!pip install qwen_vl_utils
```

### Out of Memory
- Reduce `--test_number`
- Use smaller models
- Restart runtime: Runtime → Restart runtime

### Dataset Download Fails
Check if `dataset/download_ScienceQA.sh` exists and is executable:
```bash
!chmod +x dataset/download_ScienceQA.sh
!bash dataset/download_ScienceQA.sh
```

### Ollama Connection Errors
The code tries to connect to `localhost:11434` for Ollama. On Colab:
1. Install Ollama (see notebook instructions)
2. Start Ollama service in background
3. Pull required models: `qwen2.5:7b`, `nomic-embed-text`

## Expected Output

After successful run:
```
====Input Arguments====
...
number of test problems: 5

training question ids for prompting: []

Results saved to outputs/test_run_test.json after 5 examples.
Number of correct answers: 3/5
Accuracy: 0.6000
Failed question ids: ['123', '456']
Number of failed questions: 2
```

## Tips for Best Results

1. **Start Small**: Use `--test_number 5` first
2. **Enable Captions**: Add `--use_caption` for better accuracy
3. **Use Few-Shot**: Try `--shot_number 2` for better prompting
4. **Save Frequently**: Use `--save_every 10` on Colab
5. **Monitor Resources**: Check GPU/RAM usage in Colab

## Getting API Keys

### SerpAPI Key
1. Go to https://serpapi.com/
2. Sign up (free tier: 100 searches/month)
3. Get your API key from dashboard
4. Copy and save it securely

Note: OpenAI API key is not required. The system uses Ollama for all LLM inference.

## Support

For issues, check:
- GitHub Issues: https://github.com/ab-2109/HMRAG/issues
- Original Paper: [HM-RAG arXiv](https://arxiv.org/abs/2504.12330)
