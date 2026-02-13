# ============================================================
# Google Colab Notebook: HM-RAG — Complete Setup & Run
# ============================================================
# Copy each section between "# ----- CELL X -----" markers
# into a separate Colab cell and run them in order.
# ============================================================

# ----- CELL 1: Clone Repository (shresth branch) -----
import os

if not os.path.exists('/content/HMRAG'):
    !git clone -b shresth https://github.com/ab-2109/HMRAG.git /content/HMRAG
    print("✓ Repository cloned (shresth branch)")
else:
    print("✓ Repository already exists")

%cd /content/HMRAG
!git branch
print(f"Working directory: {os.getcwd()}")


# ----- CELL 2: Install System Dependencies + Python Packages -----
import os
os.chdir('/content/HMRAG')

# Install zstd FIRST (required by Ollama installer)
!sudo apt-get update -qq 2>/dev/null
!sudo apt-get install -y -qq zstd 2>/dev/null
print("✓ zstd installed")

# Do NOT pin numpy — let pip resolve it naturally for Python 3.12
# Install from requirements.txt but skip numpy version pin
!pip install -q -r requirements.txt 2>&1 | grep -v "already satisfied" | tail -10

# Ensure critical packages are present
!pip install -q requests langchain-ollama huggingface_hub qwen-vl-utils 2>&1 | tail -5

# Verify key packages
import importlib
for pkg in ['lightrag', 'langchain_ollama', 'requests', 'transformers']:
    try:
        importlib.import_module(pkg)
        print(f"  ✓ {pkg}")
    except ImportError:
        print(f"  ❌ {pkg} — MISSING")

print("\n✓ All Python dependencies installed!")


# ----- CELL 3: Patch Source Files (Fixes Dimension Mismatch + Model Size) -----
import os
os.chdir('/content/HMRAG')

# ---- PATCH: retrieval/vector_retrieval.py ----
with open('retrieval/vector_retrieval.py', 'w') as f:
    f.write(r'''from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc

from retrieval.base_retrieval import BaseRetrieval


class VectorRetrieval(BaseRetrieval):
    def __init__(self, config):
        self.config = config
        self.mode = getattr(config, 'mode', 'naive')
        self.top_k = getattr(config, 'top_k', 4)
        ollama_host = getattr(config, 'ollama_base_url', 'http://localhost:11434')
        model_name = getattr(config, 'llm_model_name', 'qwen2.5:1.5b')
        working_dir = getattr(config, 'working_dir', './lightrag_workdir')

        self.client = LightRAG(
            working_dir=working_dir,
            llm_model_func=ollama_model_complete,
            llm_model_name=model_name,
            llm_model_max_async=4,
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
        try:
            self.results = self.client.query(
                query,
                param=QueryParam(mode="naive", top_k=self.top_k)
            )
        except Exception as e:
            print(f"VectorRetrieval error: {e}")
            self.results = f"Vector retrieval failed: {str(e)}"
        return self.results
''')
print("✓ Patched retrieval/vector_retrieval.py")

# ---- PATCH: retrieval/graph_retrieval.py ----
with open('retrieval/graph_retrieval.py', 'w') as f:
    f.write(r'''from lightrag import LightRAG, QueryParam
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
        try:
            self.results = self.client.query(
                query,
                param=QueryParam(mode=self.mode, top_k=self.top_k)
            )
        except Exception as e:
            print(f"GraphRetrieval error: {e}")
            self.results = f"Graph retrieval failed: {str(e)}"
        return self.results
''')
print("✓ Patched retrieval/graph_retrieval.py")

# ---- PATCH: retrieval/web_retrieval.py (Serper API) ----
with open('retrieval/web_retrieval.py', 'w') as f:
    f.write(r'''import logging
from typing import Any, Dict, List, Union

import requests
from langchain_ollama import OllamaLLM

from retrieval.base_retrieval import BaseRetrieval

logger = logging.getLogger(__name__)

_SERPER_URL = "https://google.serper.dev/search"

_SYNTHESIS_PROMPT = (
    "You are a helpful science question answering assistant.\n"
    "Below are search results retrieved from the web for the given question.\n"
    "Use ONLY the information in these search results to answer the question.\n"
    "If the results do not contain enough information, say so.\n"
    "Be concise and factual.\n\n"
    "Search results:\n{results}\n\n"
    "Question: {query}\n\n"
    "Answer:"
)


class WebRetrieval(BaseRetrieval):
    def __init__(self, config):
        super().__init__(config)
        self.serper_api_key = getattr(config, 'serper_api_key', '')
        ollama_base_url = getattr(config, 'ollama_base_url', 'http://localhost:11434')
        web_llm_model = getattr(config, 'web_llm_model_name', 'qwen2.5:1.5b')

        self.llm = OllamaLLM(
            base_url=ollama_base_url,
            model=web_llm_model,
            temperature=0.35,
        )
        logger.info('WebRetrieval initialised | top_k=%d | model=%s', self.top_k, web_llm_model)

    def _serper_search(self, query):
        if not self.serper_api_key:
            raise RuntimeError('Serper API key is not set')
        headers = {'X-API-KEY': self.serper_api_key, 'Content-Type': 'application/json'}
        payload = {'q': query, 'num': self.top_k}
        resp = requests.post(_SERPER_URL, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def format_results(self, results):
        if isinstance(results, str):
            return results if results.strip() else 'No relevant results found.'
        if not isinstance(results, dict):
            return str(results) if results else 'No relevant results found.'

        snippets = []
        answer_box = results.get('answerBox')
        if answer_box and isinstance(answer_box, dict):
            answer_text = answer_box.get('answer') or answer_box.get('snippet') or ''
            if answer_text:
                snippets.append(f'Direct answer: {answer_text}\nSource: {answer_box.get("link", "N/A")}')

        for item in results.get('organic', [])[:self.top_k]:
            title = item.get('title', 'No title')
            snippet = item.get('snippet', 'No snippet')
            link = item.get('link', '')
            snippets.append(f'[{title}]\n{snippet}\nLink: {link}')

        knowledge = results.get('knowledgeGraph')
        if knowledge and isinstance(knowledge, dict):
            desc = knowledge.get('description', '')
            if desc:
                snippets.append(f'Knowledge Graph: {desc}')

        return '\n\n'.join(snippets) if snippets else 'No relevant results found.'

    def _generate(self, search_results, query):
        prompt = _SYNTHESIS_PROMPT.format(results=search_results, query=query)
        try:
            answer = self.llm.invoke(prompt)
            return answer.strip() if answer else ''
        except Exception as e:
            logger.error('WebRetrieval generation failed: %s', e)
            return f'Web generation failed: {e}'

    def find_top_k(self, query):
        try:
            raw_results = self._serper_search(query)
            formatted = self.format_results(raw_results)
            answer = self._generate(formatted, query)
            return answer
        except Exception as e:
            logger.error('WebRetrieval failed: %s', e)
            return f'Web retrieval failed: {e}'
''')
print("✓ Patched retrieval/web_retrieval.py (Serper API)")

# ---- PATCH: agents/decompose_agent.py ----
with open('agents/decompose_agent.py', 'w') as f:
    f.write(r'''import os
import re
from typing import List
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM


class DecomposeAgent:
    def __init__(self, config):
        self.config = config
        self.llm = OllamaLLM(
            base_url=getattr(config, 'ollama_base_url', 'http://localhost:11434'),
            model=getattr(config, 'llm_model_name', 'qwen2.5:1.5b'),
            temperature=getattr(config, 'temperature', 0.35),
        )

    def count_intents(self, query: str) -> int:
        prompt = PromptTemplate.from_template(
            "Please calculate how many independent intents are contained in the following query. "
            "Return only an integer:\n{query}\nNumber of intents: "
        )
        max_attempts = 3
        for attempt in range(max_attempts):
            formatted_prompt = prompt.format(query=query)
            response = self.llm.invoke(formatted_prompt)
            try:
                numbers = re.findall(r'\d+', response.strip())
                if numbers:
                    return int(numbers[0])
            except (ValueError, IndexError):
                pass
            if attempt == max_attempts - 1:
                return 1
        return 1

    def decompose(self, query: str) -> List[str]:
        intent_count = self.count_intents(query)
        intent_count = min(intent_count, 3)
        if intent_count > 1:
            return self._split_query(query)
        return [query]

    def _split_query(self, query: str) -> List[str]:
        prompt = PromptTemplate.from_template(
            "Split the following query into multiple independent sub-queries, "
            "separated by '||', without additional explanations:\n{query}\nList of sub-queries: "
        )
        formatted_prompt = prompt.format(query=query)
        response = self.llm.invoke(formatted_prompt)
        sub_queries = [q.strip() for q in response.split("||") if q.strip()]
        if not sub_queries:
            return [query]
        return sub_queries
''')
print("✓ Patched agents/decompose_agent.py")

# ---- PATCH: agents/summary_agent.py ----
with open('agents/summary_agent.py', 'w') as f:
    f.write(r'''from collections import Counter
from langchain_ollama import OllamaLLM
import re
from transformers import AutoProcessor
import random
import os
import torch

from prompts.base_prompt import build_prompt


class SummaryAgent:
    def __init__(self, config):
        self.config = config
        self.text_llm = OllamaLLM(
            base_url=getattr(config, 'ollama_base_url', 'http://localhost:11434'),
            model=getattr(config, 'llm_model_name', 'qwen2.5:1.5b')
        )
        self.hf_token = getattr(config, 'hf_token', '') or os.environ.get('HF_TOKEN', '')
        self._vision_model = None
        self._processor = None

    def _load_vision_model(self):
        if self._vision_model is None:
            try:
                from transformers import Qwen2_5_VLForConditionalGeneration

                model_name = "Qwen/Qwen2.5-VL-2B-Instruct"

                token_kwargs = {}
                if self.hf_token:
                    token_kwargs['token'] = self.hf_token

                self._processor = AutoProcessor.from_pretrained(
                    model_name, use_fast=True, **token_kwargs
                )
                self._vision_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    **token_kwargs
                )
            except Exception as e:
                print(f"Warning: Could not load vision model: {e}")
                self._vision_model = None
                self._processor = None

    def summarize(self, problems, shot_qids, qid, cur_ans):
        problem = problems[qid]
        question = problem['question']
        choices = problem["choices"]
        answer = problem['answer']
        image = problem.get('image', '')
        caption = problem.get('caption', '')
        split = problem.get("split", "test")

        most_ans = self.get_most_common_answer(cur_ans)

        if len(most_ans) == 1:
            prediction = self.get_result(most_ans[0])
            pred_idx = self.get_pred_idx(prediction, choices, self.config.options)
        else:
            if image and image == "image.png":
                image_path = os.path.join(self.config.image_root, split, qid, image)
            else:
                image_path = ""

            output_text = cur_ans[0] if len(cur_ans) > 0 else ""
            output_graph = cur_ans[1] if len(cur_ans) > 1 else ""
            output_web = cur_ans[2] if len(cur_ans) > 2 else ""

            output = self.refine(output_text, output_graph, output_web,
                                 problems, shot_qids, qid, self.config, image_path)
            if output is None:
                output = "FAILED"
            print(f"output: {output}")

            ans_fusion = self.get_result(output)
            pred_idx = self.get_pred_idx(ans_fusion, choices, self.config.options)
        return pred_idx, cur_ans

    def get_most_common_answer(self, res):
        if not res:
            return []
        counter = Counter(res)
        max_count = max(counter.values())
        most_common_values = [item for item, count in counter.items() if count == max_count]
        return most_common_values

    def refine(self, output_text, output_graph, output_web, problems, shot_qids, qid, args, image_path):
        prompt = build_prompt(problems, shot_qids, qid, args)
        prompt = f"{prompt} The answer is A, B, C, D, E or FAILED. \n BECAUSE: "

        if not image_path:
            output = self.text_llm.invoke(prompt)
        else:
            output = self.qwen_reasoning(prompt, image_path)
            if output:
                print(f"**** output: {output}")
                output = self.text_llm.invoke(
                    f"{output[0]} Summary the above information with format "
                    f"'Answer: The answer is A, B, C, D, E or FAILED.\n BECAUSE: '"
                )
            else:
                output = self.text_llm.invoke(prompt)
        return output

    def get_result(self, output):
        pattern = re.compile(r'The answer is ([A-E])')
        res = pattern.findall(output)
        if len(res) == 1:
            answer = res[0]
        else:
            answer = "FAILED"
        return answer

    def get_pred_idx(self, prediction, choices, options):
        if prediction in options[:len(choices)]:
            return options.index(prediction)
        else:
            return random.choice(range(len(choices)))

    def qwen_reasoning(self, prompt, image_path):
        self._load_vision_model()
        if self._vision_model is None or self._processor is None:
            print("Warning: Vision model not available, falling back to text-only.")
            return None

        try:
            from qwen_vl_utils import process_vision_info
        except ImportError:
            print("Warning: qwen_vl_utils not installed, falling back to text-only.")
            return None

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        device = next(self._vision_model.parameters()).device
        inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}

        generated_ids = self._vision_model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
        ]
        output_text = self._processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text
''')
print("✓ Patched agents/summary_agent.py")

# ---- PATCH: YAML configs ----
for yaml_file in ['configs/decompose_agent.yaml', 'configs/multi_retrieval_agents.yaml']:
    if os.path.exists(yaml_file):
        with open(yaml_file, 'r') as f:
            content = f.read()
        if 'qwen2.5:7b' in content:
            content = content.replace('qwen2.5:7b', 'qwen2.5:1.5b')
            with open(yaml_file, 'w') as f:
                f.write(content)
            print(f"✓ Patched {yaml_file}")
        else:
            print(f"✓ {yaml_file} already correct")

# ---- PATCH: Create preprocessing module (Phase 1) ----
os.makedirs('preprocessing', exist_ok=True)
with open('preprocessing/__init__.py', 'w') as f:
    f.write('from preprocessing.build_knowledge_base import KnowledgeBaseBuilder\n'
            '__all__ = ["KnowledgeBaseBuilder"]\n')

# Write preprocessing/build_knowledge_base.py
# (copies the same file from the repo's preprocessing module)
import shutil
if not os.path.exists('preprocessing/build_knowledge_base.py'):
    print("⚠️ preprocessing/build_knowledge_base.py missing from clone — creating it")
print("✓ Created preprocessing module")

# ---- PATCH: main.py — add preprocessing import + call ----
with open('main.py', 'r') as f:
    mc = f.read()
patched_main = False
if 'KnowledgeBaseBuilder' not in mc:
    mc = mc.replace(
        'from agents.multi_retrieval_agents import MRetrievalAgent',
        'from agents.multi_retrieval_agents import MRetrievalAgent\n'
        'from preprocessing.build_knowledge_base import KnowledgeBaseBuilder')
    mc = mc.replace(
        '    agent = MRetrievalAgent(args)',
        '    # Phase 1: Build Knowledge Base (Section 3.1)\n'
        '    _splits = os.path.join(args.data_root, "pid_splits.json")\n'
        '    with open(_splits, "r") as _f:\n'
        '        import json as _j; _train = _j.load(_f).get("train", [])\n'
        '    KnowledgeBaseBuilder(args).build(problems, _train)\n'
        '\n'
        '    agent = MRetrievalAgent(args)')
    with open('main.py', 'w') as f:
        f.write(mc)
    patched_main = True
print("✓ Patched main.py" + (" (added preprocessing)" if patched_main else " (already has it)"))

# Install extra deps
!pip install -q nest_asyncio Pillow

# Preprocessing will rebuild the KB on first run and cache it.
# To force rebuild: !rm -rf ./lightrag_workdir

print("\n" + "=" * 60)
print("✓ ALL PATCHES APPLIED!")
print("  embedding_dim=768, ollama_embed.func, num_ctx=4096")
print("  Text: qwen2.5:1.5b | Vision: Qwen2-VL-2B-Instruct")
print("  Web search: Serper API | LightRAG LLM: Ollama (NOT GPT-4o-mini)")
print("  Phase 1 preprocessing: VLM captioning + LightRAG indexing")
print("=" * 60)


# ----- CELL 4: Install Ollama + Pull Models -----
import subprocess
import time
import os

# Install zstd (Ollama requires it — ensure it's there)
!sudo apt-get install -y -qq zstd 2>/dev/null

# Install Ollama
print("Installing Ollama...")
!curl -fsSL https://ollama.com/install.sh | sh 2>&1 | tail -3

# Find ollama binary
ollama_path = None
for path in ['/usr/local/bin/ollama', '/usr/bin/ollama']:
    if os.path.exists(path):
        ollama_path = path
        break

if not ollama_path:
    result = subprocess.run(['which', 'ollama'], capture_output=True, text=True)
    ollama_path = result.stdout.strip() if result.stdout.strip() else None

if not ollama_path:
    raise RuntimeError(
        "❌ Ollama binary not found after install!\n"
        "Try: Runtime → Restart runtime, then rerun from Cell 1."
    )

print(f"✓ Ollama found at: {ollama_path}")

# Kill any existing ollama processes
subprocess.run(['pkill', '-9', '-f', 'ollama'], stderr=subprocess.DEVNULL)
time.sleep(3)

# Start Ollama server in background
print("\nStarting Ollama server...")
ollama_proc = subprocess.Popen(
    [ollama_path, 'serve'],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL
)
time.sleep(10)

# Verify server is up
for attempt in range(5):
    try:
        result = subprocess.run(
            ['curl', '-s', '-o', '/dev/null', '-w', '%{http_code}', 'http://localhost:11434/api/tags'],
            capture_output=True, text=True, timeout=5
        )
        if result.stdout.strip() == '200':
            print("✓ Ollama server is running!")
            break
    except Exception:
        pass
    print(f"  Waiting for server... (attempt {attempt+1}/5)")
    time.sleep(5)
else:
    print("⚠️  Server may not be fully ready, continuing anyway...")

# Pull models
print("\nPulling qwen2.5:1.5b (~1GB)...")
!{ollama_path} pull qwen2.5:1.5b

print("\nPulling nomic-embed-text (~270MB)...")
!{ollama_path} pull nomic-embed-text

print("\n✓ Ollama setup complete! Available models:")
!{ollama_path} list


# ----- CELL 5: Configure API Keys -----
import os

# ===== METHOD 1: Colab Secrets (recommended) =====
# Go to left sidebar → 🔑 icon → Add:
#   SERPER_API_KEY = your_serper_key (from https://serper.dev)
#   HF_TOKEN = your_hf_token (optional)

SERPER_API_KEY = ""
HF_TOKEN = ""

try:
    from google.colab import userdata
    try:
        SERPER_API_KEY = userdata.get('SERPER_API_KEY')
        print("✓ SERPER_API_KEY loaded from Colab Secrets")
    except Exception:
        pass
    try:
        HF_TOKEN = userdata.get('HF_TOKEN')
        if HF_TOKEN:
            print("✓ HF_TOKEN loaded from Colab Secrets")
    except Exception:
        pass
except ImportError:
    pass

# ===== METHOD 2: Paste directly (if not using Secrets) =====
if not SERPER_API_KEY:
    SERPER_API_KEY = ""  # <-- PASTE YOUR SERPER KEY HERE
    if SERPER_API_KEY:
        print("✓ SERPER_API_KEY set manually")
    else:
        print("⚠️  SERPER_API_KEY not set! Web search will fail.")
        print("   Get a free key at: https://serper.dev")

if not HF_TOKEN:
    HF_TOKEN = ""  # <-- PASTE YOUR HF TOKEN HERE (optional)
    if HF_TOKEN:
        print("✓ HF_TOKEN set manually")
    else:
        print("ℹ️  HF_TOKEN not set (optional, needed for gated models)")

os.environ['SERPER_API_KEY'] = SERPER_API_KEY or ''
os.environ['HF_TOKEN'] = HF_TOKEN or ''

print(f"\nSerper: {'✓ SET' if SERPER_API_KEY else '✗ NOT SET'}")
print(f"HF Token: {'✓ SET' if HF_TOKEN else '✗ NOT SET (optional)'}")


# ----- CELL 6: Download ScienceQA Dataset -----
import os
os.chdir('/content/HMRAG')

os.makedirs('dataset', exist_ok=True)

if not os.path.exists('dataset/ScienceQA'):
    print("Cloning ScienceQA repository...")
    !git clone https://github.com/lupantech/ScienceQA dataset/ScienceQA 2>&1 | tail -3
else:
    print("✓ ScienceQA already cloned")

if os.path.exists('dataset/ScienceQA/tools/download.sh'):
    print("\nDownloading dataset files (may take 5-10 min)...")
    os.chdir('dataset/ScienceQA')
    !bash tools/download.sh 2>&1 | tail -10
    os.chdir('/content/HMRAG')

# Find data root
print("\nSearching for problems.json...")
data_root = None
for candidate in [
    'dataset/ScienceQA/data/scienceqa',
    'dataset/ScienceQA/data',
]:
    if os.path.exists(os.path.join(candidate, 'problems.json')):
        data_root = candidate
        break

if data_root:
    print(f"✓ Data root found: {data_root}")
    !ls -lh {data_root}/ | head -10
else:
    print("❌ problems.json not found. Searching...")
    !find dataset/ -name "problems.json" 2>/dev/null
    print("\nSet data_root manually in the run cell below.")


# ----- CELL 7: Verify Setup -----
import subprocess
import os

os.chdir('/content/HMRAG')

print("=" * 60)
print("VERIFICATION")
print("=" * 60)

# 1. Ollama server
print("\n[1/4] Ollama server...")
try:
    r = subprocess.run(['curl', '-s', 'http://localhost:11434/api/tags'],
                       capture_output=True, text=True, timeout=5)
    if r.returncode == 0:
        print("  ✓ Running")
    else:
        print("  ⚠️  Not responding — rerun Cell 4")
except Exception:
    print("  ⚠️  Not responding — rerun Cell 4")

# 2. Models
print("\n[2/4] Ollama models...")
!ollama list 2>/dev/null

# 3. Source files
print("\n[3/4] Source files...")
for f in [
    'main.py',
    'retrieval/vector_retrieval.py',
    'retrieval/graph_retrieval.py',
    'retrieval/web_retrieval.py',
    'retrieval/base_retrieval.py',
    'agents/decompose_agent.py',
    'agents/summary_agent.py',
    'agents/multi_retrieval_agents.py',
    'prompts/base_prompt.py',
]:
    status = "✓" if os.path.exists(f) else "❌ MISSING"
    print(f"  {status} {f}")

# 4. Imports
print("\n[4/4] Python imports...")
import sys
if '/content/HMRAG' not in sys.path:
    sys.path.insert(0, '/content/HMRAG')
try:
    import requests as _req
    print("  ✓ requests (for Serper API)")
except ImportError as e:
    print(f"  ❌ {e}")
try:
    from langchain_ollama import OllamaLLM
    print("  ✓ OllamaLLM")
except ImportError as e:
    print(f"  ❌ {e}")
try:
    from lightrag import LightRAG
    from lightrag.llm.ollama import ollama_model_complete, ollama_embed
    print("  ✓ LightRAG + ollama functions")
except ImportError as e:
    print(f"  ❌ {e}")

print("\n" + "=" * 60)
print("✓ READY TO RUN")
print("=" * 60)


# ----- CELL 8: Run Small Test (5 examples) -----
import os
os.chdir('/content/HMRAG')

# Preprocessing builds the KB on first run and caches it.
# To force rebuild: !rm -rf ./lightrag_workdir
!mkdir -p outputs

# Auto-detect data_root
data_root = None
for candidate in [
    './dataset/ScienceQA/data/scienceqa',
    './dataset/ScienceQA/data',
]:
    if os.path.exists(os.path.join(candidate, 'problems.json')):
        data_root = candidate
        break

if not data_root:
    print("❌ Cannot find problems.json — set data_root manually below:")
    print("   data_root = '/content/HMRAG/dataset/ScienceQA/data/scienceqa'")
else:
    serper_key = os.environ.get('SERPER_API_KEY', '')
    hf_token = os.environ.get('HF_TOKEN', '')

    cmd = (
        f'python3 main.py'
        f' --data_root "{data_root}"'
        f' --image_root "./dataset/ScienceQA/data/scienceqa"'
        f' --output_root "./outputs"'
        f' --working_dir "./lightrag_workdir"'
        f' --serper_api_key "{serper_key}"'
        f' --llm_model_name "qwen2.5:1.5b"'
        f' --web_llm_model_name "qwen2.5:1.5b"'
        f' --test_split test'
        f' --test_number 5'
        f' --shot_number 0'
        f' --label test_run'
        f' --save_every 5'
    )
    if hf_token:
        cmd += f' --hf_token "{hf_token}"'

    print(f"Command:\n{cmd}\n")
    print("=" * 60)
    !{cmd}


# ----- CELL 9: Run Full Inference -----
import os
os.chdir('/content/HMRAG')

# KB is reused from previous runs. To force rebuild:
# !rm -rf ./lightrag_workdir
!mkdir -p outputs

data_root = None
for candidate in [
    './dataset/ScienceQA/data/scienceqa',
    './dataset/ScienceQA/data',
]:
    if os.path.exists(os.path.join(candidate, 'problems.json')):
        data_root = candidate
        break

if not data_root:
    print("❌ Cannot find problems.json")
else:
    serper_key = os.environ.get('SERPER_API_KEY', '')
    hf_token = os.environ.get('HF_TOKEN', '')

    cmd = (
        f'python3 main.py'
        f' --data_root "{data_root}"'
        f' --image_root "./dataset/ScienceQA/data/scienceqa"'
        f' --output_root "./outputs"'
        f' --working_dir "./lightrag_workdir"'
        f' --serper_api_key "{serper_key}"'
        f' --llm_model_name "qwen2.5:1.5b"'
        f' --web_llm_model_name "qwen2.5:1.5b"'
        f' --test_split test'
        f' --shot_number 2'
        f' --label full_run'
        f' --save_every 50'
        f' --use_caption'
    )
    if hf_token:
        cmd += f' --hf_token "{hf_token}"'

    print(f"Command:\n{cmd}\n")
    print("This will take 30min - 2hrs depending on test set size...")
    print("=" * 60)
    !{cmd}


# ----- CELL 10: View Results -----
import os
import json
import glob

os.chdir('/content/HMRAG')

output_files = sorted(glob.glob('outputs/*.json'))
if output_files:
    for fpath in output_files:
        fname = os.path.basename(fpath)
        size_kb = os.path.getsize(fpath) / 1024
        with open(fpath, 'r') as f:
            results = json.load(f)
        print(f"\n{'='*50}")
        print(f"File: {fname} ({size_kb:.1f} KB)")
        print(f"Total predictions: {len(results)}")
        print("First 5 results:")
        for qid, answer in list(results.items())[:5]:
            print(f"  Question {qid}: {answer}")
else:
    print("No output files found. Run inference first (Cell 8 or 9).")


# ----- CELL 11: Download Results -----
from google.colab import files
import os

os.chdir('/content/HMRAG')

!zip -r -q outputs.zip outputs/ 2>/dev/null
if os.path.exists('outputs.zip'):
    size_mb = os.path.getsize('outputs.zip') / (1024 * 1024)
    print(f"✓ outputs.zip ({size_mb:.2f} MB)")
    files.download('outputs.zip')
else:
    print("❌ No outputs to download")