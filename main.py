import os
import re
import json
import argparse
import random
from tqdm import tqdm
import sys
from agents.multi_retrieval_agents import MRetrievalAgent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to root data directory containing problems.json and pid_splits.json')
    parser.add_argument('--image_root', type=str, default='',
                        help='Path to image root directory')
    parser.add_argument('--output_root', type=str, default='outputs',
                        help='Path to output directory')
    parser.add_argument('--caption_file', type=str, default='',
                        help='Path to captions JSON file')
    parser.add_argument('--model', type=str, default='gpt3')
    parser.add_argument('--options', type=str, nargs='+', default=["A", "B", "C", "D", "E"])
    # user options
    parser.add_argument('--test_split', type=str, default='test', choices=['test', 'val', 'minival'])
    parser.add_argument('--test_number', type=int, default=-1,
                        help='Number of test problems to run. -1 means all.')
    parser.add_argument('--shot_number', type=int, default=0,
                        help='Number of n-shot training examples for prompting')
    parser.add_argument('--shot_qids', type=int, nargs='+', default=None,
                        help='Specific question IDs for shot examples')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--label', type=str, default='exp', help='Experiment label for output file naming')
    parser.add_argument('--debug', action='store_true', help='Debug mode: run only first 10 examples')
    parser.add_argument('--save_every', type=int, default=50,
                        help='Save results every N examples')
    parser.add_argument('--use_caption', action='store_true', help='Whether to use image captions')
    parser.add_argument('--prompt_format',
                        type=str,
                        default='CQM-A',
                        choices=[
                            'CQM-A', 'CQM-LA', 'CQM-EA', 'CQM-LEA', 'CQM-ELA', 'CQM-AL', 'CQM-AE', 'CQM-ALE',
                            'QCM-A', 'QCM-LA', 'QCM-EA', 'QCM-LEA', 'QCM-ELA', 'QCM-AL', 'QCM-AE', 'QCM-ALE',
                            'QCML-A', 'QCME-A', 'QCMLE-A', 'QCLM-A', 'QCEM-A', 'QCLEM-A', 'QCML-AE'
                        ],
                        help='prompt format template')
    # VectorRetrieval / LightRAG settings
    parser.add_argument('--working_dir', type=str, default='./lightrag_workdir',
                        help='Working directory for LightRAG')
    parser.add_argument('--llm_model_name', type=str, default='qwen2.5:1.5b')
    parser.add_argument('--mode', type=str, default='hybrid')
    parser.add_argument('--serpapi_api_key', type=str, default='',
                        help='API key for SerpAPI web search')
    parser.add_argument('--web_llm_model_name', type=str, default='qwen2.5:1.5b',
                        help='LLM model name for web retrieval generation')
    parser.add_argument('--top_k', type=int, default=4)
    parser.add_argument('--hf_token', type=str, default='',
                        help='Hugging Face access token for downloading gated models')
    # Ollama settings (REQUIRED)
    parser.add_argument('--ollama_base_url', type=str, default='http://localhost:11434',
                        help='Base URL for Ollama server')

    args = parser.parse_args()
    return args


def load_data(args):
    problems = json.load(open(os.path.join(args.data_root, 'problems.json')))
    pid_splits = json.load(open(os.path.join(args.data_root, 'pid_splits.json')))

    # Load captions if caption_file is provided
    if args.caption_file and os.path.exists(args.caption_file):
        captions = json.load(open(args.caption_file)).get("captions", {})
    else:
        captions = {}

    for qid in problems:
        problems[qid]['caption'] = captions.get(qid, "")

    qids = pid_splits['%s' % (args.test_split)]
    qids = qids[:args.test_number] if args.test_number > 0 else qids
    print(f"number of test problems: {len(qids)}\n")

    # pick up shot examples from the training set
    shot_qids = args.shot_qids
    train_qids = pid_splits.get('train', [])
    if shot_qids is None:
        assert args.shot_number >= 0 and args.shot_number <= 32
        if args.shot_number > 0 and len(train_qids) > 0:
            shot_qids = random.sample(train_qids, min(args.shot_number, len(train_qids)))
        else:
            shot_qids = []
    else:
        shot_qids = [str(qid) for qid in shot_qids]
        for qid in shot_qids:
            assert qid in train_qids, f"Shot QID {qid} not found in training set"
    print("training question ids for prompting: ", shot_qids, "\n")

    return problems, qids, shot_qids


def main():
    args = parse_args()
    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=False))

    random.seed(args.seed)

    problems, qids, shot_qids = load_data(args)

    result_file = os.path.join(args.output_root, args.label + '_' + args.test_split + '.json')
    if not os.path.exists(args.output_root):
        os.makedirs(args.output_root)

    # Set HF token if provided (for downloading gated models)
    if args.hf_token:
        import os as _os
        _os.environ['HF_TOKEN'] = args.hf_token
        _os.environ['HUGGING_FACE_HUB_TOKEN'] = args.hf_token
        try:
            from huggingface_hub import login
            login(token=args.hf_token)
            print("✓ Logged in to Hugging Face Hub")
        except Exception as e:
            print(f"Warning: Could not login to HF Hub: {e}")

    agent = MRetrievalAgent(args)
    correct = 0
    results = {}
    outputs = {}
    failed = []

    for i, qid in enumerate(tqdm(qids)):
        if args.debug and i > 10:
            break
        if args.test_number > 0 and i >= args.test_number:
            break

        problem = problems[qid]
        answer = problem['answer']

        try:
            final_ans, all_messages = agent.predict(problems, shot_qids, qid)
        except Exception as e:
            print(f"Error processing qid {qid}: {e}")
            final_ans = -1
            all_messages = [str(e)]

        outputs[qid] = all_messages
        results[qid] = final_ans

        if final_ans == answer:
            correct += 1
        else:
            failed.append(qid)

        if (i + 1) % args.save_every == 0:
            with open(result_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {result_file} after {i + 1} examples.")

    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)

    total = len(qids) if not args.debug else min(len(qids), 11)
    print(f"\nResults saved to {result_file}")
    print(f"Number of correct answers: {correct}/{total}")
    if total > 0:
        print(f"Accuracy: {correct / total:.4f}")
    print(f"Failed question ids: {failed}")
    print(f"Number of failed questions: {len(failed)}")


if __name__ == "__main__":
    main()


