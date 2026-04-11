from collections import Counter
from langchain_openai import ChatOpenAI
import re
from transformers import AutoProcessor
import random
import os
from PIL import Image
import torch

from prompts.base_prompt import build_prompt


class SummaryAgent:
    def __init__(self, config):
        self.config = config
        self.text_llm = ChatOpenAI(
            api_key=getattr(config, 'openai_api_key', ''),
            base_url=getattr(config, 'openai_base_url', '') or None,
            model=getattr(config, 'llm_model_name', 'gpt-4o-mini')
        )
        # Lazy-load the vision model only when needed
        self._vision_model = None
        self._processor = None
        self._vision_device = None

    def _load_vision_model(self):
        """Lazy-load SmolVLM model only when image reasoning is needed."""
        if self._vision_model is None:
            try:
                try:
                    from transformers import AutoModelForVision2Seq as _AutoVisionModel
                except Exception:
                    from transformers import AutoModelForImageTextToText as _AutoVisionModel

                model_name = getattr(
                    self.config,
                    'vlm_model_name',
                    'HuggingFaceTB/SmolVLM-Instruct',
                )
                self._processor = AutoProcessor.from_pretrained(
                    model_name,
                    backend="pil",
                )
                self._vision_model = _AutoVisionModel.from_pretrained(
                    model_name,
                    torch_dtype="auto",
                    device_map="auto",
                )
                self._vision_device = next(self._vision_model.parameters()).device
            except Exception as e:
                print(f"Warning: Could not load vision model: {e}")
                self._vision_model = None
                self._processor = None
                self._vision_device = None

    def _invoke_text(self, prompt: str) -> str:
        return self.text_llm.invoke(prompt).content

    def summarize(self, problems, shot_qids, qid, cur_ans) -> str:
        problem = problems[qid]
        question = problem['question']
        choices = problem["choices"]
        answer = problem['answer']
        image = problem.get('image', '')
        caption = problem.get('caption', '')
        split = problem.get("split", "test")
        
        most_ans = self.get_most_common_answer(cur_ans)   
        
        if len(most_ans) == 1:
            prediction = self.get_result(most_ans[0])  # 'A', ..., 'E'
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
        """
        Get the most common answer from the list of answers.
        """
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
            output = self._invoke_text(prompt)
        else:
            output = self.smallvlm_reasoning(prompt, image_path)
            if output:
                print(f"**** output: {output}")
                output = self._invoke_text(
                    f"{output[0]} Summary the above information with format "
                    f"'Answer: The answer is A, B, C, D, E or FAILED.\n BECAUSE: '"
                )
            else:
                # Fallback to text-only if vision model fails
                output = self._invoke_text(prompt)
        return output
    
    def get_result(self, output):
        """Extract the answer letter from model output."""
        pattern = re.compile(r'The answer is ([A-E])')
        res = pattern.findall(output)
        if len(res) == 1:
            answer = res[0]  # 'A', 'B', ...
        else:
            answer = "FAILED"
        return answer

    def get_pred_idx(self, prediction, choices, options):
        """
        Get the index (e.g. 2) from the prediction (e.g. 'C')
        """
        if prediction in options[:len(choices)]:
            return options.index(prediction)
        else:
            return random.choice(range(len(choices)))

    def smallvlm_reasoning(self, prompt, image_path):
        """Use SmolVLM model for multimodal reasoning with image."""
        self._load_vision_model()
        if self._vision_model is None or self._processor is None:
            print("Warning: Vision model not available, falling back to text-only.")
            return None
        if not os.path.exists(image_path):
            print(f"Warning: image not found at {image_path}, falling back to text-only.")
            return None

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        chat_text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image = Image.open(image_path).convert("RGB")
        inputs = self._processor(
            text=[chat_text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )
        target_device = self._vision_device or torch.device("cuda")
        inputs = {k: v.to(target_device) for k, v in inputs.items()}

        generated_ids = self._vision_model.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        output_text = self._processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text
