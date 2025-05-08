import os
import re
import torch
from types import SimpleNamespace
import json
from typing import Optional, List, Dict


from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

import logging

logger = logging.getLogger(__name__)

FALLBACK_CONTEXT_SIZE = 256

class HFLocalWrapper:

    def __init__(self, model_id, max_new_tokens):
        api_key = os.getenv("HUGGINGFACE_API_KEY")
        if api_key is None:
            raise ValueError("HUGGINGFACE_API_KEY environment variable is not set.")

        self.model_id = model_id
        self.max_new_tokens = max_new_tokens


        self.tokenizer = AutoTokenizer.from_pretrained(((self.model_id)), token=api_key, device_map="auto",
                                                torch_dtype="auto", verbose=False)

        self.model_config = AutoConfig.from_pretrained(self.model_id, token=api_key)

        if hasattr(self.model_config, 'max_position_embeddings'):  # this is the standard attribute used by most
            self.context_size = self.model_config.max_position_embeddings
        elif hasattr(self.model_config, 'n_positions'):  # some models may have their context size under this attribute
            self.context_size = self.model_config.n_positions
        else:  # few models, especially older ones, might not have their context size in the config
            self.context_size = FALLBACK_CONTEXT_SIZE


        if not self.tokenizer.pad_token_id:  # if not set, pad_token_id is None
            # preemptively set pad_token_id to eos_token_id as automatically done to prevent warning at each generation:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, token=api_key, device_map="auto", torch_dtype="auto")
        # check if model's generation_config has pad_token_id set:
        if not self.model.generation_config.pad_token_id:
            # set pad_token_id to tokenizer's eos_token_id to prevent excessive warnings:
            self.model.generation_config.pad_token_id = self.tokenizer.eos_token_id

        self.device = "cuda" if torch.cuda.is_available() else "cpu"     
        self.temperature = 0

        self.eos_to_cull = {
            "meta-llama/Llama-3.2-1B-Instruct": "<\\|eot_id\\|>",
            "meta-llama/Llama-3.2-3B-Instruct": "<\\|eot_id\\|>",
            "meta-llama/Meta-Llama-3.1-8B-Instruct": "<\\|eot_id\\|>",
            "meta-llama/Llama-3.3-70B-Instruct": "<\\|eot_id\\|>",
            "Qwen/Qwen2.5-7B-Instruct": "<\\|im_end\\|>",
            "Qwen/Qwen2.5-32B-Instruct": "<\\|im_end\\|>",
            "meta-llama/llama-2-70b-chat-hf": "</s>",
            "meta-llama/llama-2-13b-chat-hf": "</s>"
        }

        logger.info(f"Model loaded successfully. name: {self.model_id} context size: {self.context_size} device: {self.device}")
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        temperature = 0
        messages = prompt
        tool_schema = None
        request_timeout=10
        result = self.generate_response(temperature, messages, tool_schema, request_timeout)
        return result['choices'][0]['message']['content']

    def get_max_tokens(self):
        return 100

    def set_temperature(self, temperature):
        self.temperature = temperature

    def get_temperature(self):
        return self.temperature


    def to_namespace(self, obj):
        if isinstance(obj, dict):
            return SimpleNamespace(**{k: self.to_namespace(v) for k, v in obj.items()})
        elif isinstance(obj, list):
            return [self.to_namespace(i) for i in obj]
        else:
            return obj        

    def generate_response(self, temperature, messages, tool_schema, request_timeout):

        self.set_temperature(temperature)

        prompt_tokens = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                                              return_tensors="pt")
        prompt_tokens = prompt_tokens.to(self.device)
        #prompt_text = self.tokenizer.batch_decode(prompt_tokens['input_ids'])[0]
        prompt_text = self.tokenizer.batch_decode(prompt_tokens)[0]

        prompt = {"inputs": prompt_text, "max_new_tokens": self.get_max_tokens(),
                  "temperature": self.get_temperature(), "return_full_text": False}
        
        do_sample: bool = False
        if self.get_temperature() > 0.0:
            do_sample = True       

        if do_sample:
            model_output_ids = self.model.generate(
                prompt_tokens,
                temperature=self.get_temperature(),
                max_new_tokens=self.get_max_tokens(),
                do_sample=do_sample
            )
        else:
            model_output_ids = self.model.generate(
                prompt_tokens,
                max_new_tokens=self.get_max_tokens(),
                do_sample=do_sample
            )   

        model_output = self.tokenizer.batch_decode(model_output_ids)[0]

        response_text = model_output.replace(prompt_text, '').strip()
        response_text = response_text.replace("<|im_start|>", '').strip()
        response_text = response_text.replace("<tool_call>", '').replace("</tool_call>", '').strip()
        response_text = response_text.replace("<|python_tag|>", '').strip()
        response_text = response_text.replace("<|eom_id|>", '').strip()
        response_text = response_text.replace("<|endoftext|>", '').strip()
        

        eos_to_cull = self.eos_to_cull[self.model_id]
        response_text = re.sub(eos_to_cull, "", response_text)

        completion_dict = {"model": self.model_id, "usage": {}, "choices":[{'message':{'content': response_text}}]}
        completion = self.to_namespace(completion_dict)
        logger.info(f"Completion - {completion}")
        return completion