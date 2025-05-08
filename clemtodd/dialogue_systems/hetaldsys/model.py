from typing import Any, Dict
import os
import openai
import anthropic
from huggingface_hub import InferenceClient
import logging


openai.api_key = os.environ.get('OPENAI_API_KEY', None)
anthropic.api_key = os.environ.get('ANTHROPIC_API_KEY', None)
hf_api_key = os.environ.get('HF_API_KEY', None)

logger = logging.getLogger(__name__)

from dialogue_systems.hetaldsys.prompts import FewShotPrompt, SimpleTemplatePrompt

class SimplePromptedLLM:
    def __init__(self, model, tokenizer, type='seq2seq'):
        #print(f"Inside SimplePromptedLLM model = {model}")
        self.model = model
        self.tokenizer = tokenizer
        self.type = type

    def __call__(self, prompt: SimpleTemplatePrompt, predict=True, **kwargs: Any):
        filled_prompt = prompt(**kwargs)
        prompt, raw_answer, answer = self._predict(filled_prompt, **kwargs) if predict else None
        return prompt, raw_answer, answer#prediction, filled_prompt

    def _predict(self, text, **kwargs):
        input_ids = self.tokenizer.encode(text,return_tensors="pt").to(self.model.device)
        max_length = max_new_tokens = 50
        if self.type == 'causal':
            max_length = input_ids.shape[1] + max_length
        output = self.model.generate(input_ids,
                                     do_sample=True,
                                     top_p=0.9,
                                     max_new_tokens=max_new_tokens,
                                     temperature=0.1)
        if self.type == 'causal':
            output = output[0, input_ids.shape[1]:]
        else:
            output = output[0]
        output = self.tokenizer.decode(output, skip_special_tokens=True)
        return output


class FewShotPromptedLLM(SimplePromptedLLM):
    def __init__(self, model, tokenizer, type='seq2seq'):
        #print(f"Inside FewShotPromptedLLM model = {model}")
        super().__init__(model, tokenizer, type)     

    def __call__(self, prompt: FewShotPrompt, positive_examples: list[Dict], negative_examples: list[Dict], predict=True, **kwargs: Any):
        filled_prompt = prompt(positive_examples, negative_examples, **kwargs)
      #  if len(filled_prompt) > 500:
       #     filled_prompt = prompt(positive_examples[:1], negative_examples[:1], **kwargs)
        prediction = self._predict(filled_prompt, **kwargs) if predict else None
        return prediction, filled_prompt


class FewShotOpenAILLM(FewShotPromptedLLM):
    def __init__(self, model_name):
        #print(f"Inside FewShotOpenAILLM model_name = {model_name}")
        super().__init__(None, None)
        self.model_name = model_name

        if "claude" in model_name:
            self.api_type = "ANTHROPIC"
        elif "gpt" in model_name:
            self.api_type = "OPENAI"
        else:
            self.api_type = "OPENAI_COMPLAINT"    
            self.client = InferenceClient(base_url = "",
                            api_key="")                  


    def _predict(self, text, **kwargs):
        try:
            if self.api_type == "OPENAI":
                completion = openai.OpenAI(api_key=openai.api_key).chat.completions.create(
                    model=self.model_name,
                    prompt=text,
                    temperature=0,
                )
                return completion.choices[0].message.content.strip()
            elif self.api_type == "ANTHROPIC":
                completion = anthropic.Anthropic(api_key=anthropic.api_key).messages.create(
                    model=self.model_name,
                    messages=text,
                    temperature=0,
                    max_tokens=500
                )
                return completion.content[0].text.strip()
            elif self.api_type == "OPENAI_COMPLAINT":
                completion = self.client.chat.completions.create(
                    model=self.model_name, 
                    messages=text, 
                    temperature=0
                )
                return completion.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Exception in FewShotOpenAILLM = {e}")
            return ""


class FewShotOpenAIChatLLM(FewShotOpenAILLM):
    def _predict(self, text, **kwargs):
        #print(f"Inside FewShotOpenAIChatLLM text = {text}") 
        try:
            if self.api_type == "OPENAI":
                completion = openai.OpenAI(api_key=openai.api_key).chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": text}
                    ],
                    temperature=0,
                    )
                return completion.choices[0].message.content.strip()
            elif self.api_type == "ANTHROPIC":
                completion = anthropic.Anthropic(api_key=anthropic.api_key).messages.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": text}
                    ],
                    temperature=0,
                    max_tokens=500
                )
                return completion.content[0].text.strip()
            elif self.api_type == "OPENAI_COMPLAINT":               
                completion = self.client.chat.completions.create(
                    model=self.model_name, 
                    messages=[
                        {"role": "user", "content": text}
                    ], 
                    temperature=0
                )
                return completion.choices[0].message.content.strip()            
        except Exception as e:
            logger.error(f"Exception in FewShotOpenAIChatLLM = {e}")
            return ""            

        
        #print(f"Inside FewShotOpenAIChatLLM completion = {completion}")       



class ZeroShotOpenAILLM(SimplePromptedLLM):
    def __init__(self, model_name, player_b):
        super().__init__(None, None)

        self.model_name = model_name
        self.player_b = player_b

        '''
        if "claude" in model_name:
            self.api_type = "ANTHROPIC"
        elif "gpt" in model_name:
            self.api_type = "OPENAI"
        else:
            self.api_type = "OPENAI_COMPLAINT"        
            self.client = InferenceClient(base_url = "",
                            api_key="")
        '''

    def _predict(self, text, current_turn, **kwargs):
        logger.info(f"Inside ZeroShotOpenAILLM _predict text = {text} self.model_name = {self.model_name}")        
        try:
            self.player_b.history= [{"role": "user", "content": ""}]
            self.player_b.history[-1]["content"] += text

            prompt, raw_answer, answer = self.player_b(self.player_b.history, current_turn, None, None)
            return prompt, raw_answer, answer    
        except Exception as e:
            logger.error(f"Exception in ZeroShotOpenAILLM = {e}")
            return ""


    def _predict_old(self, text, **kwargs):
        try:
            if self.api_type == "OPENAI":
                completion = openai.OpenAI(api_key=openai.api_key).chat.completions.create(
                    model=self.model_name,
                    prompt=text,
                    temperature=0,
                )
                return completion.choices[0].message.content.strip()
            elif self.api_type == "ANTHROPIC":
                completion = anthropic.Anthropic(api_key=anthropic.api_key).messages.create(
                    model=self.model_name,
                    messages=text,
                    temperature=0,
                    max_tokens=500
                )
                return completion.content[0].text.strip()
            elif self.api_type == "OPENAI_COMPLAINT":
                completion = self.client.chat.completions.create(
                    model=self.model_name, 
                    messages=text, 
                    temperature=0
                )
                return completion.choices[0].message.content.strip()            
        except Exception as e:
            logger.error(f"Exception in ZeroShotOpenAILLM = {e}")
            return ""


class ZeroShotOpenAIChatLLM(ZeroShotOpenAILLM):
    def _predict(self, text, current_turn, **kwargs):
        logger.info(f"Inside ZeroShotOpenAIChatLLM _predict text = {text} self.model_name = {self.model_name}")
        try:
            self.player_b.history= [{"role": "user", "content": ""}]
            self.player_b.history[-1]["content"] += text
            prompt, raw_answer, answer = self.player_b(self.player_b.history, current_turn, None, None)
            return prompt, raw_answer, answer
        except Exception as e:
            logger.error(f"Exception in ZeroShotOpenAIChatLLM = {e}")
            return ""


    def _predict_old(self, text, **kwargs):
        try:
            #print(f"Inside ZeroShotOpenAIChatLLM text = {text} self.model_name = {self.model_name}")
            if self.api_type == "OPENAI":
                completion = openai.OpenAI(api_key=openai.api_key).chat.completions.create(
                    model=self.model_name,
                    messages=[
                    #  {"role": "system", "content": prefix},
                        {"role": "user", "content": text}
                    ],
                    temperature=0,
                    )
                #print(f"Inside ZeroShotOpenAIChatLLM completion = {completion}")
                return completion.choices[0].message.content.strip()
            elif self.api_type == "ANTHROPIC":
                completion = anthropic.Anthropic(api_key=anthropic.api_key).messages.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": text}
                    ],
                    temperature=0,
                    max_tokens=500
                )
                #print(f"Inside ZeroShotOpenAIChatLLM completion = {completion}")
                return completion.content[0].text.strip()
            elif self.api_type == "OPENAI_COMPLAINT":
                completion = self.client.chat.completions.create(
                    model=self.model_name, 
                    messages=[
                        {"role": "user", "content": text}
                    ], 
                    temperature=0
                )
                return completion.choices[0].message.content.strip()            
        except Exception as e:
            print(f"Exception in ZeroShotOpenAIChatLLM = {e}")
            return ""


class FewShotAlpaca(FewShotPromptedLLM):
    def __init__(self, model_name):
        super().__init__(None, None)
        from alpaca import predict as predict_alpaca
        self._predict = predict_alpaca
        self.model_name = model_name

    def _predict(self, text, **kwargs):
        return self._predict(text)


class ZeroShotAlpaca(SimplePromptedLLM):
    def __init__(self, model_name):
        super().__init__(None, None)
        from alpaca import predict as predict_alpaca
        self._predict = predict_alpaca
        self.model_name = model_name

    def _predict(self, text, **kwargs):
        return self._predict(text)


