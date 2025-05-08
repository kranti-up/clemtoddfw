import copy
from typing import Dict, Any, List, Tuple
import json

#import clemgame
#from clemcore import get_logger
from dialogue_systems.modllmdsys.players import ModLLMSpeaker
from utils import cleanupanswer

import logging

logger = logging.getLogger(__name__)


class FollowupGenerator:
    def __init__(self, model_name, model_spec, response_gen_prompt):
        self.model_name = model_name
        self.model_spec = model_spec
        self.base_prompt = response_gen_prompt
        self.player = None
        self.tool_schema = None
        self._setup()

    def _prepare_response_tool_schema(self):
        self.tool_schema = [
            {
                "type": "function",
                "function": {
                    "name": "followupresponse",
                    "description": "Use this function to respond to the user with follow-up messages. This includes asking for missing or unclear information, confirming details, sharing booking reference numbers, or continuing the dialogue based on the current conversation state.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "response_generation": {
                                "type": "string",
                                "description": "The response from the dialogue system to the user"
                            }
                        },
                        "required": ["response_generation"],
                        "additionalProperties": False
                    }
                }
            }
        ]

    def _setup(self) -> None:
        # If the model_name is of type "LLM", then we need to instantiate the player
        # TODO: Implement for other model types ("Custom Model", "RASA", etc.)
        self.player = ModLLMSpeaker(self.model_spec, "followup_generation", "", {})
        #self.player.history.append({"role": "user", "content": self.base_prompt})
        self._prepare_response_tool_schema()

    def run(self, utterance: Dict, turn_idx: int) -> str:
        self.player.history = [{"role": "user", "content": self.base_prompt}] 
        message = json.dumps(utterance) if isinstance(utterance, Dict) else utterance
        self.player.history[-1]["content"] += message

        '''
        if self.player.history[-1]["role"] == "user":
            self.player.history[-1]["content"] += message
        else:
            self.player.history.append({"role": "user", "content": message})
        '''
        prompt, raw_response, raw_answer = self.player(self.player.history, turn_idx, self.tool_schema, None)
        logger.info(f"Follow-up generator raw response:\n{raw_answer}")
        answer_cleaned =  cleanupanswer(raw_answer)
        #self.player.history.append({"role": "assistant", "content": json.dumps(answer)})
        parsed_answer = self._parse_response(answer_cleaned)

        return prompt, raw_response, raw_answer, parsed_answer

    def _parse_response(self, answer):
        logger.info(f"Answer from the model: {answer}, {type(answer)}")        
        if isinstance(answer, list):
            #return f"Invalid response type. {type(answer)}. Expected dict"            
            answer = answer[0]

        if not isinstance(answer, dict):
            return f"Invalid response type. {type(answer)}. Expected dict"

        if "name" not in answer or answer["name"] != "followupresponse":
            return f"function name ( followupresponse ) is missing in the response {answer}. Cannot proceed."

        use_args_key = None
        if "arguments" in answer:
            use_args_key = "arguments"

        elif "parameters" in answer:
            use_args_key = "parameters"

        else:
            return f"function arguments is missing in the response {answer}. Cannot proceed."

        result = answer[use_args_key]
        if "response_generation" not in result:
            return f"response_generation is not available in the response {result}. Cannot proceed."        

        return result["response_generation"]


    def get_history(self):
        return self.player.history
    
    def clear_history(self):
        self.player.history = [{"role": "user", "content": self.base_prompt}] 
