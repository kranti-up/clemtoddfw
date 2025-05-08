import json

#from clemgame import get_logger
from dialogue_systems.monolithicsys.utils import cleanupanswer
from processfunccallresp import ProcessFuncCallResp

import logging

logger = logging.getLogger(__name__)

class MonoLLM:
    def __init__(self, model_id, prompts_dict, player_dict, resp_json_schema):
        self.model_id = model_id
        self.prompts_dict = prompts_dict
        self.player_b = player_dict["monollm_player"]
        self.resp_json_schema = resp_json_schema
        self.extracted_slots = None
        self.booking_slots = None
        self.func_name = None
        self.tool_call_id = None
        self.tool_calls_list = []

        self.player_b.history.append({"role": "user", "content": prompts_dict["prompt_b"]})
        self.processresp = ProcessFuncCallResp()

    def run(self, utterance, current_turn):

        if self.ishfmodel():
            tool_content = {"name": self.func_name, "content": utterance}
            self._append_utterance("tool", tool_content)

        else:
            #self._append_utterance("user", utterance)
            tool_content = {"content": utterance, 'tool_call_id': self.tool_call_id}
            self._append_utterance('tool', tool_content)
            self.tool_call_id = None


        if self.tool_calls_list:
            answer_cleanup = self.tool_calls_list[0]
            self.tool_calls_list = self.tool_calls_list[1:]

            promptlogs = [({"role": "assistant", "content": f"model response before processing: {answer_cleanup}"})]

            llm_response, error, ret_func_data = self.processresp.run(answer_cleanup, "monolithic_llm")

            if error:
                promptlogs.append({"role": "assistant", "content": f"error while parsing the data: {error}"})

            else:
                if self.ishfmodel():
                    #This will be used in next turn to append with the role: tool
                    self.func_name = ret_func_data["name"]
                    tool_content = [{"type": "function", "function": {"name": ret_func_data["name"], "arguments": ret_func_data["arguments"]}}]
                    self._append_utterance("assistant", tool_content)
                else:
                    #self._append_utterance("assistant", answer_cleanup[0])
                    self.tool_call_id = answer_cleanup['id']


            promptlogs.append({'role': "monollm", 'content': {'prompt': "processing model tool calls", 'raw_answer': answer_cleanup,
                                                                        'answer': llm_response}})

            return promptlogs, answer_cleanup, llm_response            

        tool_schema = self.resp_json_schema

        prompt, raw_answer, answer = self.player_b(self.player_b.history,
                                                   current_turn, tool_schema, None)
        
        promptlogs = [({"role": "assistant", "content": f"model response before processing: {answer}"})]
        answer_cleanup = cleanupanswer(answer)
        logger.info(f"Answer from the model: {answer}, {type(answer)}, answer_cleanup = {answer_cleanup}, {type(answer_cleanup)}")
        
        if not self.ishfmodel():
            self._append_utterance("assistant", raw_answer['tool_calls'])
            #self.tool_call_id = answer_cleanup[0]['id']
            
        if isinstance(answer_cleanup, str):
            try:
                answer_cleanup = json.loads(answer_cleanup)

                if isinstance(answer_cleanup, dict):
                    answer_cleanup = [answer_cleanup]
                    
            except Exception as error:
                logger.error(f"Error while converting the answer_cleanup from str to json: {error}")
                answer_cleanup = answer_cleanup

        if isinstance(answer_cleanup, list):
            if len(answer_cleanup) > 1:
                self.tool_calls_list = answer_cleanup[1:]

        llm_response, error, ret_func_data = self.processresp.run(answer_cleanup[0], "monolithic_llm")

        if error:
            promptlogs.append({"role": "assistant", "content": f"error while parsing the data: {error}"})

        else:
            if self.ishfmodel():
                #This will be used in next turn to append with the role: tool
                self.func_name = ret_func_data["name"]
                tool_content = [{"type": "function", "function": {"name": ret_func_data["name"], "arguments": ret_func_data["arguments"]}}]
                self._append_utterance("assistant", tool_content)
            else:
                #self._append_utterance("assistant", answer_cleanup[0])
                self.tool_call_id = answer_cleanup[0]['id']


        promptlogs.append({'role': "monollm", 'content': {'prompt': prompt, 'raw_answer': raw_answer,
                                                                    'answer': llm_response}})

        return promptlogs, raw_answer, llm_response


    def ishfmodel(self):
        #This response formatting is not helping, hence disabled this
        #return False
        return True if any(model in self.model_id for model in ["Qwen", "Llama"]) else False


    def _append_utterance(self, role, data):
        if role == "assistant":
            if self.ishfmodel():
                self.player_b.history.append({"role": role, "tool_calls": data})
            else:
                #self.player_b.history.append({"role": role, "content": data})
                self.player_b.history.append(data)
        elif role == "user":
            if len(self.player_b.history) == 1:
                #TODO: check for cases, where player_b.history is empty
                self.player_b.history[-1]["content"] += "\n\n" + data
                self.player_b.history[-1]["content"] = self.player_b.history[-1]["content"].strip()
            else:
                if "DATABASE RETRIEVAL RESULTS:" in data:
                    turn_prompt = self.prompts_dict["dbquery_prompt_b"]
                elif "BOOKING VALIDATION STATUS:" in data:
                    turn_prompt = self.prompts_dict["validbooking_prompt_b"]
                else:
                    turn_prompt = self.prompts_dict["turn_prompt_b"]

                self.player_b.history.append({"role": role, "content": turn_prompt + "\n\n" + data})
        elif role == "tool":
            if len(self.player_b.history) == 1:
                self.player_b.history[-1]["content"] += "\n\n" + data["content"]
                self.player_b.history[-1]["content"] = self.player_b.history[-1]["content"].strip()
            else:
                if self.ishfmodel():                
                    #self.player_b.history.append({"role": role, "name": data["name"], "content": data["content"]}) 
                    tool_content = {"name": data["name"], "content": data["content"]}   
                    self.player_b.history.append({"role": role, 'content': tool_content})            
                else:
                    self.player_b.history.append({"role": role, "tool_call_id": data["tool_call_id"], 'content': data["content"]})

    def get_booking_data(self):
        #Since this is Monolithic LLM System, the data is not stored in the system, instead it is on the LLM side to return this info in booking call
        return {}
    
    def get_entity_slots(self):
        #Since this is Monolithic LLM System, the data is not stored in the system, instead it is on the LLM side to return this info in booking call
        return {}
