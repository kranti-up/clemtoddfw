import time
import copy
from typing import Dict
import json

#from clemgame import get_logger
from utils import cleanupanswer, funcdatasanitycheck, preparemodelresponse
from dialogue_systems.modllmdsys.players import ModLLMSpeaker
from dialogue_systems.modllmdsys.intentdetector import IntentDetector
from dialogue_systems.modllmdsys.slotextractor import SlotExtractor
from dialogue_systems.modllmdsys.followupgenerator import FollowupGenerator
from dialogue_systems.modprogdsys.dbqueryformatter import DBQueryFormatter
from dialogue_systems.modprogdsys.bookingformatter import BookingFormatter
from processfunccallresp import ProcessFuncCallResp

import logging

logger = logging.getLogger(__name__)


class ModLLMDM:
    def __init__(self, model_name, model_spec, prompts_dict, player_dict, resp_json_schema, liberal_processing, booking_mandatory_keys) -> None:
        self.model_name = model_name
        self.model_spec = model_spec
        self.prompts_dict = prompts_dict

        self.liberal_processing = liberal_processing
        self.booking_data = {}
        self.slotdata = {}
        self.promptlogs = []
        self.dhistory = []
        self.max_reprobe = 3
        self.cur_reprobe = 0
        self.func_name = None
        self.tool_call_id = None
        self.tool_calls_list = []        


        self.respformat = resp_json_schema#["schema"]
        self.booking_mandatory_keys = booking_mandatory_keys

        self.player_b = player_dict["modllm_player"]
        self.player_b.history.append({"role": "user", "content": prompts_dict["prompt_b"]})
        self.turn_ss_prompt_player_b = prompts_dict["turn_ss_prompt_b"]
        self.turn_prompt_player_b = prompts_dict["turn_prompt_b"]
        self._create_subsystems(model_name, model_spec, prompts_dict)

        self.liberalcount = {"intent": 0, "slot": 0, "response": 0, "aggregator": 0}
        self.subsystemnamemap = {"intent_detector": "intent", "slot_extractor": "slot", 
                                 "response_generator": "response"}
        self.processresp = ProcessFuncCallResp()

        self._prepare_tool_schema()

    def _create_subsystems(self, model_name, model_spec, prompts_dict):
        self.intentdet = IntentDetector(model_name, model_spec, prompts_dict["intent_detection"])
        self.slotext = SlotExtractor(model_name, model_spec, prompts_dict["slot_extraction"], self.respformat)
        self.followupgen = FollowupGenerator(
            model_name, model_spec, prompts_dict["followup_generation"]
        )
        self.dbqueryformatter = DBQueryFormatter(
            model_name, model_spec, prompts_dict["dbquery_formatter"], self.respformat
        )
        self.bookingformatter = BookingFormatter(
            model_name, model_spec, prompts_dict["booking_formatter"], self.respformat
        )

    def _prepare_tool_schema(self):
        self.tool_schema = [
            {
                "type": "function",
                "function": {
                    "name": "followup",
                    "description": "Use this function to respond to the user's request as a final message after coordinating with the dialogue sub-systems. It is typically used to present the output of a sub-system (e.g., response_generator), such as clarifications, confirmations, or booking details (e.g., sharing reference numbers). This function serves as the interface to continue or complete the conversation with the user based on the current dialogue state and subsystem outputs. Do not use this function for database look-up or validating booking data.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": "The response from the dialogue system to the user"
                            }
                        },
                        "required": ["message"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "processnextsubsystem",
                    "description": "Use this function to route the flow to the appropriate sub-system (intent detection, slot extraction, or response generation) in the conversational pipeline. Each sub-system receives the necessary input and optional dialogue history for effective processing. Do not use this function for database look-up or validating booking data.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "next_subsystem": {
                                "type": "string",
                                "enum": ["intent_detector", "slot_extractor", "response_generator"],
                                "description": "Specifies which sub-system to route the dialogue to next. Valid values: intent_detector: Identifies the user's intent (e.g., inquiry, clarification, booking); slot_extractor: Extracts key details required to fulfill the intent (e.g., date, time, location).; response_generator: Produces user-facing responses such as clarification questions or confirmations."
                            },
                            "input_data": {
                                "type": "object",                                
                                "description": "The core input payload required by the specified subsystem. Its structure may vary depending on the subsystem type (e.g., a user utterance for 'intent_detector', user utterance and extracted intent for 'slot_extractor')."
                            },
                            "dialogue_history": {
                                "type": "array",
                                "items": { "type": "string" },                                                          
                                "description": "A chronological list of previous user-bot exchanges to provide contextual awareness for the subsystem. Useful for handling context-dependent interpretations or responses. Optional."
                            }
                        },
                        "required": ["next_subsystem", "input_data"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "retrievefromrestaurantdb",
                    "description": "Use this function to query the restaurant database and retrieve restaurants that match optional filters such as area, pricerange, food (cuisine), or restaurant name. This function is typically used to find available restaurant options before validating or making a reservation. Returns up to 5 matching restaurants, or fewer if less than 5 matches are found.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "area": {
                                "type": "string",
                                "enum": ["centre", "north", "east", "west", "south"],
                                "description": "The area/location/place of the restaurant. Optional."
                            },
                            "pricerange": {
                                "type": "string",
                                "enum": ["cheap", "moderate", "expensive"],
                                "description": "The price budget for the restaurant. Optional."
                            },
                            "food": {
                                "type": "string",
                                "description": "The cuisine of the restaurant you are looking for. Optional."
                            },
                            "name": {
                                "type": "string",
                                "description": "The name of the restaurant. Optional."
                            }
                        },
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "retrievefromhoteldb",
                    "description": "Use this function to query the hotel database and retrieve hotels/guesthouses that match optional filters such as area, pricerange, type, hotel name, internet, parking, or stars. This function is typically used to find available hotel options before validating or making a reservation. Returns up to 5 matching hotels, or fewer if less than 5 matches are found.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "area": {
                                "type": "string",
                                "enum": ["centre", "north", "east", "west", "south"],
                                "description": "The area/location/place of the hotel. Optional."
                            },
                            "pricerange": {
                                "type": "string",
                                "enum": ["cheap", "moderate", "expensive"],
                                "description": "The price budget for the hotel. Optional."
                            },
                            "type": {
                                "type": "string",
                                "enum": ["hotel", "guesthouse"],
                                "description": "What is the type of the hotel. Optional."
                            },
                            "name": {
                                "type": "string",
                                "description": "The name of the hotel. Optional."
                            },
                            "internet": {
                                "type": "string",
                                "enum": ["yes", "no"],
                                "description": "Indicates, whether the hotel has internet/wifi or not. Optional."
                            },
                            "parking": {
                                "type": "string",
                                "enum": ["yes", "no"],
                                "description": "Indicates, whether the hotel has parking or not. Optional."
                            },
                            "stars": {
                                "type": "object",
                                "description": "The star rating of the hotel. Optional.",
                                "properties": {
                                    "operator": { "type": "string", "enum": ["=", ">=", "<=", ">", "<"] },
                                    "value": { "type": "string", "enum": ["1", "2", "3", "4", "5"] }
                                },
                                "required": ["operator", "value"],
                                "additionalProperties": False
                            }
                        },
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "retrievefromtraindb",
                    "description": "Use this function to query the train database and retrieve trains that match optional filters such as destination, departure, day, arriveby, or leaveat. This function is typically used to find available options before validating or making a reservation. Returns up to 5 matching trains, or fewer if less than 5 matches are found.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "destination": {
                                "type": "string",
                                "description": "Destination of the train. Optional."
                            },
                            "departure": {
                                "type": "string",
                                "description": "Departure location of the train. Optional."
                            },
                            "day": {
                                "type": "string",
                                "enum": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"],
                                "description": "Journey day of the train. Optional."
                            },
                            "arriveby": {
                                "type": "object",
                                "description": "Arrival time of the train. Optional.",
                                "properties": {
                                    "operator": { "type": "string", "enum": ["=", ">=", "<=", ">", "<"] },
                                    "value": { "type": "string", "pattern": "^(0[0-9]|1[0-9]|2[0-3]):[0-5][0-9]$",
                                            "description": "A time string formatted as HH:MM (24-hour format)."
                                            }
                                },
                                "required": ["operator", "value"],
                                "additionalProperties": False
                            },
                            "leaveat": {
                                "type": "object",
                                "description": "Leaving time for the train. Optional.",
                                "properties": {
                                    "operator": { "type": "string", "enum": ["=", ">=", "<=", ">", "<"] },
                                    "value": { "type": "string", "pattern": "^(0[0-9]|1[0-9]|2[0-3]):[0-5][0-9]$",
                                            "description": "A time string formatted as HH:MM (24-hour format)."
                                            }
                                },
                                "required": ["operator", "value"],
                                "additionalProperties": False
                            }
                        },
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "validaterestaurantbooking",
                    "description": "Use this function to check the availability of a restaurant based on user preferences such as area, food (cuisine), pricerange, name, people, day, and time before proceeding with a reservation. This function should be called to validate whether a booking can be made with the provided details. If the details are accurate, it returns a booking reference number.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "area": {
                                "type": "string",
                                "enum": ["centre", "north", "east", "west", "south"],
                                "description": "The area/location/place of the restaurant."
                            },
                            "pricerange": {
                                "type": "string",
                                "enum": ["cheap", "moderate", "expensive"],
                                "description": "The price budget for the restaurant."
                            },
                            "food": {
                                "type": "string",
                                "description": "The cuisine of the restaurant you are looking for."
                            },
                            "name": {
                                "type": "string",
                                "description": "The name of the restaurant."
                            },
                            "people": {
                                "type": "string",
                                "enum": ["1", "2", "3", "4", "5", "6", "7", "8"],
                                "description": "Number of people for the restaurant reservation."
                            },
                            "day": {
                                "type": "string",
                                "enum": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"],
                                "description": "Day of the restaurant reservation."
                            },
                            "time": {
                                "type": "string",
                                "pattern": "^(0[0-9]|1[0-9]|2[0-3]):[0-5][0-9]$",
                                "description": "Time of the restaurant reservation, formatted as HH:MM (24-hour format)."
                            },
                            "phone": {
                                "type": "string",
                                "description": "Phone number of the restaurant. Optional."
                            },
                            "postcode": {
                                "type": "string",
                                "description": "Postal code of the restaurant. Optional."
                            },
                            "address": {
                                "type": "string",
                                "description": "Address of the restaurant. Optional."
                            }                          
                        },
                        "required": ["food", "area", "pricerange", "name", "people", "day", "time"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "validatehotelbooking",
                    "description": "Use this function to check the availability of a hotel based on user preferences such as area, type (hotel/guesthouse), pricerange, name, internet, parking, stars, people, day and stay before proceeding with a reservation. This function should be called to validate whether a booking can be made with the provided details. If the details are accurate, tit returns a booking reference number.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "area": {
                                "type": "string",
                                "enum": ["centre", "north", "east", "west", "south"],
                                "description": "The area/location/place of the hotel."
                            },
                            "pricerange": {
                                "type": "string",
                                "enum": ["cheap", "moderate", "expensive"],
                                "description": "The price budget for the hotel."
                            },
                            "type": {
                                "type": "string",
                                "enum": ["hotel", "guesthouse"],
                                "description": "What is the type of the hotel."
                            },
                            "name": {
                                "type": "string",
                                "description": "The name of the hotel."
                            },
                            "internet": {
                                "type": "string",
                                "enum": ["yes", "no"],
                                "description": "Indicates, whether the hotel has internet/wifi or not."
                            },
                            "parking": {
                                "type": "string",
                                "enum": ["yes", "no"],
                                "description": "Indicates, whether the hotel has parking or not."
                            },
                            "stars": {
                                "type": "string",
                                "enum": ["1", "2", "3", "4", "5"],
                                "description": "The star rating of the hotel."
                            },
                            "people": {
                                "type": "string",
                                "enum": ["1", "2", "3", "4", "5", "6", "7", "8"],
                                "description": "Number of people for the hotel booking."
                            },
                            "day": {
                                "type": "string",
                                "enum": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"],
                                "description": "Day of the hotel booking."
                            },
                            "stay": {
                                "type": "string",
                                "enum": ["1", "2", "3", "4", "5", "6", "7", "8"],
                                "description": "Length of stay at the hotel."
                            },
                            "phone": {
                                "type": "string",
                                "description": "Phone number of the hotel. Optional."
                            },
                            "postcode": {
                                "type": "string",
                                "description": "Postal code of the hotel. Optional."
                            },
                            "address": {
                                "type": "string",
                                "description": "Address of the hotel. Optional."
                            }
                        },
                        "required": ["area", "pricerange", "type", "internet", "parking", "name", "stars", "people", "day", "stay"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "validatetrainbooking",
                    "description": "Use this function to check the availability of a train based on user preferences such as destination, departure, arriveby, leaveat, day, people, and trainid before proceeding with a reservation. This function should be called to validate whether a booking can be made with the provided details. If the details are accurate, it returns a booking reference number.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "destination": {
                                "type": "string",
                                "description": "Destination of the train."
                            },
                            "departure": {
                                "type": "string",
                                "description": "Departure location of the train."
                            },
                            "day": {
                                "type": "string",
                                "enum": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"],
                                "description": "Journey day of the train."
                            },
                            "arriveby": {
                                "type": "string",
                                "pattern": "^(0[0-9]|1[0-9]|2[0-3]):[0-5][0-9]$",
                                "description": "Arrival time of the train."
                            },
                            "leaveat": {
                                "type": "string",
                                "pattern": "^(0[0-9]|1[0-9]|2[0-3]):[0-5][0-9]$",
                                "description": "Leaving time for the train."
                            },
                            "people": {
                                "type": "string",
                                "enum": ["1", "2", "3", "4", "5", "6", "7", "8"],
                                "description": "Number of train tickets for the booking."
                            },
                            "trainid": {
                                "type": "string",
                                "description": "ID of the train."
                            },
                            "price": {
                                "type": "string",
                                "description": "Price of the train journey. Optional."
                            },
                            "duration": {
                                "type": "string",
                                "description": "Duration of the travel. Optional."
                            }
                        },
                        "required": ["destination", "departure", "day", "arriveby", "leaveat", "people", "trainid"],
                        "additionalProperties": False
                    }
                }
            }           
        ]        



    def _append_utterance(self, subsystem: str, data: str, role: str) -> None:
        """Add an utterance to the history of a player (firstlast specific)."""

        message = data
        if isinstance(data, dict) or isinstance(data, list):
            message = json.dumps(data)

        if role == "assistant":
            if self.ishfmodel():
                #don't use message - it is in str format - we need dict format
                self.player_b.history.append({"role": role, "tool_calls": data})
            else:
                #self.player_b.history.append({"role": role, "content": message})
                self.player_b.history.append(data)

        elif role == "user":
            if subsystem is None:
                if len(self.player_b.history) == 1:
                    #TODO: check for cases, where player_b.history is empty
                    self.player_b.history[-1]["content"] += "\n\n" + message
                    self.player_b.history[-1]["content"] = self.player_b.history[-1]["content"].strip()
                else:
                    if "DATABASE RETRIEVAL RESULTS:" in message:
                        turn_prompt = self.prompts_dict["dbquery_prompt_b"]
                    elif "BOOKING VALIDATION STATUS:" in message:
                        turn_prompt = self.prompts_dict["validbooking_prompt_b"]
                    else:
                        turn_prompt = self.prompts_dict["turn_prompt_b"]

                    self.player_b.history.append({"role": role, "content": turn_prompt + "\n\n" + message})                    
            else:
                if message:
                    turn_prompt = self.turn_ss_prompt_player_b.replace(
                        "$sub-system", subsystem
                    )
                    turn_prompt += "\n\n" + message
                else:
                    turn_prompt = subsystem
                self.player_b.history.append({"role": role, "content": turn_prompt.strip()})
        elif role == "tool":
            if len(self.player_b.history) == 1:
                self.player_b.history[-1]["content"] += "\n\n" + data['content']
                self.player_b.history[-1]["content"] = self.player_b.history[-1]["content"].strip()           
            else:
                if self.ishfmodel():                
                    #self.player_b.history.append({"role": role, "name": data["name"], "content": data["content"]}) 
                    tool_content = {"name": data["name"], "content": data["content"]}   
                    self.player_b.history.append({"role": role, 'content': tool_content})            
                else:
                    self.player_b.history.append({"role": role, "tool_call_id": data["tool_call_id"], 'content': data["content"]})

                #don't use messages - it is in str format. We need dict and the values of name, content
                #self.player_b.history.append({"role": role, "name": data["name"], "content": data["content"]})  



    def _validate_subsystem(self, nextsubsystem: str) -> bool:
        if nextsubsystem in self.subsystemnamemap:
            self.cur_reprobe = 0
            return True, nextsubsystem
        else:
            '''
            if self.liberal_processing:
                for key, value in self.subsystemnamemap.items():
                    if value in nextsubsystem.lower():
                        self.liberalcount[key] += 1
                        self.cur_reprobe = 0
                        return True, key
                if self.cur_reprobe  < self.max_reprobe:
                    self.cur_reprobe += 1
                    return False, "reprobe"
                else:
                    return False, None
            '''
        return False, None
    
    def _validate_subsystem_input(self, sub_system: str, taskinput: Dict) -> Dict:
        logger.info(f"Validating Subsystem Input: {sub_system}-> {taskinput} {type(taskinput)}")
        
        if taskinput is None or isinstance(taskinput, str) or isinstance(taskinput, json.decoder.JSONDecodeError):
            return None
        else:
            return taskinput

    def _validate_subsystem_output(self, sub_system, sub_sys_response):
        if sub_sys_response is None or isinstance(sub_sys_response, json.decoder.JSONDecodeError) or "Cannot proceed." in sub_sys_response:
            return None

        else:
            # The modules does internal processing and returns only the relevant data.
            return sub_sys_response
            '''
            if sub_system == "intent_detector":
                if "intent_detection" in sub_sys_response and "domain" in sub_sys_response:
                    return sub_sys_response
                else:
                    return None
            elif sub_system == "slot_extractor":
                if "slot_extraction" in sub_sys_response:
                    return sub_sys_response
                else:
                    return None

            elif sub_system == "response_generator":
                if "response_generation" in sub_sys_response:
                    return sub_sys_response
                else:
                    return None
            elif sub_system == "dbquery_formatter":
                if "dbquery_format" in sub_sys_response:
                    return sub_sys_response
                else:
                    return None
            elif sub_system == "booking_formatter":
                if "booking_query" in sub_sys_response:
                    return sub_sys_response
                else:
                    return None
            return sub_sys_response            
            '''
        return sub_sys_response     
        

    def ishfmodel(self):
        #This response formatting is not helping, hence disabled this
        #return False
        return True if any(model in self.model_name for model in ["Qwen", "Llama"]) else False     


    def _parse_model_response(self, answer):
        result = cleanupanswer(answer)
        logger.info(f"Player B: Subsystem Flow response after cleaning:\n{result}, {type(result)}")
        if isinstance(result, json.decoder.JSONDecodeError):
            self.promptlogs.append({"role": "assistant",
                                        "content": f"Failure in parsing the model response before processing: {str(result)}"})
            return self.promptlogs, None, str(result)

        if isinstance(result, dict):
            result = [result]

        if isinstance(result, list):
            if len(result) > 1:
                self.tool_calls_list = result[1:]     

        return None, None, result[0]   



    def _parse_next_subsystem(self, result):
        next_subsystem = None
        taskinput = None
        taskcontext = None
        if isinstance(result, dict):
            result_func, error = funcdatasanitycheck(result)

            if error:
                return self.promptlogs, None, error

            self.func_name = result_func.get("name", None)
            self.func_arguments = result_func.get("arguments", None)
            logger.info(f"Function Name: {self.func_name}, arguments: {self.func_arguments}")
            if self.func_name == "processnextsubsystem":
                if self.func_arguments:
                    next_subsystem = self.func_arguments.get("next_subsystem", None)
                    taskinput = self.func_arguments.get("input_data", None)
                    taskcontext = self.func_arguments.get("dialogue_history", None)
                    #num_process_ssystem += 1

                logger.info(f"next_subsystem: {next_subsystem}, taskinput: {taskinput}, taskcontext: {taskcontext}")

            if self.ishfmodel() and self.func_name and self.func_arguments:
                #This will be used in next turn to append with the role: tool
                tool_content = [{"type": "function", "function": {"name": self.func_name, "arguments": self.func_arguments}}]
                self._append_utterance(None, tool_content, "assistant")
            else:
                #self._append_utterance(None, result, "assistant")
                self.tool_call_id = result['id']

            return None, None, {"next_ss": next_subsystem, "taskinput": taskinput, "taskcontext": taskcontext}

        else:
            self.func_name = None
            self.func_arguments = None
            next_subsystem = None
            return self.promptlogs, None, f"Function name and arguments are missing in the model response. Cannot continue processing."

    def _process_next_sub_system(self, next_subsystem, taskinput, taskcontext, current_turn, subsystem_handlers):

        next_subsystem = next_subsystem.lower()
        #taskinput = result.get("input_data", None)
        logger.info(f"Player B: Next SubSystem Input\n{taskinput}")
        self.promptlogs.append({"role": f"Input to {next_subsystem}", 'content': f"Input to {next_subsystem} sub-system:\n{json.dumps(taskinput)}"})           

        #taskcontext = result.get("dialogue_history", None)
        logger.info(f"Player B: Next SubSystem Dialogue History\n{taskinput}")
        self.promptlogs.append({"role": f"DH to {next_subsystem}", 'content': f"Input to {next_subsystem} sub-system:\n{json.dumps(taskcontext)}"})           


        status, use_subsystem = self._validate_subsystem(next_subsystem)
        logger.info(f"Player B: Subsystem Validation: status - {status}, use_subsystem - {use_subsystem}")

        if not status:
            if use_subsystem == "reprobe":
                # Probe the LLM one more time
                # TODO: Do we need to add any message to the LLM to behave itself?
                # self._append_utterance(None, answer, "user")
                logger.error(
                    "No matching sub-system found for the next task. Probing the LLM one more time."
                )
                self._append_utterance(
                    "No matching sub-system found for the next task.", None, "user"
                )                        
                return self.promptlogs, None, "reprobe"
            else:
                errormsg = f"Invalid Subsystem: {next_subsystem}. Cannot continue processing."
                logger.error(errormsg)
                #Game Master should treat this as failure and abort the game
                self.promptlogs.append({"role": "assistant", "content": errormsg})
                return self.promptlogs, None, errormsg

        '''
        usetaskinput = self._validate_subsystem_input(next_subsystem, taskinput)

        if usetaskinput is None:
            errormsg = f"Invalid Subsystem({use_subsystem}) InputData {taskinput}. Cannot continue processing."
            logger.error(errormsg)
            #Game Master should treat this as failure and abort the game
            return self.promptlogs, None, errormsg
        '''
        usetaskinput = {"input_data": taskinput, "dialogue_history": taskcontext}
        prompt, raw_response, raw_answer_ss, ss_answer = subsystem_handlers[use_subsystem](usetaskinput, current_turn)
        self.promptlogs.append({"role": f"{use_subsystem}", 'content': {'prompt': prompt, 'raw_answer': raw_response,
                                                            'answer': f"Sub-system({use_subsystem}) response: {ss_answer}"}})                
        logger.info(f"{use_subsystem} response appending to Player B\n{ss_answer}")
        self._append_utterance(use_subsystem, ss_answer, "user")


        #Validate Sub-System Response
        sub_sys_response = self._validate_subsystem_output(use_subsystem, ss_answer)

        if sub_sys_response is None:
            errormsg = f"Invalid Subsystem({use_subsystem}) Output {ss_answer}. Cannot continue processing."
            logger.error(errormsg)
            #Game Master should treat this as failure and abort the game
            return self.promptlogs, None, errormsg

        #Adding sleep to reduce the frequencey of calls to the LLM
        time.sleep(0.5)
        return None, None, sub_sys_response


    def run(self, utterance, current_turn: int) -> str:
        """
        The following actions will be done in a loop until the DM module is ready to respond to user request
        1. Feed the user input to the LLM DM
        2. Get the next action from the LLM DM
        3. Call the relevant module with the action
        4. If there is no matching module, probe the LLM DM one more time (total: 2 times)
        5. Go to step 2 and repeat the above steps until the DM module is ready to respond to the user request or the number of probes reaches 5
        """

        subsystem_handlers = {
                    "intent_detector": self.intentdet.run,
                    "slot_extractor": self.slotext.run,
                    "response_generator": self.followupgen.run,
                    "dbquery_formatter": self.dbqueryformatter.run,
                    "booking_formatter": self.bookingformatter.run
                }

        self.promptlogs = []
        self.cur_reprobe = 0
        self.promptlogs.append({"role": "user", "content": f"User Query: {utterance}"})
        #self._append_utterance(None, utterance, "user")
        num_process_ssystem = 0

        while True:
            if num_process_ssystem > 10:
                errormsg = f"Too many times ({num_process_ssystem}) sub-systems processed. Cannot continue processing."
                logger.error(errormsg)
                #Game Master should treat this as failure and abort the game
                self.promptlogs.append({"role": "assistant", "content": f"{errormsg}"})
                return self.promptlogs, None, errormsg

            if self.ishfmodel():
                tool_content = {"name": self.func_name, "content": utterance}
                self._append_utterance(None, tool_content, "tool")

            else:
                #self._append_utterance(None, utterance, "user")
                tool_content = {"content": utterance, 'tool_call_id': self.tool_call_id}
                self._append_utterance(None, tool_content, "tool")
                self.tool_call_id = None                

            if self.tool_calls_list:
                answer_cleanup = self.tool_calls_list[0]
                self.tool_calls_list = self.tool_calls_list[1:]

                self.promptlogs.append({"role": "assistant", "content": f"model response before processing: {answer_cleanup}"})

                error_logs, parse_status, data = self._parse_next_subsystem(answer_cleanup)
                if error_logs:
                    return self.promptlogs, None, data

                num_process_ssystem += 1
                logger.info(f"Player B: Next SubSystem: {data}")

                next_subsystem = data.get("next_ss", None)
                taskinput = data.get("taskinput", None)
                taskcontext = data.get("taskcontext", None)

                if next_subsystem:
                    error_logs, process_status, sub_sys_response = self._process_next_sub_system(next_subsystem, taskinput, taskcontext, current_turn, subsystem_handlers)
                    if error_logs:
                        if sub_sys_response == "reprobe":
                            continue
                        else:
                            return self.promptlogs, None, sub_sys_response
                else:
                    # Return the LLM response to user
                    use_data, error = preparemodelresponse(self.func_name, self.func_arguments)
                    if error:
                        logger.error(f"Failure in model response processing:\n{error}")
                        return self.promptlogs, None, error

                    logger.info(f"Returning the LLM response to the user\n{use_data}")
                    llm_response, error, ret_func_data = self.processresp.run(use_data, "modular_llm")
                    if error:
                        self.promptlogs.append({"role": "assistant", "content": f"error while parsing the data: {error}"})

                    self.promptlogs.append({'role': "modllm", 'content': {'prompt': "ToolCall", 'raw_answer': answer_cleanup,
                                                                        'answer': llm_response}})

                    return self.promptlogs, answer_cleanup, llm_response   


            prompt, raw_answer, answer = self.player_b(
                self.player_b.history, current_turn, self.tool_schema, None)
            logger.info(f"Player B: Subsystem Flow response\n{answer}")

            if not self.ishfmodel():
                self._append_utterance(None, raw_answer['tool_calls'], "assistant")

            self.promptlogs.append({"role": "assistant", "content": f"model response before processing: {answer}"})

            error_logs, parse_status, result = self._parse_model_response(answer)
            if error_logs:
                return self.promptlogs, None, result


            self.promptlogs.append({'role': "modllm", 'content': {'prompt': prompt, 'raw_answer': raw_answer,
                                                                    'answer': result}})
            #self._append_utterance(None, result, "assistant")

            error_logs, parse_status, data = self._parse_next_subsystem(result)
            if error_logs:
                return self.promptlogs, None, data

            num_process_ssystem += 1
            logger.info(f"Player B: Next SubSystem: {data}")

            next_subsystem = data.get("next_ss", None)
            taskinput = data.get("taskinput", None)
            taskcontext = data.get("taskcontext", None)

            if next_subsystem:
                error_logs, process_status, sub_sys_response = self._process_next_sub_system(next_subsystem, taskinput,
                                                                                             taskcontext, current_turn, subsystem_handlers)
                if error_logs:
                    if sub_sys_response == "reprobe":
                        continue
                    else:
                        return self.promptlogs, None, sub_sys_response
            else:
                # Return the LLM response to user
                use_data, error = preparemodelresponse(self.func_name, self.func_arguments)
                if error:
                    logger.error(f"Failure in model response processing:\n{error}")
                    return self.promptlogs, None, error

                logger.info(f"Returning the LLM response to the user\n{use_data}")
                llm_response, error, ret_func_data = self.processresp.run(use_data, "modular_llm")
                if error:
                    self.promptlogs.append({"role": "assistant", "content": f"error while parsing the data: {error}"})

                self.promptlogs.append({'role': "modllm", 'content': {'prompt': prompt, 'raw_answer': raw_answer,
                                                                    'answer': llm_response}})

                return self.promptlogs, raw_answer, llm_response   
