import copy
from typing import Dict, Any, List, Tuple
import json

#import clemgame
#from clemgame import get_logger
from dialogue_systems.modllmdsys.players import ModLLMSpeaker
from utils import cleanupanswer

import logging

logger = logging.getLogger(__name__)


class SlotExtractor:
    def __init__(self, model_name, model_spec, slot_ext_prompt, json_format):
        self.model_name = model_name
        self.model_spec = model_spec
        self.base_prompt = slot_ext_prompt
        self.base_json_schema = json_format
        self.tool_schema = None
        self.player = None
        self._setup()

    def _prepare_response_tool_schema(self):

        self.tool_schema = [
            {
                "type": "function",
                "function": {
                    "name": "extractrestuarantdata",
                    "description": "Extracts relevant entities from user input during a restaurant search/reservation. This includes optional fields such as area, food (cuisine), pricerange, restaurant name, number of people, reservation day, and reservation time. Use this function to interpret and structure user-provided search/booking details before search or validation.",
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
                            },
                            "people": {
                                "type": "string",
                                "enum": ["1", "2", "3", "4", "5", "6", "7", "8"],
                                "description": "Number of people for the restaurant reservation. Optional."
                            },
                            "day": {
                                "type": "string",
                                "enum": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"],
                                "description": "Day of the restaurant reservation. Optional."
                            },
                            "time": {
                                "type": "string",
                                "pattern": "^(0[0-9]|1[0-9]|2[0-3]):[0-5][0-9]$",
                                "description": "Time of the restaurant reservation, formatted as HH:MM (24-hour format). Optional."
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
                        "required": [],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "extracthoteldata",
                    "description": "Extracts relevant entities from user input during a hotel search/reservation. This includes optional fields such as area, pricerange, type, hotel name, internet, parking, stars, number of people, reservation day, and number of days of stay. Use this function to interpret and structure user-provided search/booking details before search or validation.",
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
                            },
                            "people": {
                                "type": "string",
                                "enum": ["1", "2", "3", "4", "5", "6", "7", "8"],
                                "description": "Number of people for the hotel booking. Optional."
                            },
                            "day": {
                                "type": "string",
                                "enum": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"],
                                "description": "Day of the hotel booking. Optional."
                            },
                            "stay": {
                                "type": "string",
                                "enum": ["1", "2", "3", "4", "5", "6", "7", "8"],
                                "description": "Length of stay at the hotel. Optional."
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
                        "required": [],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "extracttraindata",
                    "description": "Extracts relevant entities from user input during a train search/reservation. This includes optional fields such as destination, departure, day, arriveby, leaveat, number of people, and trainid. Use this function to interpret and structure user-provided search/booking details before search or validation.",
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
                            },
                            "people": {
                                "type": "string",
                                "enum": ["1", "2", "3", "4", "5", "6", "7", "8"],
                                "description": "Number of train tickets for the booking. Optional."
                            },
                            "trainid": {
                                "type": "string",
                                "description": "ID of the train. Optional."
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
                        "required": [],
                        "additionalProperties": False
                    }
                }
            }                               
        ]





    def _setup(self) -> None:
        # If the model_name is of type "LLM", then we need to instantiate the player
        # TODO: Implement for other model types ("Custom Model", "RASA", etc.)
        self.player = ModLLMSpeaker(self.model_spec, "slot_extraction", "", {})
        #self.player.history.append({"role": "user", "content": self.base_prompt})
        self._prepare_response_tool_schema()

    def run(self, utterance: Dict, turn_idx: int) -> str:
        self.player.history = [{"role": "user", "content": self.base_prompt}]
        message = json.dumps(utterance) if isinstance(utterance, Dict) else utterance
        self.player.history[-1]["content"] += message

        prompt, raw_response, raw_answer = self.player(self.player.history, turn_idx, self.tool_schema, None)
        logger.info(f"Slot extractor raw response:\n{raw_answer}")
        answer_cleaned =  cleanupanswer(raw_answer)
        parsed_answer = self._parse_response(answer_cleaned)

        return prompt, raw_response, raw_answer, parsed_answer

    def _parse_response(self, answer):
        logger.info(f"Answer from the model: {answer}, {type(answer)}")        
        if isinstance(answer, list):
            #return f"Invalid response type. {type(answer)}. Expected dict"
            answer = answer[0]

        if not isinstance(answer, dict):
            return f"Invalid response type. {type(answer)}. Expected dict"

        if "name" not in answer or answer["name"] not in ["extractrestaurantdata", "extracthoteldata", "extracttraindata"]:
            return f"function name ( extractrestaurantdata/extracthoteldata/extracttraindata ) is missing in the response {answer}. Cannot proceed."

        use_args_key = None
        if "arguments" in answer:
            use_args_key = "arguments"

        elif "parameters" in answer:
            use_args_key = "parameters"

        else:
            return f"function arguments is missing in the response {answer}. Cannot proceed."

        result = answer[use_args_key]
        return json.dumps(result)


    def get_history(self):
        return self.player.history
    
    def clear_history(self):
        self.player.history = [{"role": "user", "content": self.base_prompt}]     
