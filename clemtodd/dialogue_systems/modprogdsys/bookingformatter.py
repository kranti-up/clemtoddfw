import copy
from typing import Dict, Any, List, Tuple
import json

#import clemgame
#from clemcore import get_logger
from dialogue_systems.modprogdsys.players import ModProgLLMSpeaker
from utils import cleanupanswer

import logging

logger = logging.getLogger(__name__)

class BookingFormatter:
    def __init__(self, model_name, model_spec, book_aggr_prompt, json_format):
        self.model_name = model_name
        self.model_spec = model_spec
        self.base_prompt = book_aggr_prompt
        self.player = None
        self.base_json_schema = json_format
        self.json_schema_prompt = None        
        self._setup()

    def _prepare_response_json_schema(self):

        self.json_schema_prompt = {
                                    "type": "function",
                                    "function": {
                                        "name": "prepare_bookinhquery",
                                        "description": "Extract structured details for restaurant, hotel, or train-related queries.",
                                        "parameters": {
                                            "type": "object",
                                            "properties": {
                                                "booking_query": {
                                                    "type": "object",
                                                    "description": "A dictionary containing key-value pairs extracted from the input text",
                                                    "oneOf": [
                                                        {
                                                            "properties": {
                                                                "domain": { "const": "restaurant" },
                                                                "restaurant": {
                                                                    "type": "object",
                                                                    "anyOf": [
                                                                        {
                                                                            "properties": {
                                                                                "food": { "type": "string" }
                                                                            },
                                                                            "required": ["food"]
                                                                        },     
                                                                        {
                                                                            "properties": {
                                                                                "area": { "type": "string", "enum": ["centre", "north", "east", "west", "south"]  }
                                                                            },
                                                                            "required": ["area"]
                                                                        },                                                                                     
                                                                        {
                                                                            "properties": {
                                                                                "pricerange": { "type": "string",
                                                                                        "enum": ["cheap", "moderate", "expensive"] }
                                                                            },
                                                                            "required": ["pricerange"]
                                                                        },  
                                                                        {
                                                                            "properties": {
                                                                                "name": { "type": "string"}
                                                                            },
                                                                            "required": ["name"]
                                                                        },
                                                                        {
                                                                            "properties": {
                                                                                "people": { "type": "string", "enum": ["1", "2", "3", "4", "5", "6", "7", "8"]},                                                                                
                                                                            },
                                                                            "required": ["people"]
                                                                        },
                                                                        {
                                                                            "properties": {
                                                                                "day": { "type": "string", "enum": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]},
                                                                            },
                                                                            "required": ["day"]
                                                                        },
                                                                        {
                                                                            "properties": {
                                                                                "time": { "type": "string", "pattern": "^(0[0-9]|1[0-9]|2[0-3]):[0-5][0-9]$" },
                                                                            },
                                                                            "required": ["time"]
                                                                        },
                                                                        {
                                                                            "properties": {
                                                                                "phone": { "type": "string"}
                                                                            },
                                                                            "required": ["phone"]
                                                                        },
                                                                        {
                                                                            "properties": {
                                                                                "postcode": { "type": "string"}
                                                                            },
                                                                            "required": ["postcode"]
                                                                        },
                                                                        {
                                                                            "properties": {
                                                                                "address": { "type": "string"}
                                                                            },
                                                                            "required": ["address"]
                                                                        }                                                                                                                                                                                                                              
                                                                    ]
                                                                }                                            
                                                            },
                                                            "required": ["domain", "restaurant"],
                                                            "additionalProperties": False
                                                        },
                                                        {
                                                            "properties": {
                                                                "domain": { "const": "hotel" },
                                                                "hotel": {
                                                                    "type": "object",
                                                                    "anyOf": [
                                                                        {
                                                                            "properties": {
                                                                                "area": { "type": "string", "enum": ["centre", "north", "east", "west", "south"] }
                                                                            },
                                                                            "required": ["area"]
                                                                        },     
                                                                        {
                                                                            "properties": {
                                                                                "pricerange": { "type": "string", "enum": ["cheap", "moderate", "expensive"] }
                                                                            },
                                                                            "required": ["pricerange"]
                                                                        },    
                                                                        {
                                                                            "properties": {
                                                                                "type": { "type": "string", "enum": ["hotel", "guesthouse"] }
                                                                            },
                                                                            "required": ["type"]
                                                                        },
                                                                        {
                                                                            "properties": {
                                                                                "internet": { "type": "string", "enum": ["yes", "no"] }
                                                                            },
                                                                            "required": ["internet"]
                                                                        }, 
                                                                        {
                                                                            "properties": {
                                                                                "parking": { "type": "string", "enum": ["yes", "no"] }
                                                                            },
                                                                            "required": ["parking"]
                                                                        },
                                                                        {
                                                                            "properties": {
                                                                                "name": { "type": "string"}
                                                                            },
                                                                            "required": ["name"]
                                                                        }, 
                                                                        {
                                                                            "properties": {
                                                                                "stars": {
                                                                                    "type": "object",
                                                                                    "properties": {
                                                                                        "operator": { "type": "string", "enum": ["=", ">=", "<=", ">", "<"] },
                                                                                        "value": { "type": "string", "enum": ["1", "2", "3", "4", "5"] }
                                                                                    },
                                                                                    "required": ["operator", "value"],
                                                                                    "additionalProperties": False
                                                                                }
                                                                            },
                                                                            "required": ["stars"]
                                                                        },
                                                                        {
                                                                            "properties": {
                                                                                "people": { "type": "string", "enum": ["1", "2", "3", "4", "5", "6", "7", "8"]},                                                                                
                                                                            },
                                                                            "required": ["people"]
                                                                        },
                                                                        {
                                                                            "properties": {
                                                                                "day": { "type": "string", "enum": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]},
                                                                            },
                                                                            "required": ["day"]
                                                                        },
                                                                        {
                                                                            "properties": {
                                                                                "stay": { "type": "string", "enum": ["1", "2", "3", "4", "5", "6", "7", "8"]},                                                                                
                                                                            },
                                                                            "required": ["stay"]
                                                                        },
                                                                        {
                                                                            "properties": {
                                                                                "phone": { "type": "string"}
                                                                            },
                                                                            "required": ["phone"]
                                                                        },
                                                                        {
                                                                            "properties": {
                                                                                "postcode": { "type": "string"}
                                                                            },
                                                                            "required": ["postcode"]
                                                                        },
                                                                        {
                                                                            "properties": {
                                                                                "address": { "type": "string"}
                                                                            },
                                                                            "required": ["address"]
                                                                        }                                                                                                                                                         
                                                                    ]
                                                                }
                                                            },
                                                            "required": ["domain", "hotel"],
                                                            "additionalProperties": False
                                                        },
                                                        {
                                                            "properties": {
                                                                "domain": { "const": "train" },
                                                                "train": {
                                                                    "type": "object",
                                                                    "anyOf": [
                                                                        {
                                                                            "properties": {
                                                                                "destination": { "type": "string" }
                                                                            },
                                                                            "required": ["destination"]
                                                                        },     
                                                                        {
                                                                            "properties": {
                                                                                "departure": { "type": "string" }
                                                                            },
                                                                            "required": ["departure"]
                                                                        },                                                                                     
                                                                        {
                                                                            "properties": {
                                                                                "day": { "type": "string",
                                                                                        "enum": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"] }
                                                                            },
                                                                            "required": ["day"]
                                                                        },  
                                                                        {
                                                                            "properties": {
                                                                                "arriveby": {
                                                                                    "type": "object",
                                                                                    "properties": {
                                                                                        "operator": { "type": "string", "enum": ["=", ">=", "<=", ">", "<"],
                                                                                                    "description": "A comparison operator indicating the condition (e.g., '<=' means arriving by or before a time)."
                                                                                        },
                                                                                        "value": { "type": "string", "pattern": "^(0[0-9]|1[0-9]|2[0-3]):[0-5][0-9]$",
                                                                                                "description": "A time string formatted as HH:MM (24-hour format)."
                                                                                        }
                                                                                    },
                                                                                    "required": ["operator", "value"],
                                                                                    "additionalProperties": False
                                                                                }
                                                                            },
                                                                            "required": ["arriveby"]
                                                                        },   
                                                                        {
                                                                            "properties": {
                                                                                "leaveat": {
                                                                                    "type": "object",
                                                                                    "properties": {
                                                                                        "operator": { "type": "string", "enum": ["=", ">=", "<=", ">", "<"],
                                                                                                    "description": "A comparison operator indicating the condition (e.g., '<=' means departing by or before a time)."
                                                                                        },
                                                                                        "value": { "type": "string", "pattern": "^(0[0-9]|1[0-9]|2[0-3]):[0-5][0-9]$",
                                                                                                "description": "A time string formatted as HH:MM (24-hour format)."
                                                                                        }
                                                                                    },
                                                                                    "required": ["operator", "value"],
                                                                                    "additionalProperties": False
                                                                                }
                                                                            },
                                                                            "required": ["leaveat"]
                                                                        },
                                                                        {
                                                                            "properties": {
                                                                                "people": { "type": "string", "enum": ["1", "2", "3", "4", "5", "6", "7", "8"]},                                                                                
                                                                            },
                                                                            "required": ["people"]
                                                                        },
                                                                        {
                                                                            "properties": {
                                                                                "trainid": { "type": "string"}
                                                                            },
                                                                            "required": ["trainid"]
                                                                        }                                                                                                                                                            
                                                                    ]
                                                                }
                                                            },
                                                            "required": ["domain", "train"],
                                                            "additionalProperties": False
                                                        }
                                                    ]                                                           
                                                }
                                            },
                                            "required": ["booking_query"],
                                            "additionalProperties": False
                                        }
                                    }
                                }  


    def _setup(self) -> None:
        # If the model_name is of type "LLM", then we need to instantiate the player
        # TODO: Implement for other model types ("Custom Model", "RASA", etc.)
        self.player = ModProgLLMSpeaker(self.model_spec, "booking_formatter", "", {})
        #self.player.history.append({"role": "user", "content": self.base_prompt})
        self._prepare_response_json_schema()

    def run(self, utterance: Dict, turn_idx: int) -> str:
        self.player.history = [{"role": "user", "content": self.base_prompt}]     
        message = json.dumps(utterance) if isinstance(utterance, Dict) else utterance
        self.player.history[-1]["content"] += message

        prompt, raw_response, raw_answer = self.player(self.player.history, turn_idx, None, self.json_schema_prompt)
        logger.info(f"DBQuery Formatter raw response:\n{raw_answer}")
        answer_cleaned =  cleanupanswer(raw_answer)
        return prompt, raw_response, raw_answer, answer_cleaned
    
    def get_history(self):
        return self.player.history
    
    def clear_history(self):
        self.player.history = [{"role": "user", "content": self.base_prompt}]    
