from collections.abc import Mapping
from typing import Dict
import time
import json


#from clemgame import get_logger
from utils import cleanupanswer
from dialogue_systems.modprogdsys.intentdetector import IntentDetector
from dialogue_systems.modprogdsys.slotextractor import SlotExtractor
from dialogue_systems.modprogdsys.followupgenerator import FollowupGenerator
from dialogue_systems.modprogdsys.dbqueryformatter import DBQueryFormatter
from dialogue_systems.modprogdsys.bookingformatter import BookingFormatter
from processfunccallresp import ProcessFuncCallResp

import logging

logger = logging.getLogger(__name__)

class ModProgLLM:
    def __init__(self, model_name, model_spec, prompts_dict, player_dict, resp_json_schema, liberal_processing, booking_mandatory_keys):
        self.model_name = model_name
        self.model_spec = model_spec
        self.prompts_dict = prompts_dict
        #self.player_b = player_dict["monollm_player"]
        self.resp_json_schema = resp_json_schema
        self.liberal_processing = liberal_processing
        self.booking_data = {}
        self.current_state = None
        self.slotdata = {}
        self.dstate = None
        self.dhistory = []
        self.promptlogs = []
        self.processresp = ProcessFuncCallResp()

        self.respformat = resp_json_schema#["schema"]
        self.booking_mandatory_keys = booking_mandatory_keys
        self._create_subsystems(model_name, model_spec, prompts_dict)

        #self.player_b.history.append({"role": "user", "content": prompts_dict["prompt_b"]})

        #self.turn_ss_prompt_player_b = prompts_dict["turn_ss_prompt_b"]
        self.liberalcount = {"intent": 0, "slot": 0, "follow": 0, "aggregator": 0}
        logger.info(f"ProgSubSystems __init__ done")



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


    def _append_utterance(self, subsystem: str, utterance: str, role: str) -> None:
        """Add an utterance to the history of a player (firstlast specific)."""

        if isinstance(utterance, dict) or isinstance(utterance, list):
            utterance = json.dumps(utterance)

        add_data = utterance

        if role == "user":
            add_data = subsystem
            if utterance:
                turn_prompt = self.turn_ss_prompt_player_b.replace(
                    "$sub-system", subsystem
                )
                add_data = turn_prompt + "\n\n" + utterance

        #self.player_b.history.append({"role": role, "content": add_data.strip()})



    def _prepare_subsystem_input(self, taskinput: Dict, next_subsystem: str) -> Dict:
        return json.dumps({"next_subsystem": next_subsystem, "input_data": taskinput})

    def _validate_subsystem_input(self, sub_system: str, taskinput: Dict) -> Dict:
        logger.info(f"Validating Subsystem Input: {sub_system}-> {taskinput} {type(taskinput)}")
        if taskinput is None or isinstance(taskinput, str) or isinstance(taskinput, json.decoder.JSONDecodeError):
            if taskinput is None:
                errormsg = f"{sub_system} subsystem output is None. Cannot continue processing"
            elif isinstance(taskinput, str):
                errormsg = f"{sub_system} subsystem output is a string and not matching with the expected type: Dict. Cannot continue processing"
            elif isinstance(taskinput, json.decoder.JSONDecodeError):
                errormsg = f"Failure in JSON parsing of: {sub_system} subsystem output. Error ({taskinput}). Cannot continue processing"
            return None, errormsg
        # elif all(isinstance(value, dict) for value in taskinput.values()):
        #    return {}
        else:
            errormessage = "No key named $subsystem is not available in the response. Cannot continue processing"
            if sub_system == "intent_detector":
                if "intent_detection" in taskinput and "domain" in taskinput:
                    return taskinput, None
                else:
                    return None, errormessage.replace("$subsystem", "intent_detection")
            elif sub_system == "slot_extractor":
                if "slot_extraction" in taskinput:
                    if "domain" in taskinput["slot_extraction"]:
                        if taskinput["slot_extraction"]["domain"] in taskinput["slot_extraction"]:
                            if taskinput["slot_extraction"]["domain"]:
                                return taskinput, None
                            else:
                                return None, errormessage.replace("$subsystem", "empty domain data inside slot_extraction")
                        else:
                            return None, errormessage.replace("$subsystem", f"{taskinput['slot_extraction']['domain']} inside slot_extraction")
                    else:
                        return None, errormessage.replace("$subsystem", "domain inside slot_extraction")
                else:
                    return None, errormessage.replace("$subsystem", "slot_extraction")
            elif sub_system == "followup_generator":
                if "followup_generation" in taskinput:
                    return taskinput, None
                else:
                    return None, errormessage.replace("$subsystem", "followup_generation")
            elif sub_system == "dbquery_formatter":
                if "dbquery_format" in taskinput:
                    return taskinput, None
                else:
                    return None, errormessage.replace("$subsystem", "dbquery_format")
            elif sub_system == "booking_formatter":
                if "booking_query" in taskinput:
                    return taskinput, None
                else:
                    return None, errormessage.replace("$subsystem", "booking_query")
            return taskinput, None       

    def _getquery_type(self, utterance):
        if "USER REQUEST:" in utterance:
            split_query = "USER REQUEST:"
            query_type = "user-request"
        elif "DATABASE RETRIEVAL RESULTS:" in utterance:
            split_query = "DATABASE RETRIEVAL RESULTS:"
            query_type = "db-retrieval"
        elif "BOOKING VALIDATION STATUS:" in utterance:
            split_query = "BOOKING VALIDATION STATUS:"
            query_type = "booking-validation"

        user_request = (
            utterance.split(split_query)[1].strip()
        )

        return query_type, user_request
    
    def _call_subsystem(self, sub_system, taskinput: Dict, current_turn: int):
        subsystem_handlers = {
            "intent_detector": self.intentdet,
            "slot_extractor": self.slotext,
            "followup_generator": self.followupgen,
            "dbquery_formatter": self.dbqueryformatter,
            "booking_formatter": self.bookingformatter,
        }
        logger.info(
            f"Calling Subsystem: {sub_system} with taskinput: {taskinput}, Current Turn: {current_turn}"
        )
        ss_data = self._prepare_subsystem_input(taskinput, sub_system)
        self.promptlogs.append({"role": f"Input to {sub_system}", 'content': ss_data})
        #self._append_utterance(None, ss_data, "assistant")
        prompt, raw_response, raw_answer, ss_answer = subsystem_handlers[sub_system].run(taskinput, current_turn)
        
        self.promptlogs.append({"role": "assistant", "content": f"subsystem response before processing: {raw_answer}"})
        self.promptlogs.append({"role": f"{sub_system}", 'content': {'prompt': prompt, 'raw_answer': raw_response,
                                                                    'answer': ss_answer}})
        # subsystem_handlers[sub_system].clear_history()
        logger.info(f"Subsystem Answer: {ss_answer}, {type(ss_answer)}")
        #self._append_utterance(sub_system, ss_answer, "user")
        usetaskinput, errormsg = self._validate_subsystem_input(sub_system, ss_answer)

        if usetaskinput is None:
            logger.error(
                f"Invalid Subsystem InputData {ss_answer}. Cannot continue processing."
            )
            # Game Master should treat this as failure and abort the game
            # TODO: Having None for prompt, raw_answer and answer is not a good idea. Need to handle this properly
            return None, errormsg
        # Adding sleep to reduce the frequencey of calls to the LLM
        time.sleep(0.5)
        return usetaskinput, None

    def _prepare_gm_response(self, status, details):
        '''
        dmanswer = {
            "status": status,
            "details": details,
        }
        '''
        # details contains both status and details in a string format
        raw_answer = {
            "model": self.model_name,
            "choices": [{"message": {"role": "user", "content": details}}],
        }

        logger.info(
            f"Returning to GM : {details} {self.model_name} {type(self.model_name)}"
        )
        #return self.promptlogs, raw_answer, json.dumps(dmanswer)
        return self.promptlogs, raw_answer, details

    def _updateslots(self, curslots, dbformatslots):
        for slot in curslots:
            if slot in dbformatslots:
                curslots[slot] = dbformatslots[slot]

    def _add_descriptions(self, missing_keys, domain):
        descriptions = {
            "restaurant": {
                "area": "The area/location/place of the restaurant.",
                "pricerange": "The price budget for the restaurant.",
                "food": "The cuisine of the restaurant you are looking for.",
                "name": "The name of the restaurant.",
                "people": "Number of people for the restaurant reservation.",
                "day": "Day of the restaurant reservation.",
                "time": "Time of the restaurant reservation.",
                "phone": "Phone number of the restaurant.",
                "postcode": "Postal code of the restaurant.",
                "address": "Address of the restaurant.",
            },
            "hotel": {
                "area": "The area/location/place of the hotel.",
                "pricerange": "The price budget for the hotel.",
                "type": "What is the type (hotel/guesthouse) of the hotel.",
                "name": "The name of the hotel.",
                "internet": "Indicates whether the hotel has internet/wifi or not.",
                "parking": "Indicates whether the hotel has parking or not.",
                "stars": "The star rating of the hotel.",
                "people": "Number of people for the hotel booking.",
                "day": "Day of the hotel booking.",
                "stay": "Length of stay at the hotel.",
                "phone": "Phone number of the hotel.",
                "postcode": "Postal code of the hotel.",
                "address": "Address of the hotel.",
            },
            "train": {
                "destination": "Destination of the train.",
                "departure": "Departure location of the train.",
                "day": "Journey day of the train.",
                "arriveby": "Arrival time of the train.",
                "leaveat": "Leaving time for the train.",
                "people": "Number of train tickets for the booking.",
                "trainid": "ID of the train.",
            },
        }

        if domain not in descriptions:
            return missing_keys

        return [{key: descriptions[domain].get(key, f"Missing field for {domain} booking")} for key in missing_keys]

    def _isbookingdatapresent(self, slotdata):
        logger.info(f"Booking Keys: {self.booking_mandatory_keys}, SlotData: {slotdata}")

        missing_keys = []

        if "domain" in slotdata and slotdata["domain"] in slotdata and slotdata[slotdata["domain"]] and slotdata["domain"] in self.booking_mandatory_keys:
            missing_keys = [bkey for bkey in self.booking_mandatory_keys[slotdata["domain"]] if bkey not in slotdata[slotdata["domain"]]]

            #if all(bkey in slotdata[slotdata["domain"]] for bkey in self.booking_mandatory_keys[slotdata["domain"]]):
            if not missing_keys:
                logger.info(f"All mandatory keys are present in the booking data. Can proceed with the booking")
                return True, None
            else:
                missing_keys = self._add_descriptions(missing_keys, slotdata["domain"])

        logger.info(f"Missing keys in the booking data: {missing_keys}")
        return False, missing_keys

    def _call_followup_for_missing_booking_data(self, user_request, taskinput, errormsg_base, current_turn, current_state):

        if current_state:
            self.current_state = current_state

        '''
        taskinput = {"user request": user_request,
                     "extracted data": self.slotdata,}
        '''
        if self.dhistory:
            taskinput["dialog_history"] = self.dhistory

        followup_answer, follow_error = self._call_subsystem(
            "followup_generator", taskinput, current_turn
        )
        logger.info(f"followup_answer = {followup_answer}")
        if followup_answer is None:
            #errormsg = errormsg_base.replace("$subsystem", "followup_generator")
            self.promptlogs.append({"role": "assistant", "content": follow_error})
            return self.promptlogs, None, follow_error

        self.dhistory.append(
            {
                #"role": "assistant",
                "intent": "follow-up",
                "response": followup_answer["followup_generation"],
            }
        )
        # self.dhistory[-1].update({"assistant": followup_answer["followup_generation"]})
        ss_output = {"status": "follow-up", "details": followup_answer["followup_generation"]}
        llm_response, error, _ = self.processresp.run(ss_output, "modular_prog")
        if error:
            self.promptlogs.append({"role": "assistant", "content": str(error)})
            return self.promptlogs, None, error
        return self._prepare_gm_response("follow-up", llm_response)
    
    def _handle_booking(self, booking_slots):
        self.current_state = "validate-booking" 
        self.dhistory.append(
            { #"role": "assistant",
              "intent": "validate-booking",
              "response": booking_slots}
        )                       
        ss_output = {"status": "validate-booking", "details": booking_slots}
        llm_response, error, _ = self.processresp.run(ss_output, "modular_prog")                
        # self.dhistory[-1].update({"assistant": "validate-booking"})
        if error:
            self.promptlogs.append({"role": "assistant", "content": str(error)})
            return self.promptlogs, None, error
        return self._prepare_gm_response("validate-booking", llm_response)
    
    def _call_booking_formatter(self, taskinput, current_turn):
        '''
        bookingformatter_answer = self._call_subsystem(
            "booking_formatter", taskinput, current_turn
        )
        logger.info(
            f"After Booking Formatter: bookingformatter_answer = {bookingformatter_answer}"
        )
        if bookingformatter_answer is None:
            errormsg = "Failure in the booking formatting. Cannot continue processing."
            self.promptlogs.append({"role": "assistant", "content": errormsg})
            return self.promptlogs, None, errormsg
        
        return self._handle_booking(bookingformatter_answer["booking_query"])
        '''
        return self._handle_booking(taskinput["extracted data"])

    def deep_merge_slots(self, d1, d2):
        """Recursively merges d2 into d1 without overwriting existing keys."""
        if not d2 or not isinstance(d2, dict):
            return

        for key, value in d2.items():
            if key in d1 and isinstance(d1[key], dict) and isinstance(value, Mapping):
                self.deep_merge_slots(d1[key], value)  # Recursive merge
            else:
                d1[key] = value  # Overwrite/insert key



    def run(self, utterance, current_turn):
        """
        The following actions will be done in a loop until the DM module is ready to respond to user request
        1. Feed the user input to the LLM DM
        2. Get the next action from the LLM DM
        3. Call the relevant module with the action
        4. If there is no matching module, probe the LLM DM one more time (total: 2 times)
        5. Go to step 2 and repeat the above steps until the DM module is ready to respond to the user request or the number of probes reaches 5
        """
        self.promptlogs = []
        taskinput = {}

        query_type, user_request = self._getquery_type(utterance)
        logger.info(
            f"Query Type: {query_type}, User Request: {user_request} Current Turn: {current_turn}"
        )
        if current_turn == 1:
            if self.dstate is None:
                self.dstate = user_request
                self.current_state = "user-request"
                # self.dhistory.append({"role": "user", "content": user_request})

        self.promptlogs.append({"role": "user", "content": user_request})

        errormsg_base = "$subsystem subsystem output not matching with the expected response format. Cannot continue processing"
        while True:
            #if query_type == "user-request":
            taskinput = {"input": user_request}
            #elif query_type == "db-retrieval":
            #    taskinput = {"db-query-response": user_request}
            #elif query_type == "booking-validation":
            #    taskinput = {"booking-status": user_request}

            if self.dhistory:
                taskinput["dialog_history"] = self.dhistory

            #Commenting for testing multi-domain dialogues
            '''
            if self.current_state == "booking-success":
                taskinput = {
                    "booking-status": "Success"
                }
                return self._call_followup_for_missing_booking_data(user_request, taskinput, errormsg_base, current_turn, None)    
            '''

            intent_answer, intent_error = self._call_subsystem(
                "intent_detector", taskinput, current_turn
            )
            logger.info(f"Intent Answer: {intent_answer} {type(intent_answer)}")
            if intent_answer is None:
                #errormsg = errormsg_base.replace("$subsystem", "intent_detector")
                self.promptlogs.append({"role": "assistant", "content": intent_error})
                return self.promptlogs, None, intent_error

            '''
            self.dhistory.append(
                {
                    #"role": "user",
                    "input": user_request,
                    "intent": intent_answer["intent_detection"],
                    "domain": intent_answer["domain"],
                }
            )
            '''

            if query_type == "user-request":
                slot_answer, slot_error = self._call_subsystem(
                    "slot_extractor", taskinput, current_turn
                )
                logger.info(f"Slot Answer: {slot_answer}, Current Slot Data: {self.slotdata}")
                if slot_answer is None:
                    #errormsg = errormsg_base.replace("$subsystem", "slot_extractor")
                    self.promptlogs.append({"role": "assistant", "content": slot_error})
                    return self.promptlogs, None, slot_error

                if slot_answer["slot_extraction"]:
                    #self.slotdata.update(slot_answer["slot_extraction"])
                    self.deep_merge_slots(self.slotdata, slot_answer["slot_extraction"])

            intent_detection = intent_answer["intent_detection"]
            if intent_detection == "booking-request":
                # self.booking_data.update(slot_answer["slot_extraction"])
                self.dhistory.append(
                    {
                        #"role": "user",
                        "intent": intent_detection,
                        "input": user_request,
                    }
                )

                ext_slots = self.slotdata
                logger.info(f"ext_slots: {ext_slots}")

                book_status, book_keys_miss = self._isbookingdatapresent(ext_slots)

                if book_status:
                    taskinput = {"extracted data": ext_slots}
                    return self._call_booking_formatter(taskinput, current_turn)
                else:
                    taskinput = {
                        "input": user_request,
                        "extracted data": self.slotdata,
                        "missing data for booking": book_keys_miss,
                    }
                    return self._call_followup_for_missing_booking_data(user_request, taskinput, errormsg_base,
                                                                        current_turn, "validate-booking")


            elif intent_detection in ["booking-success", "booking-failure"]:
                self.dhistory.append(
                    {
                        #"role": "user",
                        "intent": intent_detection,
                        "response": user_request,
                    }
                )

                taskinput = {"booking_confirmation_status": user_request}
                return self._call_followup_for_missing_booking_data(user_request, taskinput,
                                                                    errormsg_base, current_turn, intent_detection)


            elif intent_detection == "dbretrieval-request":
                self.dhistory.append(
                    {
                        #"role": "user",
                        "input": user_request,
                        "intent": intent_answer["intent_detection"],
                        "domain": intent_answer["domain"],
                        "extracted slots": self.slotdata,
                    }
                )

                taskinput = {"extracted data": self.slotdata}
                '''
                dbqueryformatter_answer = self._call_subsystem(
                    "dbquery_formatter", taskinput, current_turn
                )
                logger.info(
                    f"After DB Formatter: dbqueryformatter_answer = {dbqueryformatter_answer}"
                )
                if dbqueryformatter_answer is None:
                    errormsg = "Failure in the formatting the dbquery. Cannot continue processing."
                    self.promptlogs.append({"role": "assistant", "content": errormsg})
                    return self.promptlogs, None, errormsg

                self._updateslots(
                    self.slotdata, dbqueryformatter_answer["dbquery_format"]
                )
                '''
                self.current_state = "db-query"

                ss_output = {"status": "db-query", "details": self.slotdata}
                llm_response, error, _ = self.processresp.run(ss_output, "modular_prog")
                # self.dhistory[-1].update({"assistant": "db-query"})
                if error:
                    self.promptlogs.append({"role": "assistant", "content": str(error)})
                    return self.promptlogs, None, error
                return self._prepare_gm_response("db-query", llm_response)

            elif intent_detection == "dbretrieval-success":
                # Success in fetching the DB response
                # The answer could be a list of results or a single result
                # Pass the results to follow-up generator to generate the follow-up

                taskinput = {
                    #"extracted data": self.slotdata,
                    #"required data for booking": self.booking_keys,
                    "input": "data is fetched from the database and the results are available in the dialogue history",
                }

                self.dhistory.append(
                    {
                        #"role": "assistant",
                        "intent": intent_detection,
                        "domain": intent_answer["domain"],
                        "DB data": user_request,
                    }
                )

                return self._call_followup_for_missing_booking_data(user_request, taskinput, errormsg_base, current_turn, None)

            elif intent_detection == "dbretrieval-failure":
                # Failure in fetching the DB response
                self.dhistory.append(
                    {
                        #"role": "user",
                        "intent": intent_detection,
                        "domain": intent_answer["domain"],
                        "response": user_request,
                    }
                )

                taskinput = {"failure in db retrieval due to missing columns in the query": user_request}
                return self._call_followup_for_missing_booking_data(user_request, taskinput, errormsg_base, current_turn, None)

            else:
                self.dhistory.append(
                    {
                        #"role": "user",
                        "input": user_request,
                    }
                )                
                isbookingready, book_keys_miss = self._isbookingdatapresent(self.slotdata)
                if isbookingready:
                    logger.info(f"Information ready for booking {self.slotdata}")
                    taskinput = {"extracted data": self.slotdata}
                    return self._call_booking_formatter(taskinput, current_turn)

                taskinput = {
                    "input": user_request,
                    "extracted data": self.slotdata,
                    "missing data for booking": book_keys_miss,
                }
                return self._call_followup_for_missing_booking_data(user_request, taskinput, errormsg_base, current_turn, None)

    def get_booking_data(self):
        #TODO: Check this
        return self.slotdata
    
    def get_entity_slots(self):
        #TODO: Check this
        return self.slotdata
