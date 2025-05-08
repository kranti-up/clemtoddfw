import random
import string
import json
#from clemgame import get_logger
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def processgtslots(slots: dict) -> dict:
    modgt_slots = {}
    for domain, data in slots.items():
        domain_lower = domain.lower()
        modgt_slots[domain_lower] = {}
        for key, dvalue in data.items():
            key_lower = key.lower()

            #if key_lower not in ["fail_info", "fail_book"]:
            if isinstance(dvalue, dict):
                modgt_slots[domain_lower][key_lower] = {k.lower(): str(v).lower() for k, v in dvalue.items() if k not in ["invalid", "pre_invalid"]}
            elif isinstance(dvalue, list):
                modgt_slots[domain_lower][key_lower] = [str(v).lower() for v in dvalue]
    return modgt_slots


def preparegenslots(gen_slots: dict) -> dict:
    base_schema = {'hotel': {'info': ['internet', 'type', 'name', 'area', 'parking', 'pricerange', 'stars'],
                             'book': ['stay', 'day', 'people'],
                             'reqt': ['phone', 'area', 'postcode', 'address']},
                   'restaurant': {'info': ['food', 'name', 'pricerange', 'area'],
                                  'book': ['time', 'day', 'people'],
                                  'reqt': ['phone', 'postcode', 'address']},
                   'train': {'info': ['destination', 'departure', 'arriveby', 'leaveat', 'day'],
                             'book': ['people'],
                             'reqt': ['trainid']},
                   'attraction': {'info': ['name', 'area', 'type'],
                                   'book': [],
                                   'reqt': ['entrance fee', 'phone', 'postcode', 'address']},
                   'taxi': {'info': ['arriveby', 'leaveat'],
                            'book': [],
                            'reqt': ['phone', 'car type']}}

    gprocessed_slots = {}    
    for domain in gen_slots:
        if domain not in base_schema:
            logger.error(f"Domain {domain} not in the base schema: {base_schema.keys()}")
            return None
        gprocessed_slots[domain] = {}

        for key in gen_slots[domain]:
            keyfound = False

            use_key = key.lower()
            if domain == "train":
                if use_key in ["tickets", "bookpeople"]:
                    use_key = "people"
                elif use_key in ["train id"]:
                    use_key = "trainid"



            for stype in ["info", "book", "reqt"]:
                if use_key in base_schema[domain][stype]:
                    if stype not in gprocessed_slots[domain]:
                        gprocessed_slots[domain][stype] = {}


                    gprocessed_slots[domain][stype][use_key] = str(gen_slots[domain][key]).lower()
                    keyfound = True
                    break
            if not keyfound:
                print(f"Key {key} not found in the base schema for domain {domain}")

    return gprocessed_slots
    

def processgenslots(gen_slots: dict) -> dict:
    modgen_slots = {}
    for domain, data in gen_slots.items():
        domain_lower = domain.lower()
        if domain_lower not in modgen_slots:
            modgen_slots[domain_lower] = {}

        for key, value in data.items():
            key_lower = key.lower()
            if key_lower not in modgen_slots[domain_lower]:
                modgen_slots[domain_lower][key_lower] = {}
            if domain_lower == "train":
                if key_lower in ["info"]:
                    for k, v in value.items():
                        if isinstance(v, dict):
                            modgen_slots[domain_lower][key_lower][k.lower()] = str(list(v.values())[1]).lower()
                        else:
                            modgen_slots[domain_lower][key_lower][k.lower()] = str(v).lower()
                elif key_lower in ["reqt"]:
                    for k, v in value.items():
                        if isinstance(v, list):
                            modgen_slots[domain_lower][key_lower][k.lower()] = [str(i).lower() for i in v]
                        else:
                            modgen_slots[domain_lower][key_lower][k.lower()] = str(v).lower()

                #if key_lower in ["info", "reqt"]:
                #    modgen_slots[domain_lower][key_lower] = {k.lower(): str(v).lower() for k, v in value.items()}
                elif key_lower in ["book"]:
                    for k, v in value.items():
                        if k in ["bookpeople", "tickets"]:
                            modgen_slots[domain_lower][key_lower]["people"] = str(v).lower()
                        else:
                            modgen_slots[domain_lower][key_lower][k.lower()] = str(v).lower()
                else:
                    if key == "tickets":
                        #modgen_slots[domain_lower][key_lower]["people"] = str(value).lower()
                        modgen_slots[domain_lower]["people"] = str(value).lower()
                    else:
                        modgen_slots[domain_lower][key_lower] = str(value).lower()
            elif domain_lower == "restaurant":
                if key_lower in ["info"]:
                    for k, v in value.items():
                        if isinstance(v, dict):
                            modgen_slots[domain_lower][key_lower][k.lower()] = str(list(v.values())[1]).lower()
                        else:
                            modgen_slots[domain_lower][key_lower][k.lower()] = str(v).lower()

                elif key_lower in ["reqt"]:
                    for k, v in value.items():
                        if isinstance(v, list):
                            modgen_slots[domain_lower][key_lower][k.lower()] = [str(i).lower() for i in v]
                        else:
                            modgen_slots[domain_lower][key_lower][k.lower()] = str(v).lower()

                #if key_lower in ["info", "reqt"]:
                #    modgen_slots[domain_lower][key_lower] = {k.lower(): str(v).lower() for k, v in value.items()}
                elif key_lower in ["book"]:
                    for k, v in value.items():
                        if k == "bookpeople":
                            modgen_slots[domain_lower][key_lower]["people"] = str(v).lower()
                        elif k == "bookday":
                            modgen_slots[domain_lower][key_lower]["day"] = str(v).lower()
                        elif k == "booktime":
                            modgen_slots[domain_lower][key_lower]["time"] = str(v).lower()
                        else:
                            modgen_slots[domain_lower][key_lower][k.lower()] = str(v).lower()
                else:
                    if key == "bookpeople":
                        #modgen_slots[domain_lower][key_lower]["people"] = str(value).lower()
                        modgen_slots[domain_lower]["people"] = str(value).lower()
                    elif key == "bookday":
                        #modgen_slots[domain_lower][key_lower]["day"] = str(value).lower()
                        modgen_slots[domain_lower]["day"] = str(value).lower()
                    elif key == "booktime":
                        #modgen_slots[domain_lower][key_lower]["time"] = str(value).lower()
                        modgen_slots[domain_lower]["time"] = str(value).lower()
                    else:
                        #modgen_slots[domain_lower][key_lower][key.lower()] = str(value).lower()
                        modgen_slots[domain_lower][key_lower] = str(value).lower()
            elif domain_lower == "hotel":
                if key_lower in ["info"]:
                    for k, v in value.items():
                        if isinstance(v, dict):
                            modgen_slots[domain_lower][key_lower][k.lower()] = str(list(v.values())[1]).lower()
                        else:
                            modgen_slots[domain_lower][key_lower][k.lower()] = str(v).lower()

                elif key_lower in ["reqt"]:
                    for k, v in value.items():
                        if isinstance(v, list):
                            modgen_slots[domain_lower][key_lower][k.lower()] = [str(i).lower() for i in v]
                        else:
                            modgen_slots[domain_lower][key_lower][k.lower()] = str(v).lower()


                #if key_lower in ["info", "reqt"]:
                #    modgen_slots[domain_lower][key_lower] = {k.lower(): str(v).lower() for k, v in value.items()}
                elif key_lower in ["book"]:
                    for k, v in value.items():
                        if k == "bookpeople":
                            modgen_slots[domain_lower][key_lower]["people"] = str(v).lower()
                        elif k == "bookday":
                            modgen_slots[domain_lower][key_lower]["day"] = str(v).lower()
                        elif k == "booktime":
                            modgen_slots[domain_lower][key_lower]["time"] = str(v).lower()
                        elif k == "bookstay":
                            modgen_slots[domain_lower][key_lower]["stay"] = str(v).lower()
                        else:
                            modgen_slots[domain_lower][key_lower][k.lower()] = str(v).lower()
                else:
                    if key == "bookpeople":
                        #modgen_slots[domain_lower][key_lower]["people"] = str(value).lower()
                        modgen_slots[domain_lower]["people"] = str(value).lower()
                    elif key == "bookday":
                        #modgen_slots[domain_lower][key_lower]["day"] = str(value).lower()
                        modgen_slots[domain_lower]["day"] = str(value).lower()
                    elif key == "booktime":
                        #modgen_slots[domain_lower][key_lower]["time"] = str(value).lower()
                        modgen_slots[domain_lower]["time"] = str(value).lower()
                    elif key == "bookstay":
                        #modgen_slots[domain_lower][key_lower]["stay"] = str(value).lower()
                        modgen_slots[domain_lower]["stay"] = str(value).lower()
                    else:
                        #modgen_slots[domain_lower][key_lower][key.lower()] = str(value).lower()
                        modgen_slots[domain_lower][key.lower()] = str(value).lower()
            else:
                modgen_slots[domain_lower][key_lower] = str(value).lower()
    logger.info(f"Returning modgen_slots: {modgen_slots}")
    return modgen_slots

def cleanupanswer(prompt_answer: str) -> str:
    """Clean up the answer from the LLM DM."""
    #if "```json" in prompt_answer:
    prompt_answer = prompt_answer.replace("```json", "").replace("```", "")
    try:
        prompt_answer = json.loads(prompt_answer)
        #return prompt_answer
    except Exception as error:
        logger.error(f"Error in cleanupanswer: {error}")
        return error
    return prompt_answer

def funcdatasanitycheck(model_response: dict) -> dict:
    use_data = {}
    if "name" not in model_response:
        return None, f"Function name is missing in the model response. Cannot continue processing."

    if "arguments" in model_response:
        use_args_key = "arguments"

    elif "parameters" in model_response:
        use_args_key = "parameters"

    else:
        return None, f"Function arguments is missing in the model response. Cannot continue processing."

    func_name = model_response["name"]
    func_arguments = model_response[use_args_key]

    return {"name": func_name, "arguments": func_arguments}, None


def preparemodelresponse(func_name: str, func_arguments: dict) -> dict:

    if func_name is None or func_arguments is None:
        return None, f"Function data is missing in the model response. Cannot continue processing."

    use_data = {}

    if func_name in ["retrievefromrestaurantdb", "retrievefromhoteldb", "retrievefromtraindb"]:
        use_data["status"] = "db-query"

    elif func_name in ["validaterestaurantbooking", "validatehotelbooking", "validatetrainbooking"]:
        use_data["status"] = "validate-booking"

    elif func_name in ["followup"]:
        use_data["status"] = "follow-up"

    else:
        return None, f"Function name not matching with db/booking/follow-up query. Cannot proceed."

    use_data["details"] = {}
    if func_name in ["retrievefromrestaurantdb", "validaterestaurantbooking"]:
        use_data["details"]["domain"] = "restaurant"
        use_data["details"]["restaurant"] = func_arguments

    elif func_name in ["retrievefromhoteldb", "validatehotelbooking"]:
        use_data["details"]["domain"] = "hotel"
        use_data["details"]["hotel"] = func_arguments

    elif func_name in ["retrievefromtraindb", "validatetrainbooking"]:
        use_data["details"]["domain"] = "train"
        use_data["details"]["train"] = func_arguments

    elif func_name in ["followup"]:
        if "message" not in func_arguments:
            return None, f"Follow-up message is missing in the model response. Cannot proceed."
        use_data["details"] = func_arguments["message"]

    return use_data, None
  

def generate_reference_number(length=6):
    characters = string.ascii_uppercase + string.digits  # Uppercase letters and digits
    random_string = ''.join(random.choices(characters, k=length))
    return random_string

def checktimecloseness(time1, time2):
    #If the two time values are close to each other, with a gap of +- 60 minutes, return true, else false
    #Time values are already in the format of HH:MM
    #Convert the time values to datetime objects

    if isinstance(time2, dict):
        if "operator" in time2 and "value" in time2:
            time2 = time2["value"]
        else:
            return False


    time_format = "%H:%M"
    time1 = datetime.strptime(time1, time_format)
    time2 = datetime.strptime(time2, time_format)
    #Calculate the difference between the two time values
    time_diff = abs((time1 - time2).total_seconds() / 60)
    #Check if the difference is less than or equal to 30 minutes
    if time_diff <= 60:
        return True

    return False


if __name__ == "__main__":
    gen_slots = {"taxi": {"arriveBy": "15:00"}}
    print(processgenslots(gen_slots))
