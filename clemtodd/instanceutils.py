import copy
import string
from typing import List, Dict, Tuple

from clemcore.utils import file_utils


class MultiWozDataInstance:
    def __init__(self, language: str, game_name: str, taskdialogs: Dict, game_id: int, config: Dict, tsystem: str):
        self.language = language
        self.taskdetails = taskdialogs[game_id]
        self.db_path = f"resources/data/{language}/multiwoz"
        self.tsytem = tsystem
        self.config = config
        self.game_name = game_name

    def _get_db_columns_service(self, domain):
        """
        Removed parking, internet columns
        In Schema.json, the possibe values for these columns are yes, no and free
        But in the db, the values are yes and no
        This causes the db to return empty results for queries with parking or internet with values 'free'
        """
        db_columns = {
            "restaurant": ["area", "pricerange", "name", "food"],
            "hotel": [
                "area",
                "pricerange",
                "name",
                "stars",
                "type",
                "internet",
                "parking",
            ],
            "attraction": ["area", "type", "name"],
            "train": [
                "destination",
                "day",
                "departure",
                "arriveby",
                "leaveat",
                "duration",
                "trainid",
            ],
        }
        return db_columns.get(domain, [])   


    def _fill_domain_schema(self, gamedata, domain_schema):
        #domain_schema = file_utils.load_json(f"resources/data/{self.language}/multiwoz/schema.json", self.game_name)        
        normalized_schema = {}
        gamedata["domaindbkeys"] = {}
        domains = list(gamedata["domains"].keys())

        for domain in domains:
            gamedata["domaindbkeys"][domain] = self._get_db_columns_service(domain)
            for entry in domain_schema:
                if entry["service_name"] != domain:
                    continue
                normalized_entry = copy.deepcopy(entry)
                normalized_entry["slots"] = [
                    {
                        "name": slot["name"].split("-")[1].strip(),
                        "is_categorical": slot["is_categorical"],
                        "possible_values": slot["possible_values"]
                        if "possible_values" in slot
                        else [],
                    }
                    for slot in entry["slots"]
                ]
                normalized_schema[domain] = normalized_entry
        gamedata["mwozschema"] = normalized_schema

    def _get_possible_values(self, domain, slot_name, domain_schema):
        for key, entry in domain_schema.items():
            if entry["service_name"] == domain:
                for slot in entry["slots"]:
                    #If the domain_schema is normalized in _fill_domain_schema, then already the slot name was split
                    #if slot["name"].split("-")[1].strip() == slot_name:
                    if slot["name"] == slot_name:
                        # parking and internet slots have possible values 'free' in schema.json and 'yes', 'no' in db
                        if slot_name in ["internet", "parking"]:
                            return ["yes", "no"]
                        return (
                            slot["possible_values"] if "possible_values" in slot else []
                        )
        return []


    def _update_properties(self, domain, domain_schema, keys, details):
        data = {}
        for key in keys:
            data[key] = {"type": "string"}
            values = self._get_possible_values(domain, key, domain_schema)
            if values:
                data[key]["enum"] = values
            details["properties"].update(data)

    def _fill_book_keys(self, gamedata):
        domaindata = gamedata["domains"]

        book_keys = {}
        for name, details in domaindata.items():
            for key, keyvalue in details.items():
                if key == "book":
                    if name not in book_keys:
                        book_keys[name] = {}

                    for k1, v1 in keyvalue.items():
                        if k1 in ["invalid", "pre_invalid"]:
                            continue
                        book_keys[name][k1] = v1
        gamedata["book_keys"] = book_keys


    def _fill_jsonschema(self, gamedata):
        gamedata["json_schema"] = self.config["json_schema"]
        json_schema = copy.deepcopy(gamedata["json_schema"])

        #domain = list(gamedata["domains"].keys())[0]
        #domain_schema = gamedata["mwozschema"]

        '''
        This can be removed as all the domain related fields are pre-defined in the json_schema
        db_details = json_schema["properties"]["details"]["oneOf"][1]
        self._update_properties(domain, domain_schema,
                                gamedata["domaindbkeys"], db_details)

        '''
        #booking_details = json_schema["properties"]["details"]["oneOf"][2]

        self._fill_book_keys(gamedata)
        '''
        for domain, book_details in gamedata["book_keys"].items():
            book_keys = list(book_details.keys())
            self._update_properties(domain, domain_schema, book_keys, booking_details)
            booking_details["required"] = book_keys

        gamedata["json_schema"] = {
            "name": "response_format_schema",
            "schema": json_schema,
        }
        '''
        gamedata["json_schema"] = json_schema
    def fill_mwoz_details(self, datainstance: Dict, domain_schema: Dict ):
        self._fill_domain_schema(datainstance, domain_schema)
        self._fill_jsonschema(datainstance)