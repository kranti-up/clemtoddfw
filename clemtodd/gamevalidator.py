#from clemgame import get_logger

from utils import processgenslots, checktimecloseness

import logging

logger = logging.getLogger(__name__)

class GameValidator:
    def __init__(self, game_name: str, gt_slots: dict):
        self.game_name = game_name
        self.gt_slots = gt_slots
        #self.gt_slots_info, self.gt_slots_book = self._processgtslots(gt_slots)
    

    
    def _compare_attributes(self, gt_slots: list, gen_slots: dict):
        missed_keys = []
        for key in gt_slots:
            if key not in gen_slots:
                logger.error(f"Key {key} not found in generated slots")
                missed_keys.append(key)
        return not missed_keys, missed_keys


    def _compare_slots(self, gt_slots: dict, gt_slots_fail_data: dict, gen_slots: dict):
        print(gt_slots)
        print(gt_slots_fail_data)
        print(gen_slots)

        logger.info(f"gt_slots: {gt_slots} gt_slots_fail_data: {gt_slots_fail_data} gen_slots: {gen_slots}")

        missed_keys = [key for key in gt_slots if key not in gen_slots]
        if missed_keys:
            logger.error(f"Keys of the ground truth slots and generated do not match {missed_keys}")
            return False, missed_keys

        missed_values = []
        for key, value in gt_slots.items():          
            if value != gen_slots[key]:
                if key in gt_slots_fail_data:
                    if gt_slots_fail_data[key] != gen_slots[key]:
                        data_match = False
                        if key in ["leaveat", "arriveby"]:
                            # If user agrees to the time, then why bother comparing it with ground truth?
                            data_match = True#checktimecloseness(gt_slots_fail_data[key], gen_slots[key])
                        if not data_match:
                            missed_values.append({"gt": {key: gt_slots_fail_data[key]}, "gen": {key: gen_slots[key]}})
                    else:
                        logger.info(f"Key {key} has the same value {gen_slots[key]} in the ground truth fail data and generated slots")
                else:
                    data_match = False
                    if key in ["leaveat", "arriveby"]:
                        # If user agrees to the time, then why bother comparing it with ground truth?
                        data_match = True#checktimecloseness(value, gen_slots[key])
                    
                    if not data_match:               
                        missed_values.append({"gt": {key: value}, "gen": {key: gen_slots[key]}})
                

        if missed_values:
            logger.error(f"Values of the ground truth slots and generated slots do not match {missed_values}")
            return False, missed_values                      
        
        return True, None

    def run(self, gen_slots: dict) -> bool:
        logger.info(f"Validating slots for game {self.game_name}")
        logger.info(f"Ground truth slots: {self.gt_slots}")
        logger.info(f"Generated slots: {gen_slots}")

        if not self.gt_slots or not gen_slots:
            logger.error(f"self.gt_slots: {self.gt_slots} gen_slots: {gen_slots}")
            return False, list(self.gt_slots.keys())
        
        modgen_slots = processgenslots(gen_slots)

        gt_domains = list(self.gt_slots.keys())
        gen_domains = list(modgen_slots.keys())
        missed_domains = [domain for domain in gt_domains if domain not in gen_domains]
        if missed_domains:
            logger.error(f"Domains of the ground truth slots and generated do not match {missed_domains}")
            return False, missed_domains
        
        data_match = {}
        for domain in self.gt_slots:
            if domain not in data_match:
                data_match[domain] = {"info": {"match": True, "missed": None},
                                     "book": {"match": True, "missed": None},
                                    "reqt": {"match": True, "missed": None}}

            if "info" in self.gt_slots[domain]:
                gt_fail_data = self.gt_slots[domain]["fail_info"] if "fail_info" in self.gt_slots[domain] else {}
                if "info" in modgen_slots[domain]:
                    info_match, missdata = self._compare_slots(self.gt_slots[domain]["info"], gt_fail_data, modgen_slots[domain]["info"])
                else:
                    info_match, missdata = self._compare_slots(self.gt_slots[domain]["info"], gt_fail_data, modgen_slots[domain])

                data_match[domain]["info"]["match"] = info_match
                data_match[domain]["info"]["missed"] = missdata

                if info_match:
                    # Now check the booking information                  
                    if "book" in self.gt_slots[domain]:
                        gt_fail_data = self.gt_slots[domain]["fail_book"] if "fail_book" in self.gt_slots[domain] else {}
                        if "book" in modgen_slots[domain]:
                            book_match, missdata = self._compare_slots(self.gt_slots[domain]["book"], gt_fail_data, modgen_slots[domain]["book"])
                        else:
                            book_match, missdata = self._compare_slots(self.gt_slots[domain]["book"], gt_fail_data, modgen_slots[domain])
                        data_match[domain]["book"]["match"] = book_match
                        data_match[domain]["book"]["missed"] = missdata

                else:
                    data_match[domain]["book"]["match"] = False
                    data_match[domain]["book"]["missed"] = None

            if "reqt" in self.gt_slots[domain]:
                if "reqt" in gen_slots[domain]:
                    reqt_match, missdata = self._compare_attributes(self.gt_slots[domain]["reqt"], modgen_slots[domain]["reqt"])
                else:
                    reqt_match, missdata = self._compare_attributes(self.gt_slots[domain]["reqt"], modgen_slots[domain])

                data_match[domain]["reqt"]["match"] = reqt_match
                data_match[domain]["reqt"]["missed"] = missdata

        # Process the data_match
        for domain, dataval in data_match.items():
            for key in ["info", "book", "reqt"]:
                if key in dataval and not dataval[key]["match"]:
                    logger.error(f"{key.capitalize()} information for domain {domain} does not match")
                    return False, dataval[key]["missed"]
            
        return True, None
        


if __name__ == "__main__":
    game_name = "llm-monolithic"

    gt_slots = {'restaurant': {'info': {'pricerange': 'moderate', 'area': 'centre'}, 'book': {'time': '14:00', 'day': 'friday', 'people': '1'}}}

    gt_slots_1 = {
              "taxi": {
                "info": { "arriveBy": "15:00" },
                "reqt": ["car type", "phone"],
                "fail_info": {}
              },
              "attraction": {
                "info": { "area": "east" },
                "reqt": ["entrance fee"],
                "fail_info": { "area": "west" }
              },
              "restaurant": {
                "info": { "name": "nandos" },
                "fail_info": { "name": "travellers rest" },
                "book": {
                  "people": "6",
                  "day": "monday",
                  "invalid": False,
                  "time": "15:00"
                },
                "fail_book": {}
              }
            } 
    gen_slots_1 = {
              "taxi": {"arriveBy": "15:00"},
              "attraction": {"area": "east"},
              "restaurant": {"name": "nandos", "bookpeople": "2", "bookday": "monday", "booktime": "15:00"},
            }  
    gen_slots = {'restaurant': {'reqt': {'phone': '01223337766', 'address': 'De Vere University Arms, Regent Street, City Centre'}, 'info': {'name': 'Restaurant One Seven', 'area': 'centre', 'pricerange': 'moderate', 'food': 'british'}, 'book': {'people': '1', 'day': 'Friday', 'time': '14:00'}}}  
    gt_slots_2 = {"train": {"info": {"destination": "ely", "day": "sunday", "arriveby": "13:30",
                                      "departure": "cambridge"}, "book": {"people": "8"}}}
    gen_slots_2 = {"train": {"info": {"departure": "cambridge", "destination": "ely",
                                      "day": "sunday", "arriveby": {"operator": "<=", "value": "13:30"}},
                             "book": {"people": "8"}, "reqt": {}}}

    gt_slots_3 = {"hotel": {"info": {"stars": "3", "type": "guesthouse", "parking": "yes"}, "book": {"people": "2", "stay": '1'}}}
    gen_slots_3 = {"hotel": {"info": {"type": "guesthouse", "parking": "yes", "stars": {"operator": "=", "value": "3"}},
                             "book": {"people": "2", "stay": 1}, "reqt": {}}}


    gvd = GameValidator(game_name, gt_slots_3)
    status, misses = gvd.run(gen_slots_3)
    print(status, misses)

