import os
import json
import pandas as pd
from tqdm import tqdm
import logging

from dialogue_systems.cetaldsys.promptconstruct import PromptConstructor

logger = logging.getLogger(__name__)

class MWOZ_Dataset(PromptConstructor):
    def __init__(self,
                 config,
                 data_args):
        PromptConstructor.__init__(self, config)
        self.dataset = {"id":[],
                        "dialogue_id":[],
                        "dialogue_context":[],
                        "turn":[],
                        "prompt_dst":[],
                        "prompt_dst_update":[],
                        "prompt_rg":[],
                        "prompt_e2e":[],
                        "domains":[],
                        "turn_domain":[],
                        "gold_turn_bs":[],
                        "gold_bs":[],
                        "gold_act":[],
                        "gold_response":[],
                        "gold_database_result":[],
                        }
        
        print("Loading data...")
        self.all_data, self.testfiles, self.system_acts, self.ontology = self._get_mwoz_data(data_args.mwoz_path)
        print("Loading databases...")
        self.dbs_lexicalized = self._get_dbs_lexicalized(data_args.mwoz_path, data_args.db_format_type)
        self.idx = 0
        self.dialog_history_limit_dst = data_args.dialog_history_limit_dst
        self.dialog_history_limit_rg = data_args.dialog_history_limit_rg
        self.dialog_history_limit_e2e = data_args.dialog_history_limit_e2e
        self.single_domain_only = data_args.single_domain_only
        self.with_slot_description = data_args.with_slot_description
        self.with_slot_domain_diff = data_args.with_slot_domain_diff
        self.with_all_slots = data_args.with_all_slots
        self.all_domains = ["restaurant", "taxi", "hotel", "train", "attraction"]

        '''
        print("Processing mwoz...")
        for sample in tqdm(self.all_data):
            if sample in self.testfiles:
                dialogue_log = self.all_data[sample]["log"]
                self._process_dialogue_log(sample=sample,
                                           dialogue_log=dialogue_log)

        self.dataset = pd.DataFrame(self.dataset)
        if self.single_domain_only:
            for index, row in tqdm(self.dataset.iterrows()):
                if "sng" not in row["dialogue_id"].lower():
                    self.dataset.drop(index, inplace=True)

        for index, row in self.dataset.iterrows():
            if row["turn_domain"] == "":
                self.dataset.loc[index, 'turn_domain'] = row["domains"][0]
        '''
    def _get_mwoz_data(self, mwoz_path):
        data_path = os.path.join(mwoz_path, "data.json")
        testListFile_path = os.path.join(mwoz_path, "testListFile.json")
        system_acts_path = os.path.join(mwoz_path, "dialog_acts.json")
        ontology_path = os.path.join(mwoz_path, "ontology.json")

        with open(data_path, "r") as f:
            all_data = json.load(f)
            
        with open(testListFile_path, "r") as f:
            testfiles = f.read()
        testfiles = testfiles.split("\n")
        
        with open(system_acts_path, "r") as f:
            system_acts = json.load(f)
            
        with open(ontology_path, "r") as f:
            ontology = json.load(f)
            
        return all_data, testfiles, system_acts, ontology
    
    def _get_dbs_lexicalized(self, mwoz_path, format_type):
        domains = ["restaurant", "hotel", "train", "attraction"]
        keep_data = {"restaurant":["address", "area", "food", "name", "pricerange", "phone", "postcode"],
                    "attraction":["name", "area", "address", "type", "postcode"],
                    "hotel":["name", "address", "area", "phone", "postcode", "pricerange", "stars"],
                    "train":["departure", "destination"]}
        dbs_lexicalized = {}
        for domain in domains:
            db_path = os.path.join(mwoz_path, f"{domain}_db.json")
            with open(db_path, "r") as f:
                db_data = json.load(f)

            db_lexicalized = []
            if format_type == "1":
                for row in db_data:
                    row_keep = []
                    for key in keep_data[domain]:
                        if key in row:
                            row_keep.append(f"{key}: {row[key]}")
                    db_lexicalized.append(", ".join(row_keep))
            
            elif format_type == "2":
                #more concise db to fit in context length limit
                db_lexicalized.append(", ".join(keep_data[domain]))
                for row in db_data:
                    row_keep = []
                    for key in keep_data[domain]:
                        if key in row:
                            row_keep.append(f"{row[key]}")
                    db_lexicalized.append(", ".join(row_keep))
                    # db_lexicalized.append(", ".join([f"{row[key]}" for key in keep[domain]]))
            dbs_lexicalized[domain] = "\n".join(set(db_lexicalized))

        return dbs_lexicalized
    
    def _process_dialogue_log(self, sample, dialogue_log):

        dialog_history_memory_dst = []
        dialog_history_memory_rg = []
        dialog_history_memory_e2e = []
        dialog_history_dst = ""
        dialog_history_rg = ""
        dialog_history_e2e = ""
        turn_domain = ""
        domains = self._get_domains_from_log(dialogue_log)
        slots = self._get_slots_from_domains(domains=domains, 
                                             ontology=self.ontology,
                                             with_slot_description=self.with_slot_description,
                                             with_slot_domain_diff=self.with_slot_domain_diff,
                                             with_all_slots=self.with_all_slots) # or all
        if self.dialog_history_limit_dst == 0:
            example = self.examples["dst_dh0"]
        else:
            example = self.examples["dst_dh-1"]

        for turn_nb, turn in enumerate(dialogue_log):

            if turn_nb % 2 == 0:
                speaker = "USER"
            else:
                speaker = "SYSTEM"
            
            utterance = f"""{speaker}: {turn["text"]}\n"""
            dialog_act = turn["dialog_act"]
            cur_system_act = self.system_acts[sample.split(".")[0]][str((turn_nb//2)+1)]
            
            dialogue_context_dst = dialog_history_dst + utterance
            prompt_dst = self._build_prompt(mode="dst",
                                            slots=slots,
                                            example=example,
                                            dialogue_context=dialogue_context_dst)
            
            lexicalized_act = self._lexicalize_act(cur_system_act)
            dialogue_context_rg = dialog_history_rg + utterance + f"ACT:{lexicalized_act}\nSYSTEM:"
            prompt_rg = self._build_prompt(mode="response_generation",
                                            dialogue_context=dialogue_context_rg)
            
            dialogue_context_e2e = dialog_history_e2e + utterance + "SYSTEM:"
    
            turn_domain = self._get_domain_from_turn(turn_domain, cur_system_act)
            if turn_domain and turn_domain != "taxi":
                database = self.dbs_lexicalized[turn_domain]
            else:
                database = ""
            prompt_e2e = self._build_prompt(mode="e2e",
                                            database=database,
                                            dialogue_context=dialogue_context_e2e).replace("\n\n\n", "\n")

            dialog_history_dst, dialog_history_memory_dst = self._update_dialogue_memory(utterance, 
                                                                                         dialogue_log, 
                                                                                         self.dialog_history_limit_dst, 
                                                                                         dialog_history_memory_dst)
            dialog_history_rg, dialog_history_memory_rg = self._update_dialogue_memory(utterance, 
                                                                                       dialogue_log, 
                                                                                       self.dialog_history_limit_rg,
                                                                                       dialog_history_memory_rg)
            dialog_history_e2e, dialog_history_memory_e2e = self._update_dialogue_memory(utterance, 
                                                                                         dialogue_log, 
                                                                                         self.dialog_history_limit_e2e, 
                                                                                         dialog_history_memory_e2e) 
                
            metadata = turn["metadata"]
            bspn = {}
            if metadata:
                for domain in domains:
                    for k, v in metadata[domain].items():
                        for slot, value in v.items():
                            if isinstance(value, str) and value not in ["", "not mentioned", "none"]:
                                bspn[domain+"-"+slot] = value
            self.idx += 1
            if turn_nb % 2 == 0:
                self.dataset["gold_turn_bs"].append(dialog_act)
                self.dataset["dialogue_context"].append(dialogue_context_dst)
                self.dataset["gold_database_result"].append(None) 
                self.dataset["turn"].append(turn_nb//2)
                self.dataset["domains"].append(domains)
                self.dataset["id"].append(self.idx//2)
                self.dataset["dialogue_id"].append(sample)
                self.dataset["prompt_dst"].append(prompt_dst)
                self.dataset["prompt_dst_update"].append(prompt_dst)
                self.dataset["prompt_rg"].append(prompt_rg)
                self.dataset["prompt_e2e"].append(prompt_e2e)
                self.dataset["turn_domain"].append(turn_domain)
            else:
                self.dataset["gold_response"].append(utterance)
                self.dataset["gold_bs"].append(bspn)
                self.dataset["gold_act"].append(dialog_act)

    def _update_dialogue_memory(self, utterance, dialogue_log, dialog_history_limit, dialog_history_memory):
        if dialog_history_limit != 0:
            if dialog_history_limit == -1:
                dialog_history_limit = len(dialogue_log)
            if len(dialog_history_memory) >= dialog_history_limit:
                dialog_history_memory.pop(0)
            dialog_history_memory.append(utterance)

        dialog_history = "".join(dialog_history_memory)
        return dialog_history, dialog_history_memory
    
    def _lexicalize_act(self, act):
        if act == "No Annotation":
            return "None"
        
        lexicalized_acts = []
        lexicalize_mapping = {"leave": "leave time",
                              "arrive":"arrival time",
                              "departure":"departure place",
                              "post":"postcode",
                              "addr":"address"}

        for act, slot_values in act.items():


            if "request" in act.lower():
                requests = []
                for (slot, value) in slot_values:
                    slot = slot.lower()
                    if slot in lexicalize_mapping:
                        slot = lexicalize_mapping[slot]
                    if slot == "none":
                        break
                    else:
                        requests.append(slot)
                if requests:
                    lexicalized_act = "Request the user about " + ", ".join(requests) + "."
                    lexicalized_acts.append(lexicalized_act)

            elif "recommend" in act.lower():
                recommends = []
                for (slot, value) in slot_values:
                    slot, value = slot.lower(), value.lower()
                    if slot in lexicalize_mapping:
                        slot = lexicalize_mapping[slot]
                    if slot == "none":
                        break
                    else:
                        recommends.append(value)
                if recommends:
                    lexicalized_act = "Recommend the user for " + ", ".join(recommends) + "."
                    lexicalized_acts.append(lexicalized_act)

            elif "inform" in act.lower():
                informs = []
                for (slot, value) in slot_values:
                    slot, value = slot.lower(), value.lower()
                    if slot in lexicalize_mapping:
                        slot = lexicalize_mapping[slot]
                    if slot == "none":
                        break
                    else:
                        informs.append(f"the {slot} is {value}")
                if informs:
                    lexicalized_act = "Inform the user that " + ", ".join(informs) + "."  
                    lexicalized_acts.append(lexicalized_act)

            else:
                pass
        if lexicalized_acts:
            return " ".join(lexicalized_acts)
        else:
            return "None"
        
    def _get_domain_from_turn(self, domain, act):
        for k in act:
            turn_domain = k.lower().split("-")[0]
            if turn_domain in self.all_domains:
                return turn_domain
        return domain
            

    def _get_domains_from_log(self, dialogue_log):
        domains = []
        for log in dialogue_log:
            try:
                for domain_act in log["dialog_act"]:
                    domain = domain_act.split("-")[0].lower()
                    if domain in self.all_domains and domain not in domains:
                        domains.append(domain)
            except Exception as error:
                logger.error(f"Error: {error} while getting domains from log.\n{dialogue_log}")
        return domains