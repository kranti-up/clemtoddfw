from typing import Dict, Optional
from transformers import TrainingArguments
from dataclasses import dataclass, field
from langchain import PromptTemplate

from dialogue_systems.cetaldsys.DST.dst import SLOTS_DESCRIPTIONS, SLOTS_REVERSE_REMAPPING

class PromptConstructor():
    def __init__(self, 
                 config):
        self.config = config
        self.instructions = config["INSTRUCTIONS"]
        self.prompt_templates = config["PROMPT_TEMPLATES"]
        self.examples = config["EXAMPLES"]
        
    def _get_slots_from_domains(self, domains, ontology, with_slot_description, with_all_slots, with_slot_domain_diff):
        
        if with_all_slots:
            domains = ["restaurant", "train", "attraction", "hotel", "taxi"]
        
        slots = []
        for slot in list(ontology.keys()):
            splitted_slot = slot.split("-")
            if splitted_slot[0] in domains:
                if with_slot_domain_diff:
                    if splitted_slot[-1] not in slots:
                        slots.append(splitted_slot[-1])
                else:
                    slots.append(splitted_slot[0] + "-" + splitted_slot[-1])
        
        slots_info = []
        added_slots = []
        if with_slot_description:
            for slot in slots:
                splitted_slot = slot.split("-")
                if with_slot_domain_diff:
                    if slot in added_slots:
                        continue
                    slots_info.append(f"name: {slot}, description: {SLOTS_DESCRIPTIONS[slot.lower()]}")
                    added_slots.append(slot)
                else:
                    slots_info.append(f"name: {slot}, description: {SLOTS_DESCRIPTIONS[splitted_slot[1].lower()]}")

                    
            slots = slots_info
        
        slots_prompt = "\n".join(slots)
        if with_slot_domain_diff:
            return slots_prompt + f"\n\nDOMAINS: {', '.join(domains)}"
        else:
            return slots_prompt
        
    def _build_prompt(self, mode="", example="", dialogue_context="", ontology="", slots="", dialogue_acts="", belief_states="", database=""):
        prompt = ""
        if mode == "dst":
            instruction = self.instructions["instruction_with_slots"]
            template_variables = self.prompt_templates["template_with_slots"]
            template = PromptTemplate(input_variables= template_variables["input_variables"],
                                      template = template_variables["template"])
            prompt = template.format(instruction=instruction,
                                     slots=slots,
                                     example=example,
                                     dialogue_context=dialogue_context)
            
        elif mode == "dst_recorrect":
            instruction = self.instructions["instruction_with_slots_recorrect"]
            template_variables = self.prompt_templates["template_with_slots_recorrect"]
            template = PromptTemplate(input_variables= template_variables["input_variables"],
                                      template = template_variables["template"])            
            prompt = template.format(instruction=instruction,
                                    slots=slots,
                                    dialogue_context=dialogue_context,
                                    belief_states=belief_states)
            
        elif mode == "database_query":
            instruction = self.instructions["instruction_query_database"]
            template_variables = self.prompt_templates["template_query_database"]
            template = PromptTemplate(input_variables= template_variables["input_variables"],
                                      template = template_variables["template"])
            prompt = template.format(instruction=instruction,
                                     belief_states=belief_states)
            
        elif mode == "response_generation":
            example = self.config["EXAMPLES"]["response_generation"]
            
            instruction = self.instructions["instruction_response_generation"]
            template_variables = self.prompt_templates["template_response_generation"]
            template = PromptTemplate(input_variables = template_variables["input_variables"],
                                      template = template_variables["template"])
            prompt = template.format(instruction=instruction,
                                     example=example,
                                     dialogue_context=dialogue_context)
        elif mode == "e2e":
            instruction = self.instructions["instruction_e2e"]
            template_variables = self.prompt_templates["template_e2e"]
            template = PromptTemplate(input_variables = template_variables["input_variables"],
                                      template = template_variables["template"])
            prompt = template.format(instruction=instruction,
                                     database=database,
                                     dialogue_context=dialogue_context)

        else:
            raise ValueError("'mode' should be one of: [dst, dst_recorrect, database_query, response_generation, e2e]")
        
        return prompt        