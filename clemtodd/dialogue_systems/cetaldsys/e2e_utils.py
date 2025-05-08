import os
import re
from typing import Any, List, Optional, Tuple
import json
import openai
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from pprint import pprint
from dialogue_systems.cetaldsys.DST.dst import VALUES_FIX
from langchain import PromptTemplate
from langchain.llms.base import LLM
from langchain.pydantic_v1 import BaseModel, Field, ConfigDict
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain.agents import AgentType


import logging

logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)
dsys_logs = []


class E2E_InstrucTOD():
    def __init__(self, config, model_args, data_args, dataset, player_dict, dialogue_domains = None):
        self.model_args = model_args
        self.data_args = data_args
        self.player_llm = player_dict["llm_player"]
        self.player_agent = player_dict["llm_agent"]
        self.dialogue_domains = dialogue_domains
        self.domains = ["attraction", "hotel", "restaurant", "train", "taxi"]
        self.dataset = dataset
        self.config = config
                       
        self._setup()
        self.agents = self._load_agents()

    def _load_agents_one_domain(self):
        #Only support using OpenAI models currently (GPT3.5, GPT4)
        model_name = self.model_args.model_name_or_path_agent
        #model = OpenAI(model_name=model_name, 
        #               temperature=0)
        model = ChatOpenAI(model_name=model_name, temperature=0)

        agent = create_pandas_dataframe_agent(llm=model, 
                                              df=self.kb_df, 
                                              max_iterations=self.data_args.agent_max_iterations, 
                                              verbose=True,#self.data_args.verbose,
                                              allow_dangerous_code=True)
        
        return agent


    def _load_agents(self):
        agents = {}
        all_dbs = []
        if "openai" in self.model_args.model_name_or_path_agent:
            model_name = self.model_args.model_name_or_path_agent.split("/")[1]
        else:
            #if "gpt-3.5" not in self.model_args.model_name_or_path_agent:
            #    raise ValueError("Only gpt-3.5 is used for e2e")
            #else:
            model_name = self.model_args.model_name_or_path_agent
        
        #llm = OpenAI(model_name=model_name, temperature=0)
        #llm = ChatOpenAI(model_name=model_name, temperature=0)
        llm = LangChainLLMWrapper(self.player_agent)
        

        for domain in self.domains:
            file_path=os.path.join(self.data_args.mwoz_path, f"{domain}_db.json")
            logger.info(f"Loading {domain} database, file_path: {file_path}")
            data = json.loads(Path(file_path).read_text())

            df = pd.DataFrame(data)

            '''
            if domain != "taxi":
                df = pd.DataFrame(data)
            else:
                max_length = max(len(data["taxi_colors"]), len(data["taxi_types"]))
                df = pd.DataFrame({"taxi_colors": pd.Series(data["taxi_colors"]),
                                "taxi_types": pd.Series(data["taxi_types"]),
                                "taxi_phone": [data["taxi_phone"][0]] * max_length})                
            '''

            if domain == "attraction":
                df = df.drop(columns=["location"])
            elif domain == "hotel":
                df = df.drop(columns=["location", "price", "takesbookings", "type"])
                df['stars'] = df['stars'].astype(int)
                df = df[['name'] + [col for col in df.columns if col != 'name']]
            elif domain == "restaurant":
                df = df.drop(columns=["location", "type", "introduction", "signature", "id"])
                # df = df.rename(columns={'food': 'cuisine'})
                df = df[['name'] + [col for col in df.columns if col != 'name']]
                #print(df[df['name'].str.contains('nandos', case=False, na=False)])
            elif domain == "train":
                pass
            
            all_dbs.append(df)
            '''
            agent = create_pandas_dataframe_agent(llm=llm, 
                                                  df=df,
                                                  max_iterations=self.data_args.agent_max_iterations, 
                                                  verbose=self.data_args.verbose,
                                                  allow_dangerous_code=True)
            agents[domain] = agent  
            '''
        #if self.data_args.multi_only:
        agents = create_pandas_dataframe_agent(llm=llm, 
                                                df=all_dbs,
                                                max_iterations=self.data_args.agent_max_iterations, 
                                                verbose=self.data_args.verbose,
                                                allow_dangerous_code=True)

        self.attributes = [col for df in all_dbs for col in df.columns]
        logger.info(f"Loaded agents")
        return agents

    def completion(self, prompt, current_turn):
        logger.info(f"Prompt: {prompt}, current_turn: {current_turn}")
        messages=[{"role": "user", "content": prompt}]
        return self.player_llm(messages, current_turn, None)

    
    def completion_old(self, prompt):
        if "gpt-3.5-turbo" in self.model_args.model_name_or_path_agent or "gpt-4" in self.model_args.model_name_or_path_agent:
                
            try:
                #completion = openai.ChatCompletion.create(
                completion = openai.OpenAI(api_key="").chat.completions.create(
                        model=self.model_args.model_name_or_path_agent.replace("openai/", ""),
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0
                    )
            except: #Try twice
                #completion = openai.ChatCompletion.create(
                completion = openai.OpenAI(api_key="").chat.completions.create(
                    model=self.model_args.model_name_or_path_agent.replace("openai/", ""),
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0
                )
                
            response = completion.choices[0].message.content.strip()
        else:
            raise ValueError(f"model_name_or_path should be valid: {self.model_args.model_name_or_path_agent} for this setting")
        return response    

    
    def inference(self, input_idx=None, utterance=None):
        prompt_query_db_template = PromptTemplate(
            input_variables=self.config["PROMPT_TEMPLATES"]["template_e2e_query_database"]["input_variables"],
            template=self.config["PROMPT_TEMPLATES"]["template_e2e_query_database"]["template"],
        )
        prompt_rg_template = PromptTemplate(
            input_variables=self.config["PROMPT_TEMPLATES"]["template_e2e_rg"]["input_variables"],
            template=self.config["PROMPT_TEMPLATES"]["template_e2e_rg"]["template"],
        )
        examples_e2e_query_db = self.config["EXAMPLES"]["e2e_query_database"]
        examples_e2e_rg = self.config["EXAMPLES"]["e2e_response_generation"]
        instructions_e2e_query_db = self.config["INSTRUCTIONS"]["instruction_e2e_query_database"]
        instructions_e2e_rg = self.config["INSTRUCTIONS"]["instruction_e2e_rg"]

        prompts_e2e_query_db = []
        preds_e2e_query_db = []
        preds_e2e_dialog_acts = []
        prompts_e2e_rg = []
        preds = []
        gold_responses = []
        idxs = []
        turn_domains = []
        domains = []
        for idx, row in tqdm(self.dataset.iterrows()):
            sample_id = row["id"]
            turn_domain = row["turn_domain"]
            dialogue_context = row["prompt_e2e"].split("\n\n")[-1]
            gold_response = row["gold_response"]
            domain = row["domains"]
            
            prompt_query_db = prompt_query_db_template.format(instruction=instructions_e2e_query_db,
                                                              example=examples_e2e_query_db,
                                                              #dialogue_context=dialogue_context[:-9])
                                                              dialogue_context=utterance)
            if turn_domain in self.domains:
                print(f"DialgoueContext:\n{dialogue_context}")
                pred_query_db = self.completion(prompt_query_db)
                print(pred_query_db)
                with get_openai_callback() as cb:
                    
                    try:
                        if self.data_args.multi_only:
                            dialog_act = self.agents.run(f"If there are many fitting this criteria, pick a few to propose: {pred_query_db}")
                        else:
                            dialog_act = self.agents[turn_domain].run(f"If there are many fitting this criteria, pick a few to propose: {pred_query_db}")
                    except:
                        # response = str(e)
                        # if not response.startswith("Could not parse LLM output: `"):
                        #     raise e
                        dialog_act = "none"
                    #print(f"Total Tokens: {cb.total_tokens}")
                    #print(f"Prompt Tokens: {cb.prompt_tokens}")
                    #print(f"Completion Tokens: {cb.completion_tokens}")
                    #print(f"Total Cost (USD): ${cb.total_cost}")
                if dialog_act == "Agent stopped due to iteration limit or time limit.":
                    dialog_act = "none."
                #print(dialog_act)

                prompt_rg = prompt_rg_template.format(instruction=instructions_e2e_rg,
                                                      example=examples_e2e_rg,
                                                      dialogue_context=utterance,#dialogue_context[:-9],
                                                      dialogue_act=dialog_act)
            
                response = self.completion(prompt_rg)
            else:
                pred_query_db = "none"
                intermediate_step = "none"
                dialog_act = "none"
                prompt_rg = prompt_rg_template.format(instruction=instructions_e2e_rg,
                                                     example=examples_e2e_rg,
                                                     dialogue_context=dialogue_context[:-9],
                                                     dialogue_act=dialog_act)
                response = self.completion(prompt_rg)
            
        
            prompts_e2e_query_db.append(prompt_query_db)
            turn_domains.append(turn_domain)
            preds_e2e_query_db.append(pred_query_db)
            preds_e2e_dialog_acts.append(dialog_act)
            prompts_e2e_rg.append(prompt_rg)
            preds.append(response)
            gold_responses.append(gold_response)
            domains.append(domain)
            idxs.append(sample_id)
            
            if idx % self.data_args.save_every == 0:
                temp_save_path = self.data_args.save_path[:-4] + "_latestSave.csv"
                temp_df = pd.DataFrame({"id":idxs,
                                        "gold_response":gold_responses,
                                        "preds":preds,
                                        "prompts_e2e_query_db":prompts_e2e_query_db,
                                        "preds_e2e_query_db":preds_e2e_query_db,
                                        "preds_e2e_dialog_acts":preds_e2e_dialog_acts,
                                        "prompts_e2e_rg":prompts_e2e_rg,
                                        "turn_domain":turn_domains,
                                        "domains":domains,
                                        })
                temp_df.to_csv(temp_save_path)
        
        df = pd.DataFrame({"id":idxs,
                           "gold_response":gold_responses,
                           "preds":preds,
                           "prompts_e2e_query_db":prompts_e2e_query_db,
                           "preds_e2e_query_db":preds_e2e_query_db,
                           "preds_e2e_dialog_acts":preds_e2e_dialog_acts,
                           "prompts_e2e_rg":prompts_e2e_rg,
                           "turn_domain":turn_domains,
                           "domains":domains,
                           })
        df.to_csv(self.data_args.save_path)
        return df

           
    def _setup(self):
        self.prompt_query_db_template = PromptTemplate(
            input_variables=self.config["PROMPT_TEMPLATES"]["template_e2e_query_database"]["input_variables"],
            template=self.config["PROMPT_TEMPLATES"]["template_e2e_query_database"]["template"],
        )
        '''
        self.prompt_rg_template = PromptTemplate(
            input_variables=self.config["PROMPT_TEMPLATES"]["template_e2e_rg"]["input_variables"],
            template=self.config["PROMPT_TEMPLATES"]["template_e2e_rg"]["template"],
        )
        '''
        self.examples_e2e_query_db = self.config["EXAMPLES"]["e2e_query_database"]
        self.examples_e2e_rg = self.config["EXAMPLES"]["e2e_response_generation"]
        self.instructions_e2e_query_db = self.config["INSTRUCTIONS"]["instruction_e2e_query_database"]
        self.instructions_e2e_rg = self.config["INSTRUCTIONS"]["instruction_e2e_rg"]        

        self.prompt_bs_template = PromptTemplate(template=self.config["proxy_bs"]["template"],
                                                 input_variables=self.config["proxy_bs"]["input_variables"])
        self.bs_instruction = self.config["proxy_bs"]["instruction"]     
        self.bs_example = self.config["proxy_bs"]["example"]  

        self.prompt_rg_template = PromptTemplate(template=self.config["response_generation"]["template"],
                                                 input_variables=self.config["response_generation"]["input_variables"])         
        self.rg_instruction = self.config["response_generation"]["instruction"]
        self.rg_example = self.config["response_generation"]["example"]

        #self.kb_df = self._load_knowledge_base()
        #self.attributes = list(self.kb_df.columns)



        self.prompts_e2e_query_db = []
        self.preds_e2e_query_db = []
        self.preds_e2e_dialog_acts = []
        self.prompts_e2e_rg = []
        self.preds = []
        self.gold_responses = []
        self.idxs = []
        self.turn_domains = []
        #self.domains = []   
        print("Setup completed")

    def _load_knowledge_base(self):
        
        print("Loading KB")
        kb_ext = os.path.splitext(self.data_args.load_path)[-1]
        kb_path = self.data_args.load_path
        print(f"kb_ext = {kb_ext}, kb_path = {kb_path}")
        if kb_ext == ".json":
            kb_df = pd.read_json(kb_path)
        elif kb_ext == ".csv":
            kb_df = pd.read_csv(kb_path)
        elif kb_ext == ".xlsx":
            kb_df = pd.read_excel(kb_path) 
        else:
            raise ValueError(f"Knowledge base should be either json, csv or xslx. Current kb_path: {kb_path}")
        print("Loaded KB")
        return kb_df        
     


    def process_user_input(self, dialogue_context: str, utterance: str, current_turn: int):

        global dsys_logs
        dsys_logs = [{'role': 'user', 'content': f"User Input: {utterance}"}]


        prompt = self.prompt_bs_template.format(instruction=self.bs_instruction,
                                                example=self.bs_example,
                                                information=", ".join(self.attributes),
                                                dialogue_context=dialogue_context)
        
        prompt, raw_answer, answer = self.completion(prompt, current_turn)
        dsys_logs.append({'role': 'assistant', 'content': {"prompt": prompt, "raw_answer": raw_answer,
                                                           "answer": f"Belief State: {answer}"}})
        logger.info(f"Prompt Belief State: {answer}")

        with get_openai_callback() as cb:
            try:
                dsys_logs.append({'role': 'user', 'content': f"Invoking Agent to prepare the query: {answer}"})
                query_df = self.agents.run(f"If there are many fitting this criteria, pick a few to propose: {answer}") #Use fake intermediary belief state
            except Exception as e:
                response = str(e)
                if response.startswith("Could not parse LLM output: `"):
                    query_df = response.removeprefix("Could not parse LLM output: `").removesuffix("`")
                else:
                    query_df = "Error in fetching data from the database. Please try again."
            """
            if self.model_args.print_cost:
                print(f"Total Tokens: {cb.total_tokens}")
                print(f"Prompt Tokens: {cb.prompt_tokens}")
                print(f"Completion Tokens: {cb.completion_tokens}")
                print(f"Total Cost (USD): ${cb.total_cost}")        
            """
        logger.info(f"Query database results: {query_df}")
        if query_df == "Agent stopped due to iteration limit or time limit.":
            query_df = "There is nothing that fits the criteria. Ask for more information."

        dsys_logs.append({'role': 'assistant', 'content': f"Query Results: {query_df}"})
        dialogue_context_rg = dialogue_context+ f"USER: {utterance}\nSYSTEM:"#self._parse_dialogue_history(mode="rg") + f"USER: {utterance}\nSYSTEM:"

        logger.info(f"dialogue_context_rg: {dialogue_context_rg}")

        prompt = self.prompt_rg_template.format(instruction=self.rg_instruction,
                                                example=self.rg_example,
                                                dialogue_context=dialogue_context_rg,
                                                dialogue_act=query_df)

        logger.info(f"Prompt for Response Generator: {prompt}")
        dsys_logs.append({'role': 'user', 'content': f"Input for Response Generator: {prompt}"})

        prompt, raw_answer, answer = self.completion(prompt, current_turn)
        dsys_logs.append({'role': 'assistant', 'content': {"prompt": prompt, "raw_answer": raw_answer,
                                                           "answer": f"Response Generator: {answer}"}})
        
        logger.info(f"Response Generator: {answer}")
        return dsys_logs, answer, answer 
    
    def get_booking_data(self, preds):
        delex_dbs = self.delexicalize_dbs(self.data_args, self.data_args.ontology_path)
        return self.delexicalize_dsys(preds, delex_dbs)

    def _save_dsys_response(self, response, prompt_query_db, pred_query_db, dialog_act, prompt_rg, gold_response):
        self.prompts_e2e_query_db.append(prompt_query_db)
        self.turn_domains.append(self.turn_domain)
        self.preds_e2e_query_db.append(pred_query_db)
        self.preds_e2e_dialog_acts.append(dialog_act)
        self.prompts_e2e_rg.append(prompt_rg)
        self.preds.append(response)
        self.gold_responses.append(gold_response)
        self.domains.append(self.dialogue_domains)    
        self.idxs.append(7)

    def _save_final_result(self, idxs, gold_responses, preds, prompts_e2e_query_db, preds_e2e_query_db,
                            preds_e2e_dialog_acts, prompts_e2e_rg, turn_domains, domains):
        df = pd.DataFrame({"id":idxs,
                            "gold_response":gold_responses,
                            "preds":preds,
                            "prompts_e2e_query_db":prompts_e2e_query_db,
                            "preds_e2e_query_db":preds_e2e_query_db,
                            "preds_e2e_dialog_acts":preds_e2e_dialog_acts,
                            "prompts_e2e_rg":prompts_e2e_rg,
                            "turn_domain":turn_domains,
                            "domains":domains,
                            })
        df.to_csv(self.data_args.save_path) 

    def delexicalize_dsys(self, preds, dbs):
        ext_slots = {}
        phone_pattern = r"\d{11}"
        taxi_contact_number_pattern = r"\b0?\d{10}\b"
        for pred in preds:
            pred = pred.lower()

            for value_fix in VALUES_FIX:
                pred = pred.replace(value_fix, VALUES_FIX[value_fix])
            extracted_phone_nums = re.findall(phone_pattern, pred)
            if extracted_phone_nums:
                if "restaurant" in self.dialogue_domains:
                    if "restaurant" not in ext_slots:
                        ext_slots["restaurant"] = {}
                    ext_slots["restaurant"]["phone"] = " ".join(extracted_phone_nums)
                if "hotel" in self.dialogue_domains:
                    if "hotel" not in ext_slots:
                        ext_slots["hotel"] = {}
                    ext_slots["hotel"]["phone"] = " ".join(extracted_phone_nums)

            extracted_taxi_nums = re.findall(taxi_contact_number_pattern, pred)
            if extracted_taxi_nums:
                if "taxi" in self.dialogue_domains:
                    if "taxi" not in ext_slots:
                        ext_slots["taxi"] = {}
                    ext_slots["taxi"]["phone"] = " ".join(extracted_taxi_nums)

            #pred = re.sub(phone_pattern, "[value_phone]", pred)
            for domain in self.dialogue_domains:
                for k, values in dbs[domain].items():
                    for v in values:
                        if v in pred:
                            if domain in ext_slots:
                                ext_slots[domain][k] = v
                            else:
                                ext_slots[domain] = {k:v}
        logger.info(ext_slots)
        return ext_slots
            
    def delexicalize_dbs(self, data_args, ontology_path):
        domains = ["restaurant", "hotel", "train", "attraction", "taxi"]
        keep_data = {"restaurant":["address", "name", "food", "area", "pricerange", "phone", "postcode"],
                    "attraction":["name", "area", "address", "type", "postcode", "entrance fee"],
                    "hotel":["name", "address", "area", "phone", "postcode", "pricerange", "stars", "internet", "parking", "type"],
                    "train":["departure", "destination", "arriveBy", "day", "leaveAt", "price", "trainID", "duration"],
                    "taxi": ["taxi_types", "taxi_colors"]}
        dbs = {}
        for domain in domains:
            db_path = os.path.join(data_args.mwoz_path, f"{domain}_db.json")
            with open(db_path, "r") as f:
                db_data = json.load(f)
            db = {}
            for d in db_data: 
                for k, v in d.items():
                    if k in keep_data[domain]:
                        if k == "taxi_types":
                            use_key = "car type"
                        elif k == "taxi_colors":
                            use_key = "color"
                        else:
                            use_key = k
                        if use_key in db:
                            if isinstance(v, list):
                                for v_ in v:
                                    if v_ not in db[use_key]:
                                        db[use_key].append(v_.lower())
                            else:
                                if v not in db[use_key]:
                                    db[use_key].append(v.lower())
                        else:
                            if isinstance(v, list):
                                db[use_key] = [value.lower() for value in v]
                            else:
                                db[use_key] = [v.lower()]
            dbs[domain] = db

        with open(ontology_path, "r") as f:
            db_data = json.load(f)
        taxi_slots = ["departure", "destination", "arriveBy", "leaveAt", "type", "color"]
        book_slots = {"restaurant":["book time", "book day", "people"],
                    "hotel":["book day", "people", "book stay"],
                    "train":["people"]}

        #dbs["taxi"] = {}
        for slot in taxi_slots:
            if slot == "leaveAt":
                dbs["taxi"][slot] = db_data[f"taxi-leave at"]
            elif slot == "arriveBy":
                dbs["taxi"][slot] = db_data[f"taxi-arrive by"]
            else:
                if f"taxi-{slot}" in db_data:
                    dbs["taxi"][slot] = db_data[f"taxi-{slot}"]#db_data[f"taxi-semi-{slot}"]
                else:
                    continue

        for domain, slots in book_slots.items():
            for slot in slots:
                if slot == "people":
                    dbs[domain]["book-people"] = [value+" people" for value in db_data[f"{domain}-book {slot}"]] + [value+" person" for value in db_data[f"{domain}-book {slot}"]]
                else:
                    dbs[domain][slot] = db_data[f"{domain}-{slot}"]

        for domain in domains:
            if domain == "train":
                continue
            reordered = {k:v for k, v in dbs[domain].items() if k == "name"}
            for k, v in dbs[domain].items():
                if k != "name":
                    reordered[k] = v
            dbs[domain] = reordered
        return dbs

# old
# def delexicalize(df, delex_dbs):
#     delex_preds = []
#     for idx, row in df.iterrows():
#         pred = row["preds"]
#         domain = row["turn_domain"]
#         for k, values in delex_dbs[domain].items():
#             for v in values:
#                 if v in pred.lower():
#                     pred = pred.lower().replace(v, f"[{k.lower()}_value]")
#         delex_preds.append(pred)
#     df["delexicalized_preds"] = delex_preds
#     return df

def delexicalize(df, dbs, delex_column="preds"):
    delex_preds = []
    phone_pattern = r"\d{11}"
    for idx, row in tqdm(df.iterrows()):
        pred = row[delex_column].lower()
        domain = row["turn_domain"]
        for value_fix in VALUES_FIX:
            pred = pred.replace(value_fix, VALUES_FIX[value_fix])
        pred = re.sub(phone_pattern, "[value_phone]", pred)
        for k, values in dbs[domain].items():
            for v in values:
                if v in pred:
                    pred = pred.replace(v, f"[value_{k.lower()}]")
        delex_preds.append(pred)
    df["delexicalized_preds"] = delex_preds
    return df



def get_subset_multi(df):
    df['domain_length'] = df['domains'].apply(lambda x: len(x))
    filtered_df = df.loc[(df['domain_length'] == 2)].head(500).append(df.loc[(df['domain_length'] == 3)].head(500))
    return filtered_df


class LangChainLLMWrapper(LLM):
    player_llm: Any = Field(..., exclude=True)  # Exclude from Pydantic validation

    model_config = ConfigDict(extra="allow") 

    def __init__(self, agent_llm, **kwargs):
        super().__init__(**kwargs)
        self.agent_llm = agent_llm  # Store your custom LLM instance

    @property
    def _llm_type(self) -> str:
        return "custom-llm-wrapper"

    def _call(self, messages: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        """
        Calls the custom LLM framework and returns the output.
        """
        current_turn = kwargs.get('current_turn', -1)
        logger.info(f"LangChainLLMWrapper _call messages={messages} current_turn={current_turn}")

        global dsys_logs
        dsys_logs.append({'role': 'user', 'content': f"Model Prompt:\n{messages}"})

        prompt = [{'role': 'user', 'content': messages}]
        prompt, raw_answer, answer = self.agent_llm(prompt, current_turn, None)
        dsys_logs.append({'role': 'assistant', 'content': {"prompt": prompt, "raw_answer": raw_answer,
                                                           "answer": f"Model Response:\n{answer}"}})
        logger.info(f"LangChainLLMWrapper _call raw_answer={raw_answer} answer={answer}")
        return answer

