import sys
import re

# sys.path = [p for p in sys.path if "schmidtova" not in p and "mukherjee" not in p]
# print(sys.path)
import argparse
import pickle
import json
import tqdm
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

# from pynvml import *
from datasets import load_dataset
#import wandb
import logging
import transformers
import random

from dialogue_systems.hetaldsys.model import (
    FewShotPromptedLLM,
    SimplePromptedLLM,
    FewShotOpenAILLM,
    ZeroShotOpenAILLM,
    FewShotOpenAIChatLLM,
    ZeroShotOpenAIChatLLM,
    FewShotAlpaca,
    ZeroShotAlpaca,
)
from dialogue_systems.hetaldsys.loaders import load_mwoz, load_sgd
from dialogue_systems.hetaldsys.delex import (
    prepareSlotValuesIndependent,
    delexicalise,
    delexicaliseReferenceNumber,
)
from dialogue_systems.hetaldsys.definitions import (
    MW_FEW_SHOT_DOMAIN_DEFINITIONS,
    MW_ZERO_SHOT_DOMAIN_DEFINITIONS,
    SGD_FEW_SHOT_DOMAIN_DEFINITIONS,
    SGD_ZERO_SHOT_DOMAIN_DEFINITIONS,
    multiwoz_domain_prompt,
    sgd_domain_prompt,
)

from dialogue_systems.hetaldsys.database import MultiWOZDatabase
from dialogue_systems.hetaldsys.utils import (
    parse_state,
    ExampleRetriever,
    ExampleFormatter,
    print_gpu_utilization,
    SGDEvaluator,
)

# from mwzeval.metrics import Evaluator as MWEvaluator


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

transformers.set_seed(42)


class Interact:
    def __init__(self, model_id, player_dict, db_path):
        self.model_id = model_id
        self.player_dict = player_dict
        self.db_path = db_path
        logger.info(f"Interact: model_id: {model_id} db_path = {db_path}")
        self.__setup()

    def __setup(self):
        parser = argparse.ArgumentParser()
        # parser.add_argument("--cache_dir", type=str, default="/home/hudecek/hudecek/hf_cache")
        # parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo-0125")
        parser.add_argument("--model_name", type=str, default=self.model_id)
        parser.add_argument(
            "--faiss_db",
            type=str,
            default=self.db_path
            + "/multiwoz-context-db.vec",  # "games/todsystem/dialogue_systems/hetaldsys/data/multiwoz-context-db.vec"
        )
        parser.add_argument("--num_examples", type=int, default=0)
        parser.add_argument("--dials_total", type=int, default=1)
        parser.add_argument(
            "--database_path",
            type=str,
            default=self.db_path,  # "games/todsystem/dialogue_systems/hetaldsys/data/multiwoz_database"
        )
        parser.add_argument("--dataset", type=str, default="multiwoz")
        parser.add_argument("--context_size", type=int, default=3)
        parser.add_argument(
            "--ontology",
            type=str,
            default=self.db_path
            + "/ontology.json",  # "games/todsystem/dialogue_systems/hetaldsys/data/multiwoz_database/ontology.json"
        )
        parser.add_argument(
            "--n",
            type=str,
            default="clembench/dialogue_systems/hetaldsys/results",
        )
        parser.add_argument("--run_name", type=str, default="")
        # parser.add_argument("--use_gt_state", action='store_true')
        parser.add_argument("--use_gt_state", default=False)
        # parser.add_argument("--use_gt_domain", action='store_true')
        parser.add_argument("--use_gt_domain", default=False)
        # parser.add_argument("--use_zero_shot", action='store_true')
        parser.add_argument("--use_zero_shot", default=True)
        parser.add_argument("--verbose", action="store_true")
        # parser.add_argument("--goal_data", type=str, default=self.goal)
        args, _ = parser.parse_known_args()
        config = {
            "model_name": args.model_name,
            "faiss_db": args.faiss_db,
            "num_examples": args.num_examples,
            "dataset": args.dataset,
            "context_size": args.context_size,
            "use_gt_state": args.use_gt_state,
            "use_zero_shot": args.use_zero_shot,
            "use_gt_domain": args.use_gt_domain,
        }
        if "tk-instruct-3b" in args.model_name:
            model_name = "tk-3B"
        elif "tk-instruct-11b" in args.model_name:
            model_name = "tk-11B"
        elif "opt-iml-1.3b" in args.model_name:
            model_name = "opt-iml-1.3b"
        elif "opt-iml-30b" in args.model_name:
            model_name = "opt-iml-30b"
        elif "NeoXT" in args.model_name:
            model_name = "GPT-NeoXT-20b"
        elif "gpt-3.5" in args.model_name:
            model_name = "ChatGPT"
        elif "gpt-4" in args.model_name:
            model_name = "gpt-4o-2024-08-06"
        elif args.model_name == "alpaca":
            model_name = "Alpaca-LoRA"
        elif args.model_name == "claude-3-5-sonnet-20240620":
            model_name = "claude-3-5-sonnet-20240620"
        elif args.model_name == "remote-qwen2.5-34b-instruct-hf":
            args.model_name = "Qwen/Qwen2.5-32B-Instruct"
            model_name = "Qwen/Qwen2.5-32B-Instruct"
        else:
            # model_name = "GPT3.5"
            model_name = args.model_name  # "gpt-4o-2024-08-06"

        logger.info(
            f'Model name: {model_name}, args.model_name: {args.model_name} {args.model_name.startswith("claude-")}'
        )
        self.model_name = model_name
        self.dials_total = args.dials_total

        """
        if "mukherjee" not in args.run_name:
            wandb.init(
                project="llmbot-interact",
                entity="kclearns-student",
                config=config,
                settings=wandb.Settings(init_timeout=120, start_method="fork"),
            )
        else:
            wandb.init(
                project="llmbot-interact",
                entity="kclearns-student",
                config=config,
                settings=wandb.Settings(init_timeout=120, start_method="fork"),
            )

        wandb.run.name = f"{args.run_name}-{args.dataset}-{model_name}-examples-{args.num_examples}-ctx-{args.context_size}"
        self.report_table = wandb.Table(
            columns=[
                "id",
                "goal",
                "context",
                "raw_state",
                "parsed_state",
                "response",
                "predicted_domain",
            ]
        )
        """

        # self.mw_dial_goals = []
        # with open(args.goal_data, "rt") as fd:
        # data = json.load(fd)
        # mw_dial_goals = [dial['goal']['message'] for did, dial in data.items()]
        # self.mw_dial_goals.append(args.goal_data)
        # logger.info(f"Goals: {self.mw_dial_goals}")
        if args.model_name.startswith("text-"):
            model_factory = (
                ZeroShotOpenAILLM if args.use_zero_shot else FewShotOpenAILLM
            )
            self.model = model_factory(args.model_name)
            self.domain_model = ZeroShotOpenAILLM(args.model_name)
        elif (
            args.model_name.startswith("gpt-")
            or args.model_name.startswith("claude-")
            or "Qwen2.5" in args.model_name
            or "qwen" in args.model_name
            or "llama" in args.model_name
            or "Llama" in args.model_name
            or "deepseek" in args.model_name
        ):
            model_factory = (
                ZeroShotOpenAIChatLLM if args.use_zero_shot else FewShotOpenAIChatLLM
            )
            self.model = model_factory(args.model_name, self.player_dict["state_tracker"])
            self.domain_model = ZeroShotOpenAIChatLLM(args.model_name, self.player_dict["domain_detection"])
        elif any([n in args.model_name for n in ["opt", "NeoXT"]]):
            raise ValueError(f"Unknown model name: {args.model_name}")
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_name, cache_dir=args.cache_dir
            )
            model_w = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                low_cpu_mem_usage=True,
                cache_dir=args.cache_dir,
                device_map="auto",
                load_in_8bit=True,
            )
            model_factory = (
                SimplePromptedLLM if args.use_zero_shot else FewShotPromptedLLM
            )
            self.model = model_factory(model_w, tokenizer, type="causal")
            self.domain_model = SimplePromptedLLM(model_w, tokenizer, type="causal")
        elif "alpaca" in args.model_name:
            model_factory = ZeroShotAlpaca if args.use_zero_shot else FewShotAlpaca
            self.model = model_factory(model_name="Alpaca-LoRA")
            self.domain_model = ZeroShotAlpaca(model_name="Alpaca-LoRA")
        else:
            raise ValueError(f"Unknown model name: {args.model_name}")
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_name,
            )
            model_w = AutoModelForSeq2SeqLM.from_pretrained(
                args.model_name,
                low_cpu_mem_usage=True,
                cache_dir=args.cache_dir,
                device_map="auto",
                load_in_8bit=True,
            )
            model_factory = (
                SimplePromptedLLM if args.use_zero_shot else FewShotPromptedLLM
            )
            self.model = model_factory(model_w, tokenizer, type="seq2seq")
            self.domain_model = SimplePromptedLLM(model_w, tokenizer, type="seq2seq")

        with open(args.faiss_db, "rb") as f:
            faiss_vs = pickle.load(f)
        with open(args.ontology, "r") as f:
            self.ontology = json.load(f)
        if args.dataset == "multiwoz":
            self.domain_prompt = multiwoz_domain_prompt
            logger.info(f"Loading MultiWOZ database from {args.database_path}")
            self.database = MultiWOZDatabase(args.database_path)
            self.state_vs = faiss_vs
            self.delex_dic = prepareSlotValuesIndependent(args.database_path)
        else:
            self.domain_prompt = sgd_domain_prompt
            self.state_vs = faiss_vs
            self.delex_dic = None
        self.example_retriever = ExampleRetriever(faiss_vs)
        self.state_retriever = ExampleRetriever(self.state_vs)
        self.example_formatter = ExampleFormatter(ontology=self.ontology)

        self.dataset = args.dataset
        self.context_size = args.context_size
        self.use_zero_shot = args.use_zero_shot
        self.num_examples = args.num_examples
        self.history = []
        self.last_dial_id = None
        self.total = self.dials_total
        self.dialogue_id = 1
        self.total_state = {}
        # self.goal = self.mw_dial_goals[0]  # random.choice(mw_dial_goals)
        self.booking_done = False
        self.gmbookingresponse = None
        logger.info(f"Setup completed, model = {self.model}, config = {config}")

    def lexicalize(self, results, domain, response):
        if domain not in results:
            return response
        elif len(results[domain]) == 0:
            return response
        item = results[domain][0]
        extend_dct = {f"{domain}_{key}": val for key, val in item.items()}
        item.update(extend_dct)
        item.update({f"value_{key}": val for key, val in item.items()})
        item["choice"] = str(len(results[domain]))
        for key, val in item.items():
            x = f"[{key}]"
            if x in response:
                response = response.replace(x, val)
        return response
    
    def extract_word_if_valid(self, domain_value):
        # Trim leading/trailing spaces
        domain_value = domain_value.strip()

        # Check if the input consists of only a single word followed by a comma or period (with optional space after)
        match = re.match(r"^(\S+)[,.]\s*$", domain_value)  # Matches "word," or "word. " etc.
        
        if match:
            return match.group(1).lower()  # Return word in lowercase

        # Check if it's a single word without punctuation (only one word with no commas or periods)
        if re.match(r"^\S+$", domain_value):
            return domain_value.lower()  # Convert to lowercase and return

        # If there are multiple words, return the input as is (no processing)
        return domain_value  # Keep the original format

    def run(self, user_input, current_turn):
        """
        if self.booking_done:
            logger.info(f"Booking already done, returning booking response: {self.gmbookingresponse}")
            result = {"status": "follow-up", "details": self.gmbookingresponse}
            return self.history[-2], result, result
        """
        question = user_input

        dsys_logs = [{"role": "user", "content": f"User Input: {question}"}]

        retrieve_history = self.history + ["Customer: " + question]
        retrieved_examples = self.example_retriever.retrieve(
            "\n".join(retrieve_history[-self.context_size :]), k=20
        )
        logger.info(
            f"Entered run: question: {question}, retrieve_history: {retrieve_history}"
        )
        retrieved_domains = [example["domain"] for example in retrieved_examples]
        logger.info(f"Retrieved domains: {retrieved_domains}")
        #selected_domain, dp = self.domain_model(
        prompt, raw_answer, selected_domain = self.domain_model(
            self.domain_prompt,
            predict=True,
            history="\n".join(self.history[-2:]),
            utterance=f"Customer: {question.strip()}",
            current_turn=current_turn,
        )
        dsys_logs.append({"role": "assistant", "content": {"prompt": prompt, "raw_answer": raw_answer,
                                                           "answer":  f"Domain detection result: {selected_domain}"}})
        logger.info(f"Selected domain: {selected_domain}")
        #dsys_logs.append({"role": "user", "content": f"Detected Domain: {selected_domain}"})

        if self.dataset == "multiwoz":
            available_domains = list(MW_FEW_SHOT_DOMAIN_DEFINITIONS.keys())
        else:
            available_domains = list(SGD_FEW_SHOT_DOMAIN_DEFINITIONS.keys())
        logger.info(
            f"Available domains: {available_domains} self.dataset = {self.dataset}"
        )
        if selected_domain not in available_domains:
            selected_domain = self.extract_word_if_valid(selected_domain)
            if selected_domain not in available_domains:
                #base code
                selected_domain = random.choice(available_domains)

        if self.dataset == "multiwoz":
            domain_definition = (
                MW_ZERO_SHOT_DOMAIN_DEFINITIONS[selected_domain]
                if self.use_zero_shot
                else MW_FEW_SHOT_DOMAIN_DEFINITIONS[selected_domain]
            )
        else:
            domain_definition = (
                SGD_ZERO_SHOT_DOMAIN_DEFINITIONS[selected_domain]
                if self.use_zero_shot
                else SGD_FEW_SHOT_DOMAIN_DEFINITIONS[selected_domain]
            )
        retrieved_examples = [
            example
            for example in retrieved_examples
            if example["domain"] == selected_domain
        ]
        num_examples = min(len(retrieved_examples), self.num_examples)
        num_state_examples = 5
        state_examples = [
            example
            for example in self.state_retriever.retrieve(
                "\n".join(retrieve_history[-self.context_size :]), k=20
            )
            if example["domain"] == selected_domain
        ][:num_state_examples]
        positive_state_examples = self.example_formatter.format(
            state_examples[:num_state_examples],
            input_keys=["context"],
            output_keys=["state"],
        )
        # use_json=True)
        negative_state_examples = self.example_formatter.format(
            state_examples[:num_state_examples],
            input_keys=["context"],
            output_keys=["state"],
            corrupt_state=True,
        )
        response_examples = self.example_formatter.format(
            retrieved_examples[:num_examples],
            input_keys=["context", "full_state", "database"],
            output_keys=["response"],
            use_json=True,
        )

        state_prompt = domain_definition.state_prompt
        response_prompt = domain_definition.response_prompt

        logger.info(f"State prompt: {state_prompt}")
        logger.info(f"Response prompt: {response_prompt}")

        try:
            kwargs = {"history": "\n".join(self.history), "utterance": question.strip()}
            logger.info(f"kwargs: {kwargs}")
            if not self.use_zero_shot:
                kwargs["positive_examples"] = positive_state_examples
                kwargs["negative_examples"] = []  # negative_state_examples

            dsys_logs.append({"role": "user", "content": f"State tracker input: {question.strip()}"})
            #state, filled_state_prompt = self.model(
            prompt, raw_answer, state = self.model(
                state_prompt, predict=True, current_turn=current_turn, **kwargs
            )
            dsys_logs.append({"role": "assistant", "content": {"prompt": prompt, "raw_answer": raw_answer,
                                                           "answer":  f"State tracker response: {state}"}})            
            logger.info(f"State: {state}")
            #logger.info(f"Filled state prompt: {filled_state_prompt}")
        except Exception as e:
            logger.error(f"Error in model prediction: {e}")
            state = "{}"
            dsys_logs.append({"role": "assistant", "content": f"Error: {error} in fetching state tracker response: {state}"})            
            logger.info(f"Failure in fetching response -> State: {state}")            

        parsed_state = parse_state(state, default_domain=selected_domain)
        logger.info(f"Parsed state: {parsed_state}")

        dsyscontent = json.dumps(parsed_state) if isinstance(parsed_state, dict) else parsed_state
        #dsys_logs.append({"role": "user", "content": f"Extracted State: {dsyscontent}"})
        dsys_logs[-1]["content"]["answer"] += f"\nParsed State: {dsyscontent}"

        if selected_domain not in parsed_state:
            parsed_state[selected_domain] = {}
        if not isinstance(parsed_state[selected_domain], dict):
            parsed_state[selected_domain] = {}
        keys_to_remove = [
            k
            for k in parsed_state[selected_domain].keys()
            if k not in domain_definition.expected_slots
        ]
        for k in keys_to_remove:
            del parsed_state[selected_domain][k]
        try:
            for domain, ds in parsed_state.items():
                for slot, value in ds.items():
                    pass
        except:
            parsed_state = {selected_domain: {}}

        logger.info(f"Parsed state after processing: {parsed_state}")
        final_state = {}
        for domain, ds in parsed_state.items():
            if domain in available_domains:
                final_state[domain] = ds
        logger.info(f"Final state: {final_state}")

        for domain, dbs in final_state.items():
            if domain not in self.total_state:
                self.total_state[domain] = dbs
            else:
                for slot, value in dbs.items():
                    value = str(value)
                    if value not in ["dontcare", "none", "?", ""] and len(value) > 0:
                        self.total_state[domain][slot] = value

        logger.info(f"Belief State: {self.total_state}")
        logger.info(f"DB Query domain: {domain} total_state: {self.total_state}")

        dsyscontent = json.dumps(self.total_state) if isinstance(self.total_state, dict) else self.total_state
        dsys_logs.append({"role": "user", "content": f"DBQuery input: {dsyscontent}"})


        if self.dataset == "multiwoz":
            database_results = {
                domain: self.database.query(domain=domain, constraints=ds)
                for domain, ds in self.total_state.items()
                if len(ds) > 0
            }

            for domain, results in database_results.items():
                database_results[domain] = results[:5]
        else:
            database_results = self.turn["metadata"]["database"]

        dsyscontent = json.dumps(database_results) if isinstance(database_results, dict) else database_results
        dsys_logs.append({"role": "assistant", "content": f"DBQuery result: {dsyscontent}"})

        logger.info(f"Database Results: {database_results}")
        # print(
        #    f"Database Results: {database_results[selected_domain][0] if selected_domain in database_results and len(database_results[selected_domain]) > 0 else 'EMPTY'}",
        #    flush=True,
        # )

        try:
            kwargs = {
                "history": "\n".join(self.history),
                "utterance": question.strip(),
                "state": json.dumps(
                    self.total_state
                ),  # .replace("{", '<').replace("}", '>'),
                # "database": str(
                #    {
                #        domain: len(results)
                #        for domain, results in database_results.items()
                #    }
                # ),
                "database": str(database_results),
            }
            logger.info(f"kwargs: {kwargs}")
            # input()
            if not self.use_zero_shot:
                kwargs["positive_examples"] = response_examples
                kwargs["negative_examples"] = []

            dsys_logs.append({"role": "user", "content": f"Response Generator input: {json.dumps(kwargs)}"})

            # response, filled_prompt = "IDK", "-"
            #response, filled_prompt = self.model(
            prompt, raw_answer, response = self.model(
                response_prompt, predict=True, current_turn=current_turn, **kwargs
            )
            dsys_logs.append({"role": "assistant", "content": {"prompt": prompt, "raw_answer": raw_answer,
                                                           "answer":  f"Response Generator output: {response}"}})
            logger.info(f"Response: {response}")
            #logger.info(f"Filled prompt: {filled_prompt}")

            #dsys_logs.append({"role": "user", "content": f"Response Generator output: {response}"})

        except Exception as error:
            logger.error(f"Error in model prediction: {error}")
            response = ""

        #if self.dataset == "multiwoz":
            #logger.info(f"Delexicalising response")
            #response = delexicalise(response, self.delex_dic)
            #logger.info(f"Delexicalised response: {response}")
            # response = delexicaliseReferenceNumber(response)

        #logger.info(f"Response: {response}")
        # print(
        #    f"Lexicalized response: {self.lexicalize(database_results, selected_domain, response)}",
        #    flush=True,
        # )

        self.history.append("Customer: " + question)
        """
        self.report_table.add_data(
            f"dial_{self.dialogue_id}-turn_{current_turn}",
            " ".join(self.goal),
            " ".join(self.history),
            state,
            json.dumps(final_state),
            response,
            selected_domain,
        )
        """
        self.history.append("Assistant: " + response)

        logger.info(f"Returning response from dmsystems: {response}")
        result = {"status": "follow-up", "details": response}
        #return self.history[-2], result, result
        #return dsys_logs, response, response
        return dsys_logs, result, result

    def getgenslots(self):
        # return self.total_state[next(iter(self.total_state))]
        return self.total_state


if __name__ == "__main__":
    itact = Interact(
        "gpt-3.5-turbo-0125",
        "I'm looking for a cheap place to dine, preferably in the centre of town.",
    )
    itact.run(
        "I'm looking for a cheap place to dine, preferably in the centre of town.", 1
    )
