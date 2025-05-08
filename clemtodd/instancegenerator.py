import copy
import random
import string
from typing import List, Dict, Tuple

from clemcore.clemgame import GameInstanceGenerator
from instanceutils import MultiWozDataInstance

# set the name of the game in the script, as you named the directory
# this name will be used everywhere, including in the table of results
GAME_NAME = "clemtod"
# we will create 10 instances for each experiment; vary this as you wish
N_INSTANCES = 5
# if the generation involves randomness, remember to set a random seed
SEED = 123

LANGUAGE = "en"


class ClemTODSystemInstanceGenerator(GameInstanceGenerator):
    def __init__(self, game_name):
        # always do this to initialise GameInstanceGenerator
        super().__init__(game_name)
        self.game_name = game_name


    def _prepare_prompts(self, goal: str, tsystem: str, data: Dict) -> Dict[str, str]:
        prompt_file_names = {
            "initial_prompt_a": "prompt_a",
            "turn_prompt_a": "turn_prompt_a",
        }

        promptsdict = {}
        for file_name, match_key in prompt_file_names.items():
            filedata = self.load_template(
                f"resources/initial_prompts/{LANGUAGE}/{file_name}")
            promptsdict[match_key] = self.create_prompt(goal, filedata, None)

        if tsystem in ["monolithic_llm", "modular_llm"]:
            prompt_file_names = {
                "initial_prompt_b": "prompt_b",
                "turn_prompt_b": "turn_prompt_b",
            }

            for file_name, match_key in prompt_file_names.items():
                filedata = self.load_template(
                    f"resources/initial_prompts/{LANGUAGE}/{tsystem}/{file_name}")
                promptsdict[match_key] = self.create_prompt(goal, filedata, None)

        if tsystem in ["monolithic_llm", "modular_prog", "modular_llm"]:
            prompt_file_names = {
                "dbquery_prompt_b": "dbquery_prompt_b",
                "validbooking_prompt_b": "validbooking_prompt_b",
            }

            for file_name, match_key in prompt_file_names.items():
                filedata = self.load_template(
                    f"resources/initial_prompts/{LANGUAGE}/{tsystem}/{file_name}")
                promptsdict[match_key] = self.create_prompt(goal, filedata, None)                

        if tsystem in ["modular_prog", "modular_llm"]:
            prompt_file_names = {
                "initial_prompt_booking_formatter": "booking_formatter",
                "initial_prompt_dbquery_formatter": "dbquery_formatter",
                "initial_prompt_followup_generation": "followup_generation",
                "initial_prompt_intent_detection": "intent_detection",
                "initial_prompt_slot_extraction": "slot_extraction"
            }

            for file_name, match_key in prompt_file_names.items():
                filedata = self.load_template(
                    f"resources/initial_prompts/{LANGUAGE}/{tsystem}/{file_name}")
                '''
                if match_key in ["dbquery_formatter", "slot_extraction", "followup_generation"]:
                    db_schema = data["json_schema"]["schema"]["properties"]["details"]["oneOf"][1]["oneOf"]
                    book_schema = data["json_schema"]["schema"]["properties"]["details"]["oneOf"][2]["oneOf"]
                    prompt_schema = f"DB Schema: {db_schema}\nBooking Schema: {book_schema}"
                    promptsdict[match_key] = self.create_prompt(goal, filedata, prompt_schema)
                elif match_key == "booking_formatter":
                    db_schema = data["json_schema"]["schema"]["properties"]["details"]["oneOf"][1]["oneOf"]
                    book_schema = data["json_schema"]["schema"]["properties"]["details"]["oneOf"][2]["oneOf"]
                    prompt_schema = f"DB Schema: {db_schema}\nBooking Schema: {book_schema}"
                    #book_schema = data["json_schema"]["schema"]["properties"]["details"]["oneOf"][2]["oneOf"]
                    promptsdict[match_key] = self.create_prompt(goal, filedata, prompt_schema)
                else:
                '''
                promptsdict[match_key] = self.create_prompt(goal, filedata, None)


            if tsystem == "modular_llm":
                prompt_file_names = {
                    "turn_subsystem_prompt_b": "turn_ss_prompt_b"
                }

                for file_name, match_key in prompt_file_names.items():
                    filedata = self.load_template(
                        f"resources/initial_prompts/{LANGUAGE}/{tsystem}/{file_name}")
                    promptsdict[match_key] = filedata#self.create_prompt(goal, filedata, None)
        return promptsdict


    # define on_generate, a mandatory method
    def on_generate(self):
        num_instances = 0


        taskdialogs = self.load_json(
            f"resources/tasks/{LANGUAGE}/subset_taskdata_dev.json")
        config = self.load_json(
            f"resources/config/{LANGUAGE}/taskconfig.json")

        tot_instances = 0
        #game_ids = random.sample(range(len(taskdialogs)), N_INSTANCES)
        game_ids = range(len(taskdialogs))
        for tsystem in config["todsystems"]:
            experiment = self.add_experiment(tsystem)
            #for game_id in range(len(taskdialogs)):
            num_instances = 0
            for game_id in game_ids:
                if config["data_split"] != taskdialogs[game_id]["data_split"]:
                    continue

                if not any(topic in taskdialogs[game_id]["domains"] for topic in config["topics"]):
                    continue

                if not any(dtype in taskdialogs[game_id]["dialogue_type"] for dtype in config["dialogue_type"]):
                    continue


                instance = self.add_game_instance(experiment, game_id)
                instance["data"] = dict(taskdialogs[game_id])
                instance["data"]["filename"] = taskdialogs[game_id]["filename"]
                instance["data"]["db_path"] = f"clemtod/resources/data/{LANGUAGE}/multiwoz"#"games/todsystem/dialogue_systems/data/multiwoz"
                instance["data"]["tsystem"] = tsystem
                instance["data"]["tasktype"] = taskdialogs[game_id]["tasktype"]
                instance["data"]["statusmsg"] = config["statusmsg"]
                instance["data"]["n_turns"] = config["n_turns"]
                instance["data"]["liberal_processing"] = config["liberal_processing"]
                instance["data"]["booking_mandatory_keys"] = config["booking_mandatory_keys"]

                domain_schema = self.load_json(f"resources/data/{LANGUAGE}/multiwoz/schema.json")
                instanceutils = MultiWozDataInstance(LANGUAGE, GAME_NAME, taskdialogs, game_id, config, tsystem)
                instanceutils.fill_mwoz_details(instance["data"], domain_schema)

                promptsdict = self._prepare_prompts(taskdialogs[game_id]["message"], tsystem, instance["data"])
                instance["data"]["prompts"] = promptsdict


                num_instances += 1
                if num_instances == 2:
                    break
 
            tot_instances += num_instances

        print(
            f"Generated instances for -{self.game_name} game - {tot_instances} instances."
        )

    # an additional method, specific for our example
    def create_prompt(self, goal: str, prompt: str, slots: Dict) -> str:
        """Replace a prompt template with slot values."""
        text = string.Template(prompt).substitute(goal=goal, slots=slots)
        return text


if __name__ == "__main__":
    random.seed(SEED)
    # always call this, which will actually generate and save the JSON file
    ClemTODSystemInstanceGenerator(GAME_NAME).generate()
