import argparse
import os
import string
import numpy as np
import json
from typing import List
from collections import defaultdict

from clemcore.utils import file_utils
from tqdm import tqdm

import openai

from computecost import API_PRICE, calc_openai_cost
from dialogueeval_hf_wrapper import HFLocalWrapper

import logging

logger = logging.getLogger(__name__)


OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', "your-api-key")
OPENAI_ORGANIZATION = os.environ.get('OPENAI_ORGANIZATION', "your-org")
LLAMA3_API_KEY = os.environ.get('LLAMA3_API_KEY', "your-api-key")

def prepareprompt(file_dir_path, dialogue, user_goal):
    initial_prompt = file_utils.load_template(f"{file_dir_path}/resources/initial_prompts/en/dialogueeval", "todsystem")
    promptmessage = string.Template(initial_prompt).substitute(user_goal=user_goal, dialogue=dialogue)
    return promptmessage

def process_modelscore(model_details):
    exp_overall = []
    for exp in model_details:
        if exp in ["overall"]:
            continue
        exp_overall.append(model_details[exp]["overall"]["gen-dlg-score"])

    if not exp_overall:
        raise ValueError(f"No episodes found for computing model score for {model_details}")

    model_details["overall"] = {}
    model_details["overall"]["gen-dlg-score"] = {}
    for key in ["nat_us", "nat_ds", "coh_us", "coh_ds", "diver_us", "taskcompletion"]:
        model_details["overall"]["gen-dlg-score"][key] = round(np.mean([exp[key] for exp in exp_overall]), 2)


def parse_resp_score(episode_response, scores_dict):
    logger.info(f"Parsing response: {episode_response}")
    try:
        taskcompletion, nat_us, nat_ds, coh_us, coh_ds, diver_us = episode_response.split(",")
        taskcompletion = 1 if taskcompletion.lower() == "yes" else 0

        labels = ["nat_us", "nat_ds", "coh_us", "coh_ds", "diver_us", "taskcompletion"]
        values = [int(nat_us), int(nat_ds), int(coh_us), int(coh_ds), int(diver_us), int(taskcompletion)]

        for key, val in zip(labels, values):
            if key not in scores_dict:
                scores_dict[key] = []
            scores_dict[key].append(val)
    except Exception as error:
        logger.error(f"Error: {error}")
        print(f"Error: {error}")
        input()

def process_expscore(episode_details):
    scores = {}
    for episode in episode_details:
        if episode == "overall" or not episode_details[episode]:
            continue
        try:
            parse_resp_score(episode_details[episode]["gen-dlg-score"]["response"], scores)
        except Exception as error:
            print(f"Episode = {episode} {episode_details}")
            input()
   
    
    episode_details["overall"] = {}
    response = "gen-dlg-score"

    if scores:
        episode_details["overall"][response] = {}
        for key in ["nat_us", "nat_ds", "coh_us", "coh_ds", "diver_us", "taskcompletion"]:
            episode_details["overall"][response][key] = round(np.mean(scores[key]), 2)
    else:
        episode_details["overall"][response] = {"taskcompletion": 0.0, "nat_us": 0.0, "nat_ds": 0.0,
                                    "coh_us": 0.0, "coh_ds": 0.0, "diver_us": 0.0}


def process_gamescore(game_details):
    game_overall = []
    for model in game_details:
        if model in ["overall", "corpus-episode-dlgs"]:
            continue
        game_overall.append(game_details[model]["overall"]["gen-dlg-score"])

    if not game_overall:
        raise ValueError(f"No episodes found for computing game score for {game_details}")

    game_details["overall"] = {}
    
    game_details["overall"]["gen-dlg-score"] = {}
    for key in ["nat_us", "nat_ds", "coh_us", "coh_ds", "diver_us", "taskcompletion"]:
        game_details["overall"]["gen-dlg-score"][key] = round(np.mean([exp[key] for exp in game_overall]), 2)



    corpus_overall = []
    for episode, episode_values in game_details["corpus-episode-dlgs"].items():
        if episode == "overall":
            continue

        for key, value in episode_values.items():
            if key != "corpus-dlg-score":
                continue
            corpus_overall.append(value)

    total_values = defaultdict(int)
    for score_dict in corpus_overall:
        for key, value in score_dict.items():
            total_values[key] += value[0]

    total_values = dict(total_values)    
    game_details["corpus-episode-dlgs"]["overall"] = {key: round(value/len(corpus_overall), 2) for key, value in total_values.items()}


def process_dialogue_scores(results_file):
    with open(results_file, "r") as f:
        results = json.load(f)

    for game in results:
        for model in results[game]:
            if model in ["overall", "corpus-episode-dlgs"]:
                continue
            for exp in results[game][model]:
                if exp in ["overall"]:
                    continue
                #print(f"Processing {game}--{model}--{exp}")
                #input()
                process_expscore(results[game][model][exp])
            process_modelscore(results[game][model])
        process_gamescore(results[game])

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Scores computed and saved to {results_file}")





def getscore(hfwrapper, prompt, logdata, model_name="gpt-4o-2024-08-06"):

    tokens_data = None

    if model_name == "gpt-4o-2024-08-06":
        api_key = OPENAI_API_KEY
        organization = OPENAI_ORGANIZATION
        client = openai.OpenAI(api_key=api_key, organization=organization)

        completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=100,
            )


    elif model_name == "meta-llama/Llama-3.3-70B-Instruct":
        api_key = LLAMA3_API_KEY
        if api_key != "your-api-key":
            client = openai.OpenAI(api_key=LLAMA3_API_KEY,
                                    base_url="https://openrouter.ai/api/v1")

            completion = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0,
                    max_tokens=100,
                )

            tokens_data = {"prompt_tokens": completion.usage.prompt_tokens,
                        "completion_tokens": completion.usage.completion_tokens,
                        "prompt_tokens_details": {"cached_tokens": 0}}
            
            if "prompt_tokens_details" in completion.usage and completion.usage.prompt_tokens_details:
                if "cached_tokens" in completion.usage.prompt_tokens_details:
                    tokens_data["prompt_tokens_details"]["cached_tokens"] = completion.usage.prompt_tokens_details.cached_tokens

        else:
            completion = hfwrapper.generate_response(
                temperature=0,
                messages=[
                        {"role": "user", "content": prompt}
                    ],
                tool_schema=None,
                request_timeout=10,
            )

    logdata["model"] = model_name
    logdata["prompt"] = prompt
    logdata["usage"] = tokens_data
    logdata["response"] = completion.choices[0].message.content


    #logdata["cost"], logdata["tokens"] = calc_openai_cost(model_name, tokens_data)
    logdata["cost"] = 0#round(logdata["cost"], 2)


def getcorpusdialogue(dialogue):
    dialogue_processed = ""
    
    for index, turn in enumerate(dialogue):
        if index%2 == 0:
            role = "User"
        else:
            role = "System"
        dialogue_processed += role + ":\n" + turn["utterance"].strip() + "\n"
        if role == "System":
            dialogue_processed += "\n"
    return dialogue_processed.strip(), len(dialogue)/2

def getdsystemgendialogue(dialogue_data):
    dsystem_dialogue = ""
    for turn in dialogue_data:
        if "user" in turn:
            dsystem_dialogue += "User:\n"+turn["user"] + "\n"
        if "system" in turn:
            systext = turn["system"].replace("Response Generator:", "").strip()
            systext = systext.replace("Response Generator output:", "").strip()
            systext = systext.replace("Agent Follow-up:", "").strip()
            dsystem_dialogue += "System:\n"+systext + "\n\n"
    dsystem_dialogue = dsystem_dialogue.strip()

    return dsystem_dialogue, len(dialogue_data)


def computecorpusdialogue_score(hfwrapper, base_dir, results, score_model_name):

    #corpus_dir_path = os.path.join(base_dir, "corpus_dialogues")
    #os.makedirs(corpus_dir_path, exist_ok=True)


    #for model in os.listdir(base_dir):
    for model in tqdm(os.listdir(base_dir), desc="Scoring GroundTruth Dialogues"):
        if not os.path.isdir(os.path.join(base_dir, model)):
            continue
        model_path = os.path.join(base_dir, model)
        for game in os.listdir(model_path):
            game_path = os.path.join(model_path, game)

            for exp in os.listdir(game_path):
                exp_path = os.path.join(game_path, exp)
                if not os.path.isdir(exp_path):
                    continue

                for episode in os.listdir(exp_path):
                    if episode.endswith(".json"):
                        continue
                    episode_path = os.path.join(exp_path, episode)
                    for filename in os.listdir(episode_path):
                        if filename not in ["instance.json"]:
                            continue

                        with open(os.path.join(episode_path, "instance.json"), "r") as f:
                            instance_data = json.load(f)

                        user_goal = instance_data["data"]["message"]
                        corpusdialogue = instance_data["data"]["corpususer"]
                        corpusdialogue, corpus_turns = getcorpusdialogue(corpusdialogue)

                        #corpsu_dialg_file_path = os.path.join(corpus_dir_path, episode)
                        #os.makedirs(corpsu_dialg_file_path, exist_ok=True)
                        #with open(os.path.join(corpsu_dialg_file_path, "corpus_dialogue.txt"), "w") as f:
                        with open(os.path.join(episode_path, "gt_corpus_dialogue.txt"), "w") as f:
                            f.write(corpusdialogue)

                        file_dir_path = base_dir.split("/")[:-2]
                        file_dir_path = "/".join(file_dir_path) + "/clemtod"                           
                        promptmessage_corpus = prepareprompt(file_dir_path, corpusdialogue, user_goal)

                        if "corpus-episode-dlgs" not in results[game]:
                            results[game]["corpus-episode-dlgs"] = {}
                        corpus_dlg_score = {}
                        '''
                        getscore(hfwrapper, promptmessage_corpus, corpus_dlg_score, score_model_name)

                        scores = {}
                        parse_resp_score(corpus_dlg_score["response"], scores)
                        corpus_dlg_score.update({"corpus_turns": corpus_turns, "corpusdialogue": corpusdialogue,
                                                 "filename": instance_data["data"]["filename"],
                                                 "episode": episode,
                                                 "corpus-dlg-score": scores})

                        results[game]["corpus-episode-dlgs"][episode] = corpus_dlg_score
                        logger.info(f"Scored {model}--{episode}")
                        '''
                        break
                    #break
                #break
            #break
        break
    logger.info("Completed scoring of corpus dialogues")


def compute_scores(base_dir, score_model_name):
    results = {}

    if "meta-llama" in score_model_name or "Qwen" in score_model_name:
        hfwrapper = HFLocalWrapper(model_id=score_model_name, max_new_tokens=100)
    else:
        hfwrapper = None

    print("Loaded the model")
    input()

    #for model in os.listdir(base_dir):
    for model in tqdm(os.listdir(base_dir), desc="Scoring Generated Dialogues"):
        if not os.path.isdir(os.path.join(base_dir, model)):
            continue
        #if model in ['gpt-4o-2024-08-06-t0.0--gpt-4o-2024-08-06-t0.0', 'Llama-3.3-70B-Instruct-t0.0--Llama-3.3-70B-Instruct-t0.0',
        #                'Qwen2.5-32B-Instruct-t0.0--Qwen2.5-32B-Instruct-t0.0']:
        #    continue
        model_path = os.path.join(base_dir, model)
        for game in os.listdir(model_path):
            if game not in results:
                results[game] = {}
            if model not in results[game]:
                results[game][model] = {}
            game_path = os.path.join(model_path, game)

            for exp in os.listdir(game_path):

                if exp not in results[game][model]:
                    results[game][model][exp] = {}
                exp_path = os.path.join(game_path, exp)
                if not os.path.isdir(exp_path):
                    continue
                num_episodes = 0
                for episode in os.listdir(exp_path):
                    if episode.endswith(".json"):
                        continue
                    num_episodes += 1
                    episode_path = os.path.join(exp_path, episode)

                    required_files = {"dialogue.json", "interactions.json", "instance.json"}

                    # Check if required_files is a subset of the files in the directory
                    if not required_files.issubset(set(os.listdir(episode_path))):
                        results[game][model][exp][episode] = {}
                        continue

                    with open(os.path.join(episode_path, "dialogue.json"), "r") as f:
                        dialogue_data = json.load(f)

                    with open(os.path.join(episode_path, "interactions.json"), "r") as f:
                        interaction_data = json.load(f)    

                    with open(os.path.join(episode_path, "instance.json"), "r") as f:
                        instance_data = json.load(f)                                                  


                    user_goal = interaction_data["Evaluation"]["goal"]
                    prompt_dialogue, gen_dlg_turns = getdsystemgendialogue(dialogue_data)
                    with open(os.path.join(episode_path, "gen_cleaned_dialogue.txt"), "w") as f:
                        f.write(prompt_dialogue)

                    file_dir_path = base_dir.split("/")[:-2]
                    file_dir_path = "/".join(file_dir_path) + "/clemtod"
                    promptmessage_gen = prepareprompt(file_dir_path, prompt_dialogue, user_goal)


                    results[game][model][exp][episode] = {}
                    results[game][model][exp][episode]["n_turns"] = interaction_data["Evaluation"]["n_turns"]
                    results[game][model][exp][episode]["play_turns"] = interaction_data["Evaluation"]["play_turns"]
                    results[game][model][exp][episode]["gen_turns"] = gen_dlg_turns
                    results[game][model][exp][episode]["gendialogue"] = prompt_dialogue

                    if "filename" in interaction_data["Evaluation"]:
                        results[game][model][exp][episode]["filename"] = interaction_data["Evaluation"]["filename"]
                    else:
                        results[game][model][exp][episode]["filename"] = instance_data["data"]["filename"]

                    results[game][model][exp][episode]["gen-dlg-score"] = {}
                    #getscore(hfwrapper, promptmessage_gen, results[game][model][exp][episode]["gen-dlg-score"], score_model_name)
                    #logger.info(f"Scored {model}--{episode}")
                    #break
                #break
            #break
        #break

    logger.info("Completed scoring of generated dialogues")
    computecorpusdialogue_score(hfwrapper, base_dir, results, score_model_name)
    '''
    if "meta-llama/Llama-3.3-70B-Instruct" in score_model_name:
        suffix = "llama"
    elif "gpt-4o-2024-08-06" in score_model_name:
        suffix = "gpt"
    else:
        raise ValueError(f"Model name {score_model_name} not supported")

    with open(os.path.join(base_dir, f"dialoguemetric_{suffix}.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"Scores computed and saved to dialoguemetric_{suffix}.json")
    '''
base_dir = "/home/users/kranti/project/kranti/testtodsystem/monollm/clembench/cross_mono_multi_1/"
#compute_scores(base_dir, "gpt-4o-2024-08-06")
#compute_scores(base_dir, "meta-llama/Llama-3.3-70B-Instruct")
compute_scores(base_dir, "Qwen/Qwen2.5-32B-Instruct")

#process_dialogue_scores(os.path.join(base_dir, "dialoguemetric_llama.json"))
#process_dialogue_scores(os.path.join(base_dir, "dialoguemetric_gpt_1.json"))