import os
import string
import numpy as np
import json
from tqdm import tqdm
from clemcore.utils import file_utils
import openai
from dialogueeval_hf_wrapper import HFLocalWrapper
import random

SEED = 42


OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', "your-api-key")
OPENAI_ORGANIZATION = os.environ.get('OPENAI_ORGANIZATION', "your-org")
LLAMA3_API_KEY = os.environ.get('LLAMA3_API_KEY', "your-api-key")


def getturnpairs(dialogue):
    turns = dialogue.split("\n")
    turn_pairs = []
    for i in range(0, len(turns), 2):
        if i + 1 < len(turns):
            turn_pairs.append((turns[i], turns[i + 1]))

    print(f"Number of turn pairs: {len(turn_pairs)}")
    print(turn_pairs)
    input()
    return turn_pairs

def getpartialdialogue(dialogue):
    lines = [line.strip() for line in dialogue.strip().split('\n') if line.strip()]

    if len(lines) <= 4:
        return dialogue


    # Extract turn pairs
    turn_pairs = []
    i = 0
    try:
        while (i < len(lines) - 1) and (i+2 < len(lines) - 1):
            if lines[i].startswith("User:") and lines[i+2].startswith("System:"):
                user_turn = lines[i+1]#lines[i][len("User:"):].strip()
                system_turn = lines[i+3]# lines[i+1][len("System:"):].strip()
                conversation_turn = "User:" + "\n" + user_turn + "\n" + "System:" + "\n" + system_turn
                #turn_pairs.append((user_turn, system_turn))
                turn_pairs.append(conversation_turn)
                i += 2
            else:
                i += 1
    except Exception as error:
        print(f"Error parsing dialogue: {error}")
        #print(f"Dialogue: {dialogue}")
        input()
        return dialogue

    #Drop random turn pairs and merge the rest
    num_turn_pairs = len(turn_pairs)
    #Skipping the first and last turn pairs
    if num_turn_pairs <= 2:
        return dialogue
    #Drop 50% of the turn pairs
    num_turn_pairs_to_drop = int((num_turn_pairs-2) * 0.5)

    random.seed(SEED)
    indices_to_drop = random.sample(range(1, num_turn_pairs-1), num_turn_pairs_to_drop)
    for i in sorted(indices_to_drop, reverse=True):
        turn_pairs.pop(i)
    dialogue = ""
    for i in range(len(turn_pairs)):
        dialogue += turn_pairs[i] + "\n"
    #print(dialogue)
    #input()

    return dialogue




def prepareprompt(file_dir_path, dialogue1, dialogue2):
    initial_prompt = file_utils.load_template(f"{file_dir_path}/resources/initial_prompts/en/dialogue_human_style", "todsystem")
    promptmessage = string.Template(initial_prompt).substitute(dialogue1=dialogue1, dialogue2=dialogue2)
    return promptmessage

def getscore(hfwrapper, prompt, model_name="gpt-4o-2024-08-06"):
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


    elif model_name == "meta-llama/llama-3.3-70b-instruct":
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
    return completion.choices[0].message.content

def parse_resp_score(episode_response, scores_dict):
    print(episode_response)
    try:
        nat_us_1, coh_us_1, diver_us_1, human_us_1, nat_us_2, coh_us_2, diver_us_2, human_us_2 = episode_response.split(",")
        human_us_1 = 1 if human_us_1.lower() == "yes" else 0
        human_us_2 = 1 if human_us_2.lower() == "yes" else 0

        labels = ["nat_us_1", "coh_us_1", "diver_us_1", "human_like_1", "nat_us_2", "coh_us_2", "diver_us_2", "human_like_2"]
        values = [int(nat_us_1), int(coh_us_1), int(diver_us_1), int(human_us_1), int(nat_us_2), int(coh_us_2), int(diver_us_2), int(human_us_2)]

        for key, val in zip(labels, values):
            #if key not in scores_dict:
            #    scores_dict[key] = None
            scores_dict[key] = val
    except Exception as error:
        print(f"Error parsing response: {error}")
        input()
        return 0, 0, 0, 0, 0, 0, 0, 0


def compute_scores(base_dir, score_model_name):
    results = {}
    hfwrapper = None if "meta-llama" in score_model_name or "Qwen" in score_model_name else None

    for model in tqdm(os.listdir(base_dir), desc="Scoring dialogues"):
        model_path = os.path.join(base_dir, model)
        if not os.path.isdir(model_path):
            continue

        for game in os.listdir(model_path):
            game_path = os.path.join(model_path, game)
            results.setdefault(game, {}).setdefault(model, {})

            for exp in os.listdir(game_path):
                exp_path = os.path.join(game_path, exp)
                if not os.path.isdir(exp_path):
                    continue

                results[game][model][exp] = {}
                num_episodes = 0
                tot_hs_gt = {}
                tot_hs_gen = {}

                for episode in os.listdir(exp_path):
                    if episode.endswith(".json"):
                        continue

                    episode_path = os.path.join(exp_path, episode)
                    required_files = {"gt_corpus_dialogue.txt", "gen_cleaned_dialogue.txt", "interactions.json", "instance.json"}

                    if not required_files.issubset(set(os.listdir(episode_path))):
                        results[game][model][exp][episode] = {}
                        continue

                    num_episodes += 1
                    dialogue1_data = file_utils.load_file("gt_corpus_dialogue.txt", episode_path)
                    dialogue2_data = file_utils.load_file("gen_cleaned_dialogue.txt", episode_path)
                    interaction_data = file_utils.load_json("interactions.json", episode_path)
                    instance_data = file_utils.load_json("instance.json", episode_path)

                    dialogue1_partial = getpartialdialogue(dialogue1_data)
                    dialogue2_partial = getpartialdialogue(dialogue2_data)

                    results[game][model][exp][episode] = {
                        "n_turns": interaction_data["Evaluation"]["n_turns"],
                        "play_turns": interaction_data["Evaluation"]["play_turns"],
                        "gtdialogue": dialogue1_data,
                        "gendialogue": dialogue2_data,
                        "filename": interaction_data["Evaluation"].get("filename", instance_data["data"]["filename"]),
                        "human-style-score": {}
                    }

                    domains = "_".join(interaction_data["Evaluation"]["domains"])
                    results[game][model][exp][episode]["domains"] = domains
                    tot_hs_gt.setdefault(domains, {'full': 0, 'partial': 0})
                    tot_hs_gen.setdefault(domains, {'full': 0, 'partial': 0})

                    file_dir_path = "/".join(base_dir.split("/")[:-2]) + "/clemtod"

                    for dialogue_style, (dl1, dl2) in zip(["full", "partial"], [(dialogue1_data, dialogue2_data), (dialogue1_partial, dialogue2_partial)]):
                        promptmessage_gen = prepareprompt(file_dir_path, dl1, dl2)
                        llm_score = getscore(hfwrapper, promptmessage_gen, score_model_name)

                        scores = {}
                        parse_resp_score(llm_score, scores)

                        results[game][model][exp][episode]["human-style-score"][dialogue_style] = {
                            'gt': {
                                'naturalness': scores["nat_us_1"],
                                'coherence': scores["coh_us_1"],
                                'diversity': scores["diver_us_1"],
                                'human-like': scores["human_like_1"]
                            },
                            'gen': {
                                'naturalness': scores["nat_us_2"],
                                'coherence': scores["coh_us_2"],
                                'diversity': scores["diver_us_2"],
                                'human-like': scores["human_like_2"]
                            }
                        }

                        tot_hs_gt[domains][dialogue_style] += scores["human_like_1"]
                        tot_hs_gen[domains][dialogue_style] += scores["human_like_2"]

                    if num_episodes == 2:
                        break

                results[game][model][exp]['num_episodes'] = num_episodes
                results[game][model][exp]['human-style'] = compute_totals(tot_hs_gt, tot_hs_gen, num_episodes)

    save_results(base_dir, results, score_model_name)


def compute_totals(tot_hs_gt, tot_hs_gen, num_episodes):
    human_style = {}
    for domain in tot_hs_gt:
        human_style[domain] = {}
        for dialogue_style in tot_hs_gt[domain]:
            human_style[domain][dialogue_style] = {
                "gt": tot_hs_gt[domain][dialogue_style],
                "gen": tot_hs_gen[domain][dialogue_style]
            }

    total_count = {}
    for domain in human_style:
        for dialogue_style in human_style[domain]:
            total_count.setdefault(dialogue_style, {'gt': 0, 'gen': 0})
            total_count[dialogue_style]['gt'] += human_style[domain][dialogue_style]['gt']
            total_count[dialogue_style]['gen'] += human_style[domain][dialogue_style]['gen']
            total_count[dialogue_style]['gt_avg'] = round(total_count[dialogue_style]['gt'] / num_episodes, 2)
            total_count[dialogue_style]['gen_avg'] = round(total_count[dialogue_style]['gen'] / num_episodes, 2)

    human_style['total'] = total_count
    return human_style


def save_results(base_dir, results, score_model_name):
    suffix = "llama" if "meta-llama/llama-3.3-70b-instruct" in score_model_name else "gpt" if "gpt-4o-2024-08-06" in score_model_name else None
    if not suffix:
        raise ValueError(f"Model name {score_model_name} not supported")

    output_file = os.path.join(base_dir, f"dialoguemetric_human_style{suffix}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Scores computed and saved to {output_file}")

base_dir = "/home/admin/Desktop/codebase/cocobots/todsystems/clembench/modprog_single_2/"
compute_scores(base_dir, "gpt-4o-2024-08-06")
#compute_scores(base_dir, "meta-llama/llama-3.3-70b-instruct")