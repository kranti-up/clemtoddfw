import os
import numpy as np
import json

from utils import processgenslots, preparegenslots, checktimecloseness


def _save_episode_dialogue(generated_dialogue, episode_path):
    with open(os.path.join(episode_path, "dialogue.json"), "w", encoding="utf-8") as f:
        json.dump(generated_dialogue, f, ensure_ascii=False, indent=4)

def getslotvaluesbycategories(slots: dict):
        infoslots = {}
        attrslots = {}
        bookslots = {}
        info_fail_slots = {}
        book_fail_slots = {}

        for domain, dvalue in slots.items():
            for key, kvalue in dvalue.items():
                if key == "info":
                    if domain not in infoslots:
                        infoslots[domain] = {}

                    for k, v in kvalue.items():
                        infoslots[domain][k] = v

                if key == "fail_info":
                    if domain not in info_fail_slots:
                        info_fail_slots[domain] = {}

                    for k, v in kvalue.items():
                        info_fail_slots[domain][k] = v                        

                elif key == "book":
                    if domain not in bookslots:
                        bookslots[domain] = {}

                    for k, v in kvalue.items():
                        if k in ["invalid", "pre_invalid"]:
                            continue
                        bookslots[domain][f"book{k}"] = v

                elif key == "fail_book":
                    if domain not in book_fail_slots:
                        book_fail_slots[domain] = {}

                    for k, v in kvalue.items():
                        if k in ["invalid", "pre_invalid"]:
                            continue
                        book_fail_slots[domain][f"book{k}"] = v                        

                elif key in ["reqt", "req"]:
                    if domain not in attrslots:
                        attrslots[domain] = {}
                    
                    attrslots[domain] = kvalue
                else:
                    continue

        return infoslots, bookslots, attrslots, info_fail_slots, book_fail_slots

def _setto_lower(slots: dict) -> dict:
    slots_conv = {}
    for domain, dvalue in slots.items():
        if isinstance(dvalue, dict):
            for key, value in dvalue.items():
                if domain.lower() not in slots_conv :
                    slots_conv[domain.lower()] = {}
                if isinstance(value, dict) or isinstance(value, list):
                    print(f"Value is a dict or list {value}")
                    input()
                slots_conv[domain.lower()][key.lower()] = str(value).lower()
        elif isinstance(dvalue, list):
            if domain.lower() not in slots_conv :
                slots_conv[domain.lower()] = {}
            
            slots_conv[domain.lower()] = [str(val).lower() for val in dvalue]
        else:
            print(f"Value is a scalar {dvalue}")
            input()
    return slots_conv



    #return {
    #        str(domain).lower(): {str(key).lower(): str(value).lower() for key, value in dvalue.items()}
    #        for domain, dvalue in slots.items()
    #    }


def _compare_slots(gt_slots: dict, gt_fail_slots: dict, gen_slots: dict):

    if not gt_slots:
        return False, "Ground truth slots are empty"
    
    if not gen_slots:
        return False, "Generated slots are empty"

    gtcompslots = _setto_lower(gt_slots)
    gtfailcompslots = _setto_lower(gt_fail_slots)
    gencompslots = _setto_lower(gen_slots)



    missed_domains = [domain for domain in gtcompslots if domain not in gencompslots]
    if missed_domains:
        #print(f"Domains of the ground truth slots and generated do not match {missed_domains}")
        return False, missed_domains

    missed_values = []
    for domain, dvalue in gtcompslots.items():
        missed_keys = [key for key in dvalue if key not in gencompslots[domain]]
        if missed_keys:
            #print(f"Keys of the ground truth slots and generated slots do not match {missed_keys}")
            return False, [{domain:missed_keys}]
        try:
            mvalues = []
            for key, value in dvalue.items():
                if value != gencompslots[domain][key]:
                    if domain in gtfailcompslots and key in gtfailcompslots[domain]:
                        if gtfailcompslots[domain][key] != gencompslots[domain][key]:
                            data_match = False
                            #if key in ["leaveat", "arriveby"]:
                            #    data_match = True#checktimecloseness(gtfailcompslots[domain][key], gencompslots[domain][key])

                            if key == "leaveat":
                                #data_match = True
                                if value < gencompslots[domain][key]:
                                    data_match = True
                            elif key == "arriveby":
                                if value > gencompslots[domain][key]:
                                    data_match = True
                                #data_match = True

                            if not data_match:
                                mvalues.append({domain: {"gt": {key: gtfailcompslots[domain][key]}, "gen": {key: gencompslots[domain][key]}}})
                        else:
                            #print(f"Key {key} has the same value {gencompslots[domain][key]} in the ground truth fail data and generated slots")
                            pass
                    else:
                        data_match = False
                        if key == "leaveat":
                            #print(value, gencompslots[domain][key])
                            if value < gencompslots[domain][key]:
                                data_match = True
                            #data_match = True
                        elif key == "arriveby":
                            if value > gencompslots[domain][key]:
                                data_match = True
                            #data_match = True
                        

                        #if key in ["leaveat", "arriveby"]:
                        #    data_match = True#checktimecloseness(value, gencompslots[domain][key])
                        
                        if not data_match:
                            mvalues.append({domain: {"gt": {key: value}, "gen": {key: gencompslots[domain][key]}}})
            if mvalues:
                missed_values.append({domain: mvalues})
        except Exception as error:
            print(f"Error in comparing values {error}")
            print(gtcompslots)
            print(gtfailcompslots)
            print(gencompslots)
            input()

    if missed_values:
        #print(f"Values of the ground truth slots and generated slots do not match {missed_values}")
        return False, missed_values                      
    
    return True, None


def compute_scores(base_dir):
    results = {}

    for model in os.listdir(base_dir):
        #Check if model is a directory
        if model == "corpus_dialogues" or not os.path.isdir(os.path.join(base_dir, model)):
            continue
        #if "70B" not in model:
        #    continue
        model_path = os.path.join(base_dir, model)
        for game in os.listdir(model_path):
            if game not in results:
                results[game] = {}
            if model not in results[game]:
                results[game][model] = {}
            game_path = os.path.join(model_path, game)
            for exp in os.listdir(game_path):
                #if "xu" not in exp:
                #    continue
                if exp not in results[game][model]:
                    results[game][model][exp] = {}
                exp_path = os.path.join(game_path, exp)

                if not os.path.isdir(exp_path):
                    continue

                num_episodes = {}
                inform_ep_list = {}
                book_ep_list = {}
                attr_ep_list = {}
                game_abort_list = {}
                game_loss_list = {}

                for episode in os.listdir(exp_path):
                    episode_path = os.path.join(exp_path, episode)
                    if not os.path.isdir(episode_path):
                        continue

                    for filename in os.listdir(episode_path):
                        if filename != "interactions.json":
                            continue

                        with open(os.path.join(episode_path, filename), "r") as f:
                            interaction_data = json.load(f)

                        _save_episode_dialogue(interaction_data["Evaluation"]["gendialogue"], episode_path)

                        game_abort = int(interaction_data["Aborted"])
                        game_loss = int(interaction_data["Lose"])

                        game_evaldata = interaction_data["Evaluation"]
                        dialogue_type = game_evaldata["dialogue_type"]
                        domains = game_evaldata["domains"]
                        domain_data = "_".join(domains)
                        tsystem = game_evaldata["tsystem"]
                        play_turns = game_evaldata["play_turns"]
                        n_turns = game_evaldata["n_turns"]
                        corpususer = game_evaldata["corpususer"]
                        gt_slots = game_evaldata["slots_gt"]
                        gen_slots = game_evaldata["slots_gen"]

                        gen_slots_processed = (
                            preparegenslots(gen_slots)
                            if gen_slots and ("xu" in exp or "he" in exp)
                            else processgenslots(gen_slots)
                            if gen_slots
                            else {}
                        )

                        gen_slots_loss = game_evaldata.get("slots_gen_loss", {})

                        if game_abort or play_turns == n_turns or gen_slots_processed is None:
                            inform_episode = 0
                            book_episode = 0
                            attr_episode = 0
                            game_abort = 1 if game_abort else 0
                        else:
                            infoslots_gt, bookslots_gt, attrslots_gt, infofailslots_gt, bookfailslots_gt = getslotvaluesbycategories(gt_slots)
                            infoslots_gen, bookslots_gen, attrslots_gen, *_ = getslotvaluesbycategories(gen_slots_processed)

                            inform_episode, book_episode, attr_episode = 0, 0, 0
                            status, _ = _compare_slots(infoslots_gt, infofailslots_gt, infoslots_gen)
                            if status:
                                inform_episode = 1
                                if not bookslots_gt:
                                    book_episode = 1
                                else:
                                    status, _ = _compare_slots(bookslots_gt, bookfailslots_gt, bookslots_gen)
                                    book_episode = 1 if status else 0
                                    game_loss = 0 if status else 1

                                if attrslots_gt:
                                    status, _ = _compare_slots(attrslots_gt, {}, attrslots_gen)
                                    attr_episode = 1 if status else 0

                                else:
                                    attr_episode = 1


                            else:
                                game_loss = 1
                            '''
                            if attrslots_gt:
                                status, _ = _compare_slots(attrslots_gt, attrslots_gen)
                                attr_episode = 1 if status else 0
                            '''

                        num_episodes.setdefault(domain_data, 0)
                        num_episodes[domain_data] += 1

                        for metric, val in zip(
                            [inform_ep_list, book_ep_list, attr_ep_list, game_abort_list, game_loss_list],
                            [inform_episode, book_episode, attr_episode, game_abort, game_loss],
                        ):
                            metric.setdefault(domain_data, []).append(val)
                        break

                results[game][model][exp]["num_episodes"] = num_episodes
                results[game][model][exp]["num_episodes"]["total"] = sum(num_episodes.values())

                metrics = {
                    "entity_ext": inform_ep_list,
                    "task_success": book_ep_list,
                    "attr": attr_ep_list,
                    "game_abort": game_abort_list,
                    "game_loss": game_loss_list,
                }

                counts = {
                    "game_abort_count": game_abort_list,
                    "game_loss_count": game_loss_list,
                    "inform_ep_count": inform_ep_list,
                    "book_ep_count": book_ep_list,
                    "attr_ep_count": attr_ep_list,
                }

                for metric_name, val_list in metrics.items():
                    metrics[metric_name] = {
                        domain: round(np.mean(values), 2) if values else 0
                        for domain, values in val_list.items()
                    }
                    values = list(metrics[metric_name].values())
                    metrics[metric_name]["average"] = round(sum(values) / len(values), 2) if values else 0

                for count_name, val_list in counts.items():
                    counts[count_name] = {
                        domain: sum(1 for val in values if val == 1)
                        for domain, values in val_list.items()
                    }
                    counts[count_name]["total"] = sum(counts[count_name].values())

                results[game][model][exp] = {**results[game][model][exp], **metrics, **counts}

    with open(os.path.join(base_dir, "taskmetrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("Task metrics computed and saved to taskmetrics.json")


compute_scores(
    "/home/users/kranti/project/kranti/testtodsystem/hetod/clembench/cross_hetod_single_1/"
)