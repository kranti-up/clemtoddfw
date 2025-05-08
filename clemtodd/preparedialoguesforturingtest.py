import os
import random
import json


SEED = 42


def preparedialogues(base_dir, dlg_dirs=None):
    evaldialogues = {}

    episode_list = [f"episode_{i}" for i in range(1, 60)]
    random.seed(SEED)
    random_episode_list = random.sample(episode_list, 50)

    print(f"Randomly selected episodes: {random_episode_list}")

    for dlg_dir in dlg_dirs:
        task_type = "single" if "single" in dlg_dir else "multi"
        dir_path = os.path.join(base_dir, dlg_dir)
        if not os.path.isdir(dir_path):
            print(f"Directory {dir_path} does not exist.")
            continue
        for model in os.listdir(dir_path):
            #Check if model is a directory
            if model == "corpus_dialogues" or not os.path.isdir(os.path.join(dir_path, model)):
                continue
            #if model not in ["gpt-4o-2024-08-06-t0.0--gpt-4o-2024-08-06-t0.0",
            #if model not in ["Qwen2.5-32B-Instruct-t0.0--Qwen2.5-32B-Instruct-t0.0",
            #                 "Llama-3.3-70B-Instruct-t0.0--Qwen2.5-32B-Instruct-t0.0",
            #                 ]:
            if model not in ["Qwen2.5-32B-Instruct-t0.0--Qwen2.5-32B-Instruct-t0.0",
                             "Llama-3.3-70B-Instruct-t0.0--Qwen2.5-32B-Instruct-t0.0",
                             ]:

                print(f"Skipping model {model} as it is not in the specified list.")
                continue
            model_path = os.path.join(dir_path, model)
            num_episodes = 0
            for game in os.listdir(model_path):
                game_path = os.path.join(model_path, game)
                for exp in os.listdir(game_path):
                    exp_path = os.path.join(game_path, exp)

                    if not os.path.isdir(exp_path):
                        continue

                    for episode in os.listdir(exp_path):
                        episode_path = os.path.join(exp_path, episode)
                        if not os.path.isdir(episode_path) or episode not in random_episode_list or num_episodes == 25:
                            continue

                        #print('episode_path', episode_path)
                        with open(f"{episode_path}/interactions.json", "r") as f:
                            interaction_data = json.load(f)

                        game_abort = interaction_data["Aborted"]
                        game_loss = interaction_data["Lose"]

                        if game_abort or game_loss:
                            continue


                        with open(f"{episode_path}/gen_cleaned_dialogue.txt", "r") as f:
                            gen_dialogue = f.read()

                        add_closure = ["\n\nUser:\nGreat, thank you so much.\nSystem:\nYou are very welcome!",
                                       "\n\nUser:\nThats all.\nSystem:\nYou are very welcome!",
                                       "\n\nUser:\nThats all.\nSystem:\nHave a good day!",
                                       "\n\nUser:\nThanks for your help.\nSystem:\nYou're welcome",
                                       "\n\nUser:\nNo that's it all.\nSystem:\nWonderful. Glad to help.",]

                        add_message_choice = random.choice(add_closure)
                        gen_dialogue += add_message_choice

                        #gt_episode_path = f"{dir_path}/gpt-4o-2024-08-06-t0.0--gpt-4o-2024-08-06-t0.0/{game}/{exp}/{episode}"
                        #with open(f"{gt_episode_path}/gt_corpus_dialogue.txt", "r") as f:
                        with open(f"{episode_path}/gt_corpus_dialogue.txt", "r") as f:
                            gt_dialogue = f.read()

                        # Randomize left/right placement
                        #random.seed(SEED)
                        side_choice = random.choice([True, False])
                        if side_choice:
                            left, right = gen_dialogue, gt_dialogue
                        else:
                            left, right = gt_dialogue, gen_dialogue

                        episode_name = f"{model.split('--')[0]}-{episode}-{task_type}"

                        evaldialogues[episode_name] = {
                            "left": left,
                            "right": right,
                            "side_choice": "left" if side_choice else "right",
                            "gen_dialogue": "left" if side_choice else "right",
                            "gt_dialogue": "right" if side_choice else "left",
                            "task_type": task_type,
                            "episode": episode,
                            "play_turns": interaction_data["Evaluation"]["play_turns"],
                            "domains": "_".join(interaction_data["Evaluation"]["domains"]),
                            "model": model,
                        }
                        num_episodes += 1
                        #print(f"Processed {episode_path}")
        print(f"Processed {dlg_dir} - {num_episodes} episodes")

    with open(f"{base_dir}/{dlg_dirs[0]}/eval_dialogues_llama.json", "w") as f:
        json.dump(evaldialogues, f, indent=4)
    print(f"Prepared dialogues (len - {len(evaldialogues)}) saved to eval_dialogues.json")

if __name__ == "__main__":
    base_dir = "/home/users/kranti/project/kranti/testtodsystem/monollm/clembench/"
    #dlg_dirs = ["mono_single_2", "mono_multi_1"]
    dlg_dirs = ["cross_mono_single_2", "cross_mono_multi_1"]
    preparedialogues(base_dir, dlg_dirs)
