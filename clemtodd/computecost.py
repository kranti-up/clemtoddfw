import os
import json
import pandas as pd

API_PRICE = {
    'gpt-4o': {
        'input': 0.0025 / 1000,
        'cached': 0.00125 / 1000,
        'output': 0.01 / 1000,
        'cost_per_petaflop': 0.05,
        'model_params_billions': 175,
    },
    'gpt-4o-2024-11-20': {
        'input': 0.0025 / 1000,
        'cached': 0.00125 / 1000,
        'output': 0.01 / 1000,
        'cost_per_petaflop': 0.05,
        'model_params_billions': 175,     
    },
    'gpt-4o-2024-08-06': {
        'input': 0.0025 / 1000,
        'cached': 0.00125 / 1000,
        'output': 0.01 / 1000,
        'cost_per_petaflop': 0.05,
        'model_params_billions': 175,  
    },
    'gpt-4o-mini': {
        'input': 0.00015 / 1000,
        'cached': 0.000075 / 1000,
        'output': 0.0006 / 1000,
        'cost_per_petaflop': 0.05,
        'model_params_billions': 1,       
    },    
    'gpt-4o-mini-2024-07-18': {
        'input': 0.00015 / 1000,
        'cached': 0.000075 / 1000,
        'output': 0.0006 / 1000,
        'cost_per_petaflop': 0.05,
        'model_params_billions': 1,    
    },      
    'deepseek-chat': {
        'input': 0.0025 / 1000,
        'cached': 0.00125 / 1000,
        'output': 0.0011 / 1000,
        'cost_per_petaflop': 0.05,  
        'model_params_billions': 7,     
    },
    'Llama-3.3-70B-Instruct': {
        'input': 0.00012 / 1000,
        'cached': 0.0,
        'output': 0.0003 / 1000,
        'cost_per_petaflop': 0.05,
        'model_params_billions': 70,      
    },
    'Meta-Llama-3.1-8B-Instruct': {
        'input': 0.00002 / 1000,
        'cached': 0.0,
        'output': 0.00005 / 1000,
        'cost_per_petaflop': 0.05,
        'model_params_billions': 8,    
    },    
    'Llama-3.2-1B-Instruct': {
        'input': 0.00001 / 1000,
        'cached': 0.0,
        'output': 0.00001 / 1000,
        'cost_per_petaflop': 0.05,
        'model_params_billions': 1,    
    },  
    'Llama-3.2-3B-Instruct': {
        'input': 0.000015 / 1000,
        'cached': 0.0,
        'output': 0.000025 / 1000,
        'cost_per_petaflop': 0.05,
        'model_params_billions': 3,    
    },
    'Qwen2.5-7B-Instruct': {
        'input': 0.00005 / 1000,
        'cached': 0.0,
        'output': 0.0001 / 1000,
        'cost_per_petaflop': 0.05,
        'model_params_billions': 7,    
    },
    'Qwen2.5-32B-Instruct': {
        'input': 0.00079 / 1000,
        'cached': 0.0,
        'output': 0.00079 / 1000,
        'cost_per_petaflop': 0.05,
        'model_params_billions': 32,    
    },        
}


def calc_openai_cost(model, usage):
    if model in API_PRICE:
        price = API_PRICE[model]

        prompt_tokens = usage['prompt_tokens']
        if "prompt_tokens_details" in usage and usage["prompt_tokens_details"]:
            cached_tokens = usage["prompt_tokens_details"]["cached_tokens"]
            if cached_tokens:
                prompt_tokens -= cached_tokens
        else:
            cached_tokens = 0

        cost = prompt_tokens * price['input'] + cached_tokens * price['cached'] + usage['completion_tokens'] * price['output']
    else:
        raise ValueError(f'{model = }')
    return cost, {"prompt_tokens": prompt_tokens, "cached_tokens": cached_tokens, "completion_tokens": usage['completion_tokens']}

def approximate_token_count(text):
    return int(len(text) / 4) # Rough estimate of tokens based on average token length


def get_flops_per_token(model_name):
    model_params = API_PRICE[model_name]['model_params_billions'] * 1e9
    flops_per_token = 2 * model_params

    return flops_per_token


def compute_tokens_local_style(requests):
    token_results = []


    for idx, interaction in enumerate(requests):
        manip_prompt_obj = interaction["manipulated_prompt_obj"]
        if isinstance(manip_prompt_obj, list):
            manip_prompt_obj = manip_prompt_obj[0]
        #prompt = interaction.get('manipulated_prompt_obj', {}).get('inputs', '')
        prompt = manip_prompt_obj.get('inputs', '')
        response = interaction.get('raw_response_obj', {}).get('response', '')
        model_name = interaction.get('raw_response_obj', {}).get('clem_player', {}).get('model_name', 'Unknown')

        flops_per_token = get_flops_per_token(model_name)        
        
        prompt_tokens = approximate_token_count(prompt)
        response_tokens = approximate_token_count(response)
        total_tokens = prompt_tokens + response_tokens
        
        token_cost = (prompt_tokens * API_PRICE[model_name]["input"]) + (response_tokens * API_PRICE[model_name]["output"])
        
        # Calculate FLOPs
        input_flops = prompt_tokens * flops_per_token
        output_flops = response_tokens * flops_per_token
        total_flops = input_flops + output_flops
        petaflops = total_flops / 1e15
        flops_cost = petaflops * API_PRICE[model_name]["cost_per_petaflop"]
        
        token_results.append({
            'interaction_id': idx + 1,
            'model': model_name,
            'prompt_tokens': prompt_tokens,
            'response_tokens': response_tokens,
            'total_tokens': total_tokens,
            'petaflops': petaflops,
            'token_cost': token_cost,
            'flops_cost': flops_cost
        })

    token_df = pd.DataFrame(token_results)
    
    return token_df



def compute_tokens_api_style(interactions_data, model_name):
    token_results = []

    turn_data = interactions_data.get('turns', [])
    if not turn_data:
        raise ValueError("No turn data found in the interactions.")

    model_name_1 = interactions_data.get('players', {}).get('Player 1', 'Unknown')
    model_name_1 = model_name_1.split(":")[1].strip()
    model_name_2 = interactions_data.get('players', {}).get('Player 2', 'Unknown')
    model_name_2 = model_name_2.split(":")[1].strip()


    for turn in turn_data:
        for idx, interaction in enumerate(turn):
            from_ = interaction.get('from', '')
            to_ = interaction.get('to', '')
            type_ = interaction.get('action', {}).get('type', '')
            content = interaction.get('action', {}).get('content', '')

            prompt = None
            response = None

            if from_ == "GM" and to_ in ["Player 1", "Player 2"] and type_ == "send message":
                prompt = content
            elif from_ in ["Player 1", "Player 2"] and to_ == "GM" and type_ == "get message":
                response = content
            elif from_ == "Player 2" and to_ == "Player 2":
                if type_ == "info":
                    prompt = content
                elif type_ == "get message":
                    response = content
            else:
                print(f"Skipping interaction {idx + 1} due to unexpected format. Players: {from_, to_}, Action Type: {type_}")
                continue

            if to_ == "Player 1":
                model_name = model_name_1
            elif to_ == "Player 2":
                model_name = model_name_2

            flops_per_token = get_flops_per_token(model_name)        
            
            prompt_tokens = 0
            response_tokens = 0
            if prompt:
                prompt_tokens = approximate_token_count(prompt)
            if response:
                response_tokens = approximate_token_count(response)
            total_tokens = prompt_tokens + response_tokens
            
            token_cost = (prompt_tokens * API_PRICE[model_name]["input"]) + (response_tokens * API_PRICE[model_name]["output"])
            
            # Calculate FLOPs for simple mode (no context consideration)
            input_flops = prompt_tokens * flops_per_token
            output_flops = response_tokens * flops_per_token
            total_flops = input_flops + output_flops
            petaflops = total_flops / 1e15
            flops_cost = petaflops * API_PRICE[model_name]["cost_per_petaflop"]
            
            token_results.append({
                'interaction_id': idx + 1,
                'model': model_name,
                'prompt_tokens': prompt_tokens,
                'response_tokens': response_tokens,
                'total_tokens': total_tokens,
                'petaflops': petaflops,
                'token_cost': token_cost,
                'flops_cost': flops_cost
            })

    token_df = pd.DataFrame(token_results)
    
    return token_df

def get_cost(token_df):
    """
    Calculate the cost based on the token DataFrame
    """
    token_cost = round((token_df['token_cost'].sum()), 6)
    flops_cost = round((token_df['flops_cost'].sum()), 6)
    total_tokens = token_df['total_tokens'].sum()
    total_petaflops = token_df['petaflops'].sum()
    
    # Calculate cost per 1K tokens
    token_cost_per_1k_tokens = round(((token_cost / total_tokens) * 1000), 6)
    flops_cost_per_1k_tokens = round(((flops_cost / total_tokens) * 1000), 6)
    
    return total_petaflops, token_cost, flops_cost, token_cost_per_1k_tokens, flops_cost_per_1k_tokens

def display_cost_summary(token_df):
    """
    Display a summary of the cost calculations
    """
    # API-style summary
    api_total = token_df['token_cost'].sum()
    total_prompt_tokens = token_df['prompt_tokens'].sum()
    total_response_tokens = token_df['response_tokens'].sum()
    total_tokens = token_df['total_tokens'].sum()
    
    # FLOPs-based summary
    flops_total = token_df['flops_cost'].sum()
    total_petaflops = token_df['petaflops'].sum()
    
    print("=== COST ANALYSIS SUMMARY ===")
    print(f"\nTotal interactions: {len(token_df)}")
    print(f"Model(s): {', '.join(token_df['model'].unique())}")
    
    print("\n--- Token Statistics ---")
    print(f"Total input tokens: {total_prompt_tokens:,}")
    print(f"Total output tokens: {total_response_tokens:,}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Input/Output ratio: {total_prompt_tokens/total_response_tokens:.2f}:1")
    
    print("\n--- API-Style Cost (Token-Based) ---")
    print(f"Total cost: ${api_total:.6f}")
    print(f"Cost per 1K tokens: ${(api_total/total_tokens*1000):.6f}")
    
    print("\n--- FLOPs-Based Cost (Computational) ---")
    print(f"Total computational work: {total_petaflops:.6e} petaFLOPs")
    print(f"Total cost: ${flops_total:.6f}")
    print(f"Cost per 1K tokens: ${(flops_total/total_tokens*1000):.6f}")
    
    print("\n--- Comparison ---")
    print(f"FLOPs-to-Token cost ratio: {flops_total/api_total:.2f}:1")
    print(f"Cost difference: ${abs(api_total - flops_total):.6f}")
    if api_total > flops_total:
        print(f"Token-based pricing is {api_total/flops_total:.2f}x more expensive than FLOPs-based")
    else:
        print(f"FLOPs-based pricing is {flops_total/api_total:.2f}x more expensive than token-based")


def compute_cost(base_dir):
    results = {}    

    for model in os.listdir(base_dir):
        if model == "corpus_dialogues" or not os.path.isdir(os.path.join(base_dir, model)):# or model == "gpt-4o-2024-08-06-t0.0--gpt-4o-2024-08-06-t0.0":
            continue
        model_path = os.path.join(base_dir, model)
        for game in os.listdir(model_path):
            if game not in results:
                results[game] = {}
            if model not in results[game]:
                results[game][model] = {}
            game_path = os.path.join(model_path, game)
            for exp in os.listdir(game_path):
                exp_path = os.path.join(game_path, exp)
                if not os.path.isdir(exp_path):
                    continue

                if exp not in results[game][model]:
                    results[game][model][exp] = 0.0

                episode_costs = []
                episode_tokens = {}
                num_episodes = 0
                for episode in os.listdir(exp_path):
                    if episode.endswith(".json"):
                        continue
                    num_episodes += 1
                    episode_path = os.path.join(exp_path, episode)
                    for filename in os.listdir(episode_path):
                        if not filename in ["requests.json", "interactions.json"]:
                            continue

                        if filename == "interactions.json":
                            with open(os.path.join(episode_path, filename), "r") as f:
                                interaction_data = json.load(f)
                            token_df_api = compute_tokens_api_style(interaction_data, model)
                            total_petaflops, token_cost_api, flops_cost_api, token_cost_per_1k_tokens_api, flops_cost_per_1k_tokens_api = get_cost(token_df_api)
                            episode_costs.append({'api_cost': {'token_cost': token_cost_api, 'flops_cost': flops_cost_api,
                                                               'total_petaflops': total_petaflops,
                                                                'token_cost_per_1k_tokens': token_cost_per_1k_tokens_api,
                                                                'flops_cost_per_1k_tokens': flops_cost_per_1k_tokens_api}})
                        elif filename == "requests.json":
                            with open(os.path.join(episode_path, filename), "r") as f:
                                requests_data = json.load(f)
                            print(f"Model: {model}, Game: {game}, Episode: {episode}")
                            token_df_full = compute_tokens_local_style(requests_data)
                            total_petaflops, token_cost_full, flops_cost_full, token_cost_per_1k_tokens_full, flops_cost_per_1k_tokens_full = get_cost(token_df_full)
                            episode_costs.append({'full_cost': {'token_cost': token_cost_full, 'flops_cost': flops_cost_full, 
                                                               'total_petaflops': total_petaflops,                                                                
                                                                'token_cost_per_1k_tokens': token_cost_per_1k_tokens_full,
                                                                'flops_cost_per_1k_tokens': flops_cost_per_1k_tokens_full}})

                # Calculate the total cost for the experiment
                token_cost_api = sum([cost['api_cost']['token_cost'] for cost in episode_costs if 'api_cost' in cost])
                token_cost_api_dialogue = round(token_cost_api/num_episodes, 5)
                token_cost_full = sum([cost['full_cost']['token_cost'] for cost in episode_costs if 'full_cost' in cost])
                token_cost_full_dialogue = round(token_cost_full/num_episodes, 5)
                flop_cost_api = sum([cost['api_cost']['flops_cost'] for cost in episode_costs if 'api_cost' in cost])
                flop_cost_api_dialogue = round(flop_cost_api/num_episodes, 5)
                flop_cost_full = sum([cost['full_cost']['flops_cost'] for cost in episode_costs if 'full_cost' in cost])
                flop_cost_full_dialogue = round(flop_cost_full/num_episodes, 5)
                petaflops_full = sum([cost['full_cost']['total_petaflops'] for cost in episode_costs if 'full_cost' in cost])
                petaflops_full_dialogue = round(petaflops_full/num_episodes, 2)
                # Store the results

                results[game][model][exp] = {"token_cost_api": round(token_cost_api, 5),
                                             "token_cost_api_dialogue": round(token_cost_api_dialogue, 5),
                                             "flop_cost_api": round(flop_cost_api, 5),
                                             "flop_cost_api_dialogue": round(flop_cost_api_dialogue, 5),                       
                                             "token_cost_full": round(token_cost_full, 5),
                                             "token_cost_full_dialogue": round(token_cost_full_dialogue, 5),
                                             "flop_cost_full": round(flop_cost_full, 5),
                                             "flop_cost_full_dialogue": round(flop_cost_full_dialogue, 5),
                                             "total_petaflops": round(petaflops_full, 5),
                                             "total_petaflops_dialogue": round(petaflops_full_dialogue, 5),
                                             "num_episodes": num_episodes,
                                              #"episodes": episode_costs
                                            }
                #results[game][model][exp]["cost"] = round(sum(episode_costs), 2)




    for game in results:
        overall_api_cost = 0.0
        overall_flop_cost = 0.0
        overall_api_tokens_count = 0.0
        overall_flop_tokens_count = 0.0
        overall_petaflops_count = 0
        for model in results[game]:
            for exp in results[game][model]:
                overall_api_cost += results[game][model][exp]["token_cost_api"]
                overall_api_tokens_count += results[game][model][exp]["token_cost_full"]
                overall_flop_cost += results[game][model][exp]["flop_cost_api"]
                overall_flop_tokens_count += results[game][model][exp]["flop_cost_full"]
                overall_petaflops_count += results[game][model][exp]["total_petaflops"]
            '''
            results[game][model]["overall"] = {
                "token_cost_api": round(overall_api_cost, 2),
                "flop_cost_api": round(overall_flop_cost, 2),
                "token_cost_full": round(overall_api_tokens_count, 2),
                "flop_cost_full": round(overall_flop_tokens_count, 2),
                "total_petaflops": round(overall_petaflops_count, 5)
            }
            '''
        results[game]["overall"] = {
            "token_cost_api": round(overall_api_cost, 5),
            "flop_cost_api": round(overall_flop_cost, 5),
            "token_cost_full": round(overall_api_tokens_count, 5),
            "flop_cost_full": round(overall_flop_tokens_count, 5),
            "total_petaflops": round(overall_petaflops_count, 5)
        }
        print(f"Game: {game}")
        print(f"Overall API Cost: ${results[game]['overall']['token_cost_api']}")
        print(f"Overall FLOP Cost: ${results[game]['overall']['flop_cost_api']}")

    with open(os.path.join(base_dir, "costs.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("Results saved to costs.json")



if __name__ == '__main__':
    compute_cost("/home/users/kranti/project/kranti/testtodsystem/modllm/clembench/modllm_single_2/")
