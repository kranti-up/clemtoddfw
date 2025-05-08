import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.ticker import MaxNLocator

# Data from the table (manually extracted)
models = ['Llama-3.2-1B', 'Llama-3.2-3B', 'Llama-3.1-8B', 'Qwen2.5-7B', 'Qwen2.5-32B', 'Llama-3.3-70B']

# Metrics: Inform, Success, Book across 3 architectures
data_style1 = {
    'Monolithic LLM': {
        'Entity Match': [0.0, 0.09, 0.33, 0.15, 0.62, 0.65],
        'Attributes': [0.0, 0.29, 0.60, 0.31, 0.98, 0.90],
        'Booking Success': [0.0, 0.07, 0.22, 0.10, 0.62, 0.65]
    },
    'Modular Prog': {
        'Entity Match': [0.0, 0.0, 0.0, 0.10, 0.65, None],
        'Attributes': [0.0, 0.03, 0.0, 0.28, 0.80, None],
        'Booking Success': [0.0, 0.0, 0.0, 0.10, 0.65, None]
    },
    'Modular LLM': {
        'Entity Match': [0.0, 0.0, 0.0, 0.0, 0.03, None],
        'Attributes': [0.0, 0.0, 0.0, 0.0, 0.05, None],
        'Booking Success': [0.0, 0.0, 0.0, 0.0, 0.03, None]
    }
}

model_params_billions = {
    'Llama-3.2-1B': 1,
    'Llama-3.2-3B': 3,
    'Llama-3.1-8B': 8,
    'Llama-3.3-70B': 70,
    'Qwen2.5-7B': 7,
    'Qwen2.5-32B': 32
}


data = {
    'Llama-3.2-1B': {'size': 1, 'monolithic_llm': {'Entity Match': 0.0, 'Attributes': 0.0, 'Booking Success': 0.0,
                                                   'token_cost': 0.00081, 'flop_cost': 0.00736, 'total_petaflops': 8.59},
                     'modular_prog': {'Entity Match': 0.0, 'Attributes': 0.0, 'Booking Success': 0.0,
                                      'token_cost': 0.00087, 'flop_cost': 0.00726, 'total_petaflops': 8.46519},
                     'modular_llm': {'Entity Match': 0.0, 'Attributes': 0.0, 'Booking Success': 0.0,
                                     'token_cost': 0.00081, 'flop_cost': 0.00715, 'total_petaflops': 8.33713},
    },
    'Llama-3.2-3B': {'size': 3, 'monolithic_llm': {'Entity Match': 0.08, 'Attributes': 0.15, 'Booking Success': 0.08,
                                                   'token_cost': 0.00734, 'flop_cost': 0.46588, 'total_petaflops': 541.909},
                     'modular_prog': {'Entity Match': 0.0, 'Attributes': 0.0, 'Booking Success': 0.0,
                                      'token_cost': 0.00737, 'flop_cost': 0.36224, 'total_petaflops': 422.8296},
                     'modular_llm': {'Entity Match': 0.03, 'Attributes': 0.03, 'Booking Success': 0.05,
                                     'token_cost': 0.0022, 'flop_cost': 0.10284, 'total_petaflops': 119.18},
    },

    'Qwen2.5-7B': {'size': 7, 'monolithic_llm': {'Entity Match': 0.10, 'Attributes': 0.12, 'Booking Success': 0.10,
                                                 'token_cost': 0.0957, 'flop_cost': 0.9123, 'total_petaflops': 105.23},
                     'modular_prog': {'Entity Match': 0.47, 'Attributes': 0.63, 'Booking Success': 0.47,
                                      'token_cost': 0.01023, 'flop_cost': 0.37401, 'total_petaflops': 435.56},
                     'modular_llm': {'Entity Match': 0.30, 'Attributes': 0.43, 'Booking Success': 0.30,
                                     'token_cost': 0.0032, 'flop_cost': 0.13686, 'total_petaflops':159.26},
    },

    'Llama-3.1-8B': {'size': 8, 'monolithic_llm': {'Entity Match': 0.32, 'Attributes': 0.42, 'Booking Success': 0.18,
                                                   'token_cost': 0.218, 'flop_cost': 0.442, 'total_petaflops': 515.57},
                     'modular_prog': {'Entity Match': 0.0, 'Attributes': 0.0, 'Booking Success': 0.0,
                                      'token_cost': 0.01112, 'flop_cost': 0.66753, 'total_petaflops': 780.96},
                     'modular_llm': {'Entity Match': 0.13, 'Attributes': 0.18, 'Booking Success': 0.10,
                                     'token_cost': 0.00238, 'flop_cost': 0.289, 'total_petaflops': 265.36},
    },

    'Qwen2.5-32B': {'size': 32, 'monolithic_llm': {'Entity Match': 0.95, 'Attributes': 0.98, 'Booking Success': 0.71,
                                                   'token_cost': 0.00664, 'flop_cost': 0.17, 'total_petaflops': 199},
                     'modular_prog': {'Entity Match': 0.58, 'Attributes': 0.70, 'Booking Success': 0.41,
                                      'token_cost': 0.05, 'flop_cost': 0.48, 'total_petaflops': 564.40},
                     'modular_llm': {'Entity Match': 0.82, 'Attributes': 0.82, 'Booking Success': 0.68,
                                     'token_cost': 0.0087, 'flop_cost': 0.258, 'total_petaflops': 298.82},
    },            

    'Llama-3.3-70B': {'size': 70, 'monolithic_llm': {'Entity Match': 0.83, 'Attributes': 0.92, 'Booking Success': 0.69,
                                                     'token_cost': 0.22, 'flop_cost': 0.50, 'total_petaflops': 581.51},
                     'modular_prog': {'Entity Match': 0.53, 'Attributes': 0.67, 'Booking Success': 0.43,
                                      'token_cost':0.02, 'flop_cost': 1.33, 'total_petaflops': 1544.74},
                     'modular_llm': {'Entity Match': 0.70, 'Attributes': 0.83, 'Booking Success': 0.52,
                                     'token_cost': 0.0069, 'flop_cost': 1.3575, 'total_petaflops': 1573.9},
    },

    'GPT-4o': {'size': np.nan, 'monolithic_llm': {'Entity Match': 0.72, 'Attributes': 0.95, 'Booking Success': 0.81,
                                                     'token_cost': 0.01983, 'flop_cost': np.nan, 'total_petaflops': np.nan},
                     'modular_prog': {'Entity Match': 0.85, 'Attributes': 0.93, 'Booking Success': 0.65,
                                      'token_cost':2.1134, 'flop_cost': np.nan, 'total_petaflops': np.nan},
                     'modular_llm': {'Entity Match': 0.87, 'Attributes': 0.98, 'Booking Success': 0.84,
                                     'token_cost': 0.031, 'flop_cost': np.nan, 'total_petaflops': np.nan},
    },    
}

data_cross_model_multi_us = {
    'Llama-3.2-1B': {'size': 1, 'monolithic_llm': {'Entity Match': 0.37, 'Attributes': 0.95, 'Booking Success': 0.30, 'token_cost': 0.05, 'flop_cost': 1.39},
                     'modular_prog': {'Entity Match': 0.40, 'Attributes': 0.97, 'Booking Success': 0.40, 'token_cost': 0.63, 'flop_cost': 3.6},
    },
    'Llama-3.2-3B': {'size': 3, 'monolithic_llm': {'Entity Match': 0.52, 'Attributes': 0.93, 'Booking Success': 0.48, 'token_cost': 0.07, 'flop_cost': 2.08},
                     'modular_prog': {'Entity Match': 0.57, 'Attributes': 0.90, 'Booking Success': 0.57, 'token_cost': 1.02, 'flop_cost': 5.35},
    },

    'Qwen2.5-7B': {'size': 7, 'monolithic_llm': {'Entity Match': 0.48, 'Attributes': 0.80, 'Booking Success': 0.37, 'token_cost': 0.08, 'flop_cost': 4.0},
                     'modular_prog': {'Entity Match': 0.32, 'Attributes': 0.55, 'Booking Success': 0.30, 'token_cost': 2.0, 'flop_cost': 12.12},
    },

    'Llama-3.1-8B': {'size': 8, 'monolithic_llm': {'Entity Match': 0.52, 'Attributes': 0.87, 'Booking Success': 0.50, 'token_cost': 0.07, 'flop_cost': 2.55},
                     'modular_prog': {'Entity Match': 0.37, 'Attributes': 0.73, 'Booking Success': 0.37, 'token_cost': 1.15, 'flop_cost': 6.53},
    },
}

data_cross_model_multi_ds = {
    'Llama-3.2-1B': {'size': 1, 'monolithic_llm': {'Entity Match': 0.0, 'Attributes': 0.0, 'Booking Success': 0.0, 'token_cost': 0.02, 'flop_cost': 0.18},
                     'modular_prog': {'Entity Match': 0.0, 'Attributes': 0.0, 'Booking Success': 0.0, 'token_cost': 0.02, 'flop_cost': 0.18},
    },
    'Llama-3.2-3B': {'size': 3, 'monolithic_llm': {'Entity Match': 0.08, 'Attributes': 0.17, 'Booking Success': 0.07, 'token_cost': 0.16, 'flop_cost': 9.84},
                     'modular_prog': {'Entity Match': 0.0, 'Attributes': 0.0, 'Booking Success': 0.0, 'token_cost': 0.13, 'flop_cost': 6.41},
    },

    'Qwen2.5-7B': {'size': 7, 'monolithic_llm': {'Entity Match': 0.23, 'Attributes': 0.37, 'Booking Success': 0.23, 'token_cost': 0.06, 'flop_cost': 1.94},
                     'modular_prog': {'Entity Match': np.nan, 'Attributes': np.nan, 'Booking Success': np.nan, 'token_cost': np.nan, 'flop_cost': np.nan},
    },

    'Llama-3.1-8B': {'size': 8, 'monolithic_llm': {'Entity Match': 0.33, 'Attributes': 0.33, 'Booking Success': 0.33, 'token_cost': 0.0, 'flop_cost': 0.05},
                     'modular_prog': {'Entity Match': np.nan, 'Attributes': np.nan, 'Booking Success': np.nan, 'token_cost': np.nan, 'flop_cost': np.nan},
    },
}


performance_data = {
    'Monolithic LLM': {
        'model': 'Llama-3.3-70B',
        'task_booking_score': 0.65,
        'token_cost': 0.001,
        'flop_cost': 0.0078
    },
    'Modular Prog': {
        'model': 'Qwen2.5-32B',
        'task_booking_score': 0.65,
        'token_cost': 0.0004,
        'flop_cost': 0.0025
    },
    'Modular LLM': {
        'model': 'Qwen2.5-32B',
        'task_booking_score': 0.03,
        'token_cost': 0.0004,
        'flop_cost': 0.0025
    }
}

model_meta = {
    'Llama-3.2-1B': {'size': 1, 'monolithic_llm': {'token_cost': 0.0, 'flop_cost': 0.01},
                     'modular_prog': {'token_cost': 0.0, 'flop_cost': 0.01},
                     'modular_llm': {'token_cost': 0.0, 'flop_cost': 0.01},
    },
    'Llama-3.2-3B': {'size': 3, 'monolithic_llm': {'token_cost': 0.01, 'flop_cost': 1.6},
                     'modular_prog': {'token_cost': 0.07, 'flop_cost': 2.41},
                     'modular_llm': {'token_cost': 0.0, 'flop_cost': 0.31},
    },

    'Qwen2.5-7B': {'size': 7, 'monolithic_llm': {'token_cost': 0.01, 'flop_cost': 1.88},
                     'modular_prog': {'token_cost': 0.11, 'flop_cost': 4.62},
                     'modular_llm': {'token_cost': 0.01, 'flop_cost': 0.23},
    },

    'Llama-3.1-8B': {'size': 8, 'monolithic_llm': {'token_cost': 0.01, 'flop_cost': 3.29},
                     'modular_prog': {'token_cost': 0.07, 'flop_cost': 6.61},
                     'modular_llm': {'token_cost': 0.01, 'flop_cost': 6.48},
    },

    'Qwen2.5-32B': {'size': 32, 'monolithic_llm': {'token_cost': 0.12, 'flop_cost': 2.91},
                     'modular_prog': {'token_cost': 0.52, 'flop_cost': 4.26},
                     'modular_llm': {'token_cost': 0.15, 'flop_cost': 3.15},
    },            

    'Llama-3.3-70B': {'size': 70, 'monolithic_llm': {'token_cost': 0.03, 'flop_cost': 13.23},
                     'modular_prog': {'token_cost': 0.0, 'flop_cost': 0.0},
                     'modular_llm': {'token_cost': 0.0, 'flop_cost': 0.0},
    },
}


def line_graph_order_by_size(data):
    # Model order based on actual model sizes
    sorted_models = ['Llama-3.2-1B', 'Llama-3.2-3B', 'Qwen2.5-7B', 'Llama-3.1-8B', 'Qwen2.5-32B', 'Llama-3.3-70B']
    model_indices = [models.index(m) for m in sorted_models]
    architectures = list(data.keys())
    metrics = ['Entity Match', 'Attributes', 'Booking Success']
    # Plot again with corrected order
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        for arch in architectures:
            scores = []
            for idx in model_indices:
                val = data[arch][metric][idx]
                scores.append(val if val is not None else np.nan)
            plt.plot(sorted_models, scores, marker='o', label=arch)

        plt.title(f"{metric} Score by Model Size and Architecture (Corrected Order)")
        plt.xlabel("Model (sorted by true size)")
        plt.ylabel(f"{metric} Score")
        plt.xticks(rotation=15)
        plt.ylim(0, 1.1)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        #plt.show()
        plt.savefig(f"/home/admin/Desktop/codebase/cocobots/todsystems/clembench/mono_single/{metric}_score_by_model_size.png")
        

def bar_graph(data):
    # Prepare plot
    metrics = ['Entity Match', 'Attributes', 'Booking Success']
    architectures = list(data.keys())
    x = np.arange(len(models))
    width = 0.2

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    for i, metric in enumerate(metrics):
        ax = axes[i]
        for j, arch in enumerate(architectures):
            values = data[arch][metric]
            values = [v if v is not None else np.nan for v in values]
            ax.bar(x + j * width, values, width=width, label=arch)
        ax.set_title(f"{metric}")
        ax.set_xticks(x + width)
        ax.set_xticklabels(models, rotation=15)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.set_ylabel("Score")
        ax.legend()

    fig.suptitle("Model Performance by Metric and Architecture", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def cost_performance_graph_best(performance_data):
    architectures = list(performance_data.keys())
    scores = [performance_data[arch]['task_booking_score'] for arch in architectures]
    token_costs = [performance_data[arch]['token_cost'] for arch in architectures]
    flop_costs = [performance_data[arch]['flop_cost'] for arch in architectures]

    # Plotting performance vs cost
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Architecture')
    ax1.set_ylabel('Booking Success Score', color=color)
    ax1.bar(architectures, scores, color=color, alpha=0.6, label='Booking Success Score')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, 1.1)

    # Twin axis for cost
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Cost ($)', color=color)
    ax2.plot(architectures, token_costs, marker='o', color='red', label='Token Cost')
    ax2.plot(architectures, flop_costs, marker='s', color='orange', label='FLOP Cost')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, max(max(token_costs), max(flop_costs)) * 1.5)

    # Add legends
    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
    plt.title("Booking Success Score vs Cost per Architecture")
    plt.tight_layout()
    plt.show()    


def cost_performance_graph_all_1(performance_data):
    # Build updated DataFrame with per-(architecture, model) cost
    rows = []
    architectures = ['monolithic_llm', 'modular_prog', 'modular_llm']
    models = list(performance_data.keys())
    for arch in architectures:
        for model in models:
            idx = models.index(model)
            score = data[model][arch]['Booking Success']
            size = model_meta[model]['size']
            token_cost, flop_cost = model_meta[model][arch]['token_cost'], model_meta[model][arch]['flop_cost']
            rows.append({
                'Architecture': arch,
                'Model': model,
                'Booking Success': score if score is not None else np.nan,
                'Token Cost': token_cost,
                'FLOP Cost': flop_cost,
                'Model Size': size
            })

    df = pd.DataFrame(rows)
    df.sort_values(by=['Architecture', 'Model Size'], inplace=True)

    # Plot
    fig, ax1 = plt.subplots(figsize=(14, 6))

    x = np.arange(len(df))
    bar_width = 0.35

    # Bar plot for Booking Success
    ax1.bar(x, df['Booking Success'], bar_width, label='Booking Success', color='tab:blue')
    ax1.set_ylabel('Booking Success Score', color='tab:blue')
    ax1.set_ylim(0, 1.1)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['Model'] + "\n(" + df['Architecture'] + ")", rotation=45, ha='right')

    # Line plot for costs
    ax2 = ax1.twinx()
    ax2.plot(x, df['Token Cost'], color='red', marker='o', label='Token Cost')
    ax2.plot(x, df['FLOP Cost'], color='orange', marker='s', label='FLOP Cost')
    ax2.set_ylabel('Cost ($)', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.set_ylim(0, max(df['FLOP Cost'].max(), df['Token Cost'].max()) * 1.5)

    # Combined legend
    fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.92), bbox_transform=ax1.transAxes)
    plt.title("Booking Success vs Architecture-Specific Cost per Model")
    plt.tight_layout()
    plt.show()


def cost_performance_graph_all(performance_data):

    architectures = ['monolithic_llm', 'modular_prog', 'modular_llm']
    models = list(performance_data.keys())



    unique_models = models  # already in original order
    x = np.arange(len(unique_models))  # one group per model
    bar_width = 0.2

    # For plotting
    booking_scores = {arch: [] for arch in architectures}
    token_costs = {arch: [] for arch in architectures}
    flop_costs = {arch: [] for arch in architectures}

    for model in unique_models:
        size = model_meta[model]['size']
        for arch in architectures:
            idx = models.index(model)
            score = data[model][arch]['Booking Success']
            token_cost, flop_cost = model_meta[model][arch]['token_cost'], model_meta[model][arch]['flop_cost']
            booking_scores[arch].append(score if score is not None else 0)
            token_costs[arch].append(token_cost)
            flop_costs[arch].append(flop_cost)

    # Add FLOP Cost lines to the existing plot
    fig, ax1 = plt.subplots(figsize=(14, 6))

    # Plot Booking Success bars
    for i, arch in enumerate(architectures):
        ax1.bar(x + i * bar_width, booking_scores[arch], width=bar_width, label=f"{arch} - Booking Success")

    ax1.set_ylabel("Booking Success Score")
    ax1.set_ylim(0, 1.1)
    ax1.set_xticks(x + bar_width)
    ax1.set_xticklabels(unique_models, rotation=15)
    ax1.set_xlabel("Model")
    ax1.legend(loc="upper left")

    # Twin axis for Token and FLOP Costs
    ax2 = ax1.twinx()
    for arch in architectures:
        offset = architectures.index(arch) * bar_width
        ax2.plot(x + offset, token_costs[arch], marker='o', linestyle='--', label=f"{arch} - Token Cost")
        ax2.plot(x + offset, flop_costs[arch], marker='s', linestyle=':', label=f"{arch} - FLOP Cost")

    ax2.set_ylabel("Cost ($)")
    ax2.set_ylim(0, max([max(tc + fc) for tc, fc in zip(token_costs.values(), flop_costs.values())]) * 1.5)

    # Combined legend
    fig.legend(loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
    plt.title("Model Comparison: Booking Success with Token and FLOP Cost per Architecture")
    plt.tight_layout()
    #plt.show()
    plt.savefig(f"/home/admin/Desktop/codebase/cocobots/todsystems/clembench/mono_single/cost_perf_comparison.png")


def get_label_name(arch_name):
        if arch_name == 'monolithic_llm':
            return "Monolithic"
        elif arch_name == 'modular_prog':
            return "Modular Prog"
        elif arch_name == 'modular_llm':
            return "Modular LLM"""

def tasksuccess_arch(data):
    # Transform the data
    rows = []
    for model, info in data.items():
        size = info['size']
        for arch in ['monolithic_llm', 'modular_prog', 'modular_llm']:
            booking_success = info[arch]['Booking Success']
            rows.append({
                'Model': model,
                'Size': size,
                'Architecture': arch,
                'Booking Success': booking_success
            })

    df = pd.DataFrame(rows)
    df = df.sort_values(by='Size')

    # Re-plotting the updated graph
    plt.figure(figsize=(12, 6))
    label_name = None
    for arch in df['Architecture'].unique():
        label_name = get_label_name(arch)
        subset = df[df['Architecture'] == arch]
        plt.plot(subset['Model'], subset['Booking Success'], marker='o', markersize=15, label=label_name)

    plt.xticks(rotation=45, fontsize=18)
    plt.xlabel('Model', fontsize=18)
    plt.ylabel('Booking Accuracy', fontsize=18)
    #plt.title('Booking Success Vs Model')
    plt.legend(title='Dialogue System Architecture', fontsize=18)
    plt.tight_layout()
    plt.grid(True)
    #plt.show()
    plt.savefig(f"/home/admin/Desktop/codebase/cocobots/todsystems/clembench/statgraphs/perf_arch_comp.png")


def taskcost_arch_mul_graphs(data):
    # Re-transform the data for cost (token cost and flop cost)
    cost_rows = []
    for model, info in data.items():
        size = info['size']
        for arch in ['monolithic_llm', 'modular_prog', 'modular_llm']:
            token_cost = info[arch]['token_cost']
            flop_cost = info[arch]['flop_cost']
            cost_rows.append({
                'Model': model,
                'Size': size,
                'Architecture': arch,
                'Token Cost': token_cost,
                'FLOP Cost': flop_cost
            })

    cost_df = pd.DataFrame(cost_rows)
    cost_df = cost_df.sort_values(by='Size')

    # Plotting Token Cost
    plt.figure(figsize=(12, 6))
    for arch in cost_df['Architecture'].unique():
        subset = cost_df[cost_df['Architecture'] == arch]
        plt.plot(subset['Model'], subset['Token Cost'], marker='o', label=arch)

    plt.xticks(rotation=45)
    plt.xlabel('Model')
    plt.ylabel('Token Cost')
    plt.title('Token Cost Vs Architecture Across Models')
    plt.legend(title='Architecture')
    plt.tight_layout()
    plt.grid(True)
    plt.show()

    # Plotting FLOP Cost
    plt.figure(figsize=(12, 6))
    for arch in cost_df['Architecture'].unique():
        subset = cost_df[cost_df['Architecture'] == arch]
        plt.plot(subset['Model'], subset['FLOP Cost'], marker='o', label=arch)

    plt.xticks(rotation=45)
    plt.xlabel('Model')
    plt.ylabel('FLOP Cost')
    plt.title('FLOP Cost Vs Architecture Across Models')
    plt.legend(title='Architecture')
    plt.tight_layout()
    plt.grid(True)
    plt.show()


def taskcost_arch_single_graph(data):
    # Re-transform the data for cost (token cost and flop cost)
    cost_rows = []
    for model, info in data.items():
        size = info['size']
        for arch in ['monolithic_llm', 'modular_prog', 'modular_llm']:
            token_cost = info[arch]['token_cost']
            flop_cost = info[arch]['flop_cost']
            cost_rows.append({
                'Model': model,
                'Size': size,
                'Architecture': arch,
                'Token Cost': token_cost,
                'FLOP Cost': flop_cost
            })

    cost_df = pd.DataFrame(cost_rows)
    cost_df = cost_df.sort_values(by='Size')

    # Plotting both Token Cost and FLOP Cost in a single graph using subplots
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Create a second y-axis
    ax2 = ax1.twinx()

    # Plot token cost on ax1
    for arch in cost_df['Architecture'].unique():
        label_name = get_label_name(arch)
        #subset = cost_df[cost_df['Architecture'] == arch]
        subset = cost_df[cost_df['Architecture'] == arch].copy()
        subset['Token Cost'] += np.random.uniform(-0.001, 0.001, len(subset))  # vertical jitter

        ax1.plot(subset['Model'], subset['Token Cost'], marker='o',  markersize=12, linestyle='--', label=f'{label_name} - Token')


    # Plot flop cost on ax2
    for arch in cost_df['Architecture'].unique():
        label_name = get_label_name(arch)
        #subset = cost_df[cost_df['Architecture'] == arch]
        subset = cost_df[cost_df['Architecture'] == arch].copy()
        subset['FLOP Cost'] += np.random.uniform(-0.001, 0.001, len(subset))  # vertical jitter

        ax2.plot(subset['Model'], subset['FLOP Cost'], marker='x',  markersize=12, linestyle='-', label=f'{label_name} - FLOP')

    # Formatting
    ax1.set_xlabel('Model', fontsize=18)
    ax1.set_ylabel('Token Cost ($)', fontsize=18)
    ax2.set_ylabel('FLOPs Cost ($)', fontsize=18)
    #plt.xticks(rotation=45, fontsize=60)
    ax1.tick_params(axis='x', labelsize=17, rotation=30)
    ax1.tick_params(axis='y', labelsize=15)
    ax2.tick_params(axis='y', labelsize=15)
    #plt.title('Token Cost and FLOP Cost Vs Models')

    # Combine legends from both axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    #ax1.legend(lines_1 + lines_2, labels_1 + labels_2, title='DS Architecture - Cost Type', loc='upper left', fontsize=14)
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, title='', loc='upper left', fontsize=14)

    plt.tight_layout()
    plt.grid(True)
    #plt.show()    
    plt.savefig(f"/home/admin/Desktop/codebase/cocobots/todsystems/clembench/statgraphs/cost_arch_comp.png")


def taskcost_booking_arch_2(data):
    # Transform the data
    rows = []
    cost_rows = []    
    for model, info in data.items():
        size = info['size']
        for arch in ['monolithic_llm', 'modular_prog', 'modular_llm']:
            booking_success = info[arch]['Booking Success']
            rows.append({
                'Model': model,
                'Size': size,
                'Architecture': arch,
                'Booking Success': booking_success
            })
            token_cost = info[arch]['token_cost']
            flop_cost = info[arch]['flop_cost']
            cost_rows.append({
                'Model': model,
                'Size': size,
                'Architecture': arch,
                'Token Cost': token_cost,
                'FLOP Cost': flop_cost,
                'Total Flops': info[arch]['total_petaflops']
            })            

    df = pd.DataFrame(rows)
    df = df.sort_values(by='Size')

    cost_df = pd.DataFrame(cost_rows)
    cost_df = cost_df.sort_values(by='Size')    

    # Compute raw FLOPs per token and update cost_df
    #cost_df['Raw FLOPs per Token'] = cost_df['Model'].map(lambda m: 2 * model_params_billions.get(m, np.nan) * 1e9)
    cost_df['Raw FLOPs per Token'] = cost_df['Total Flops'] * 1e15


    # Merge with performance data
    merged_df = pd.merge(df, cost_df, on=['Model', 'Architecture', 'Size'])

    plt.figure(figsize=(12, 8))

    for arch in merged_df['Architecture'].unique():
        label_name = get_label_name(arch)
        subset = merged_df[merged_df['Architecture'] == arch]
        plt.scatter(subset['Raw FLOPs per Token'], subset['Booking Success'], s=100, label=label_name, alpha=0.8)

        # Annotate points
        for _, row in subset.iterrows():
            if pd.notna(row['Raw FLOPs per Token']) and pd.notna(row['Booking Success']):
                plt.text(row['Raw FLOPs per Token'], row['Booking Success'], row['Model'], fontsize=22, ha='center', va='bottom')

    # Apply log scale with defined limits and custom ticks
    '''
    plt.xscale('log')
    plt.xticks(
        ticks=[2e9, 1e10, 2e10, 5e10, 1e11, 2e11],
        labels=["2e9", "1e10", "2e10", "5e10", "1e11", "2e11"]
    )
    '''

    # Axis labels and styling
    plt.xlabel('FLOPs per Token', fontsize=20)
    plt.ylabel('Booking Success', fontsize=20)
    #plt.title('Booking Success vs FLOPs per Token')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    #plt.legend(title='Dialogue System Architecture', fontsize=18)
    plt.legend(title='', fontsize=20)
    plt.tight_layout()
    #plt.show()
    plt.savefig(f"/home/admin/Desktop/codebase/cocobots/todsystems/clembench/statgraphs/cost_perf_arch_comp.png")


def taskcost_booking_arch(data):
    # Transform performance data
    rows = []
    for model, info in data.items():
        size = info['size']
        for arch in ['monolithic_llm', 'modular_prog', 'modular_llm']:
            bs = info[arch]['Booking Success']
            rows.append({'Model': model, 'Size': size, 'Architecture': arch, 'Booking Success': bs})
    df = pd.DataFrame(rows)

    # Transform cost data
    cost_rows = []
    for model, info in data.items():
        size = info['size']
        for arch in ['monolithic_llm', 'modular_prog', 'modular_llm']:
            tc = info[arch]['token_cost']
            fc = info[arch]['flop_cost']
            cost_rows.append({'Model': model, 'Size': size, 'Architecture': arch, 'Token Cost': tc, 'FLOP Cost': fc})
    cost_df = pd.DataFrame(cost_rows)

    # Add Raw FLOPs per Token
    cost_df['Raw FLOPs per Token'] = cost_df['Model'].map(lambda m: 2 * model_params_billions.get(m, np.nan) * 1e9)

    # Merge for final plotting
    merged_df = pd.merge(df, cost_df, on=['Model', 'Architecture', 'Size'])

    # Replot with jitter
    jitter_strength = 0.02
    plt.figure(figsize=(12, 8))
    for arch in merged_df['Architecture'].unique():
        subset = merged_df[merged_df['Architecture'] == arch].copy()
        jitter = np.random.uniform(-jitter_strength, jitter_strength, size=len(subset))
        subset['Jittered FLOPs'] = subset['Raw FLOPs per Token'] * (1 + jitter)
        plt.scatter(subset['Jittered FLOPs'], subset['Booking Success'], s=100, label=arch, alpha=0.8)
        for _, row in subset.iterrows():
            if pd.notna(row['Jittered FLOPs']) and pd.notna(row['Booking Success']):
                plt.text(row['Jittered FLOPs'], row['Booking Success'], row['Model'], fontsize=8, ha='center', va='bottom')

    plt.xscale('log')
    plt.xticks(
        ticks=[2e9, 1e10, 2e10, 5e10, 1e11, 2e11],
        labels=["2e9", "1e10", "2e10", "5e10", "1e11", "2e11"]
    )
    plt.xlabel('FLOPs per Token (log scale)')
    plt.ylabel('Booking Success')
    plt.title('Booking Success vs FLOPs per Token')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(title='Architecture')
    plt.tight_layout()
    #plt.show()    
    plt.savefig(f"/home/admin/Desktop/codebase/cocobots/todsystems/clembench/mono_single/cost_perf_arch_comp.png")

def cross_model_multi_us_perf(data_cross_model_multi_us, filename):
    # Extract model names and corresponding values
    models = list(data_cross_model_multi_us.keys())
    monolithic_values = [data_cross_model_multi_us[model]['monolithic_llm']['Booking Success'] for model in models]
    modular_values = [data_cross_model_multi_us[model]['modular_prog']['Booking Success'] for model in models]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(models, monolithic_values, marker='o', label='Monolithic LLM')
    plt.plot(models, modular_values, marker='o', label='Modular Program')

    plt.title('Booking Success Vs Model')
    plt.xlabel('Model')
    plt.ylabel('Booking Success')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    #plt.show()    
    #plt.savefig(filename)
    plt.savefig(filename)


#line_graph_order_by_size(data)
#cost_performance_graph_all(model_meta)
#tasksuccess_arch(data)   #Use this API for perf vs arch graph
#taskcost_arch_mul_graphs(data)
taskcost_arch_single_graph(data)  #Use this API for cost vs arch graph
#tasksuccess_arch(data)
#taskcost_arch_single_graph(data)
taskcost_booking_arch_2(data) #Use this API for cost vs arch graph
#cross_model_multi_us_perf(data_cross_model_multi_us, f"/home/admin/Desktop/codebase/cocobots/todsystems/clembench/mono_single/cross_model_multi_us.png")
#cross_model_multi_us_perf(data_cross_model_multi_ds, f"/home/admin/Desktop/codebase/cocobots/todsystems/clembench/mono_single/cross_model_multi_ds.png")
