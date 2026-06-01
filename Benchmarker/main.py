import torch
import matplotlib.pyplot as plt
from operator import itemgetter
from pathlib import Path

from Benchmarkers import PPO_Benchmarker, SAC_Benchmarker, PPO_Fine_Tuned_Benchmarker

PATH_TO_PPO_MODELS = "ppo_models/"
PATH_TO_SAC_MODELS = "sac_models/"
PATH_TO_PPOft_MODELS = "ppo_fine_tuned/"

ITERATIONS = 3


def initialize_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

def models_in_directory(directory_path):
    directory = Path(directory_path)
    return list(directory.glob("*.pth"))

def generate_bar_chart(results):
    models = list(map(itemgetter(0), results)) # Gets a list of the models ['a', 'b', 'c']
    rewards = list(map(itemgetter(1), results)) # Gets a list of their corresponding reward [5, 2, 4]

    # Convert models to string since the path object isn't compatible with matplotlib
    for i in range(0, len(models)):
        models[i] = str(models[i])

    print(models, rewards)

    plt.figure(figsize=(8, 5))

    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f1c40f']
    bars = plt.bar(models, rewards, color=colors, edgecolor='black', width=0.6)

    plt.title('Model Performance Comparison', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Models', fontsize=12, labelpad=10)
    plt.ylabel('Cumilative Reward', fontsize=12, labelpad=10)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval}', ha='center', va='bottom', fontsize=10)

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim(0, max(rewards) * 1.1)  # Gives extra room at the top for labels
    plt.tight_layout()

    # 7. Display or save the graph
    plt.show()

def main():
    device = initialize_device()
    ppo_bench = PPO_Benchmarker(device)
    sac_bench = SAC_Benchmarker(device)
    ppoft_bench = PPO_Fine_Tuned_Benchmarker(device)

    ppo_models = models_in_directory(PATH_TO_PPO_MODELS)
    sac_models = models_in_directory(PATH_TO_SAC_MODELS)
    ppoft_models = models_in_directory(PATH_TO_PPOft_MODELS)

    results = []

    for model in ppo_models:
        score = 0
        for i in range(ITERATIONS):
            score += ppo_bench.benchmark_model(model)
        
        results.append((model, score/ITERATIONS))

    for model in sac_models:
        score = 0
        for i in range(ITERATIONS):
            score += sac_bench.benchmark_model(model)
        
        results.append((model, score/ITERATIONS))

    for model in ppoft_models:
        score = 0
        for i in range(ITERATIONS):
            score += ppoft_bench.benchmark_model(model)
        
        results.append((model, score/ITERATIONS))

    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    
    print("----------------------")

    for result in sorted_results:
        print(f"({result[0]}) : {result[1]}")

    print("----------------------")

    generate_bar_chart(sorted_results)

if __name__ == "__main__":
    main()