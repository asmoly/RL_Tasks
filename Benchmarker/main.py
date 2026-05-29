import torch
from pathlib import Path

from Benchmarkers import PPO_Benchmarker, SAC_Benchmarker

PATH_TO_PPO_MODELS = "ppo_models/"
PATH_TO_SAC_MODELS = "sac_models/"


def initialize_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

def models_in_directory(directory_path):
    directory = Path(directory_path)
    return list(directory.glob("*.pth"))

def main():
    device = initialize_device()
    ppo_bench = PPO_Benchmarker(device)
    sac_bench = SAC_Benchmarker(device)

    ppo_models = models_in_directory(PATH_TO_PPO_MODELS)
    sac_models = models_in_directory(PATH_TO_SAC_MODELS)

    results = []

    for model in ppo_models:
        score = ppo_bench.benchmark_model(model)
        results.append((model, score))

    for model in sac_models:
        score = sac_bench.benchmark_model(model)
        results.append((model, score))

    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    
    print("----------------------")

    for result in sorted_results:
        print(f"({result[0]}) : {result[1]}")

    print("----------------------")



if __name__ == "__main__":
    main()