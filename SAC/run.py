import torch
import gymnasium as gym
import os

from train import initialize_device, initialize_model

PATH_TO_MODEL = "saves\sac_car_racing_iter_40000.pth"

def initialize_run_env():
    env = gym.make("CarRacing-v3", continuous=True, render_mode="human")
    env = gym.wrappers.GrayscaleObservation(env, keep_dim=False) # Converts to grayscale
    env = gym.wrappers.FrameStackObservation(env, stack_size=4) # Stacks the last 4 frames as channels of the image for history, so the channel dimension is now 4 rather than 3 for rgb 
    return env

def load_model_for_testing(path_to_model, device):
    model = initialize_model(device)
    
    if os.path.exists(path_to_model):
        checkpoint = torch.load(path_to_model, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        print(f"Loaded model from {path_to_model}")
        return model

    print(f"Error: Could not find {path_to_model}")
    return

def main():
    device = initialize_device()
    env = initialize_run_env()

    model = load_model_for_testing(PATH_TO_MODEL, device)

    obs, _ = env.reset()
    done = False
    total_reward = 0

    print("Starting run")
    while not done:
        # Preprocess the observation exactly like training
        # (N, H, W, C) -> (1, C, H, W) and normalize
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).to(device).float() # Converts to tensor (C, H, W)
            obs_tensor = obs_tensor.unsqueeze(0) # Adds a batch dimension (1, C, H, W)
            
            # Get action from model
            # We use the 'mean' for evaluation to get the 'best' behavior
            mean, std = model(obs_tensor)
            
            # Clip for safety and move to CPU
            action = torch.clamp(mean, -1, 1).cpu().numpy()[0]
            action[1:] = (action[1:] + 1)/2

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    print(f"Episode Finished")
    env.close()

if __name__ == "__main__":
    main()