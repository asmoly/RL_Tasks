import torch
import gymnasium as gym
import os

from train import initialize_device, initialize_model

PATH_TO_MODEL = "saves\sac_car_racing_iter_1000.pth"
# If True, sample actions from the policy distribution (matches training-time behavior).
# If False, use tanh(action_mean) deterministically. With a high-entropy policy the
# stochastic version often reveals what the policy *actually* learned — the mean of a
# noisy policy can correspond to "do nothing" while samples are reasonable behaviors.
STOCHASTIC = True

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
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).to(device).float().unsqueeze(0)

            if STOCHASTIC:
                # Sample from the policy — matches training-time behavior.
                action_tensor, _ = model.forward(obs_tensor)
            else:
                # Deterministic: tanh(action_mean).
                action_tensor = model.get_mean_action(obs_tensor)

            action = action_tensor.cpu().numpy()[0]
            # Same gas/brake remap from [-1, 1] to [0, 1] as in training.
            action[1:] = (action[1:] + 1) / 2

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

    print(f"Episode Finished — total reward: {total_reward:.2f}")
    env.close()

if __name__ == "__main__":
    main()