import os
import torch
import gymnasium as gym

from PPO_model import PPO
from SAC_model import SAC

class Benchmarker:
    def __init__(self, device):
        self.device = device

    def initialize_run_env(self):
        env = gym.make("CarRacing-v3", continuous=True, render_mode=None)
        env = gym.wrappers.GrayscaleObservation(env, keep_dim=False) # Converts to grayscale
        env = gym.wrappers.FrameStackObservation(env, stack_size=4) # Stacks the last 4 frames as channels of the image for history, so the channel dimension is now 4 rather than 3 for rgb 
        return env
    
    # When inheriting you must fill this
    def load_model(self, path_to_model):
        pass

    def benchmark_model(self, path_to_model):
        env = self.initialize_run_env()
        model = self.load_model(path_to_model)

        obs, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            # Preprocess the observation exactly like training
            # (N, H, W, C) -> (1, C, H, W) and normalize
            with torch.no_grad():
                obs_tensor = torch.from_numpy(obs).to(self.device).float() # Converts to tensor (C, H, W)
                obs_tensor = obs_tensor.unsqueeze(0) # Adds a batch dimension (1, C, H, W)
                
                # Get action from model
                # We use the 'mean' for evaluation to get the 'best' behavior
                mean, std, _ = model(obs_tensor)
                
                # Clip for safety and move to CPU
                action = torch.clamp(mean, -1, 1).cpu().numpy()[0]

            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += reward

        env.close()

        return total_reward


class PPO_Benchmarker(Benchmarker):
    def __init__(self, device):
        super().__init__(device)

    def load_model(self, path_to_model):
        model = PPO().to(self.device)
        
        if os.path.exists(path_to_model):
            checkpoint = torch.load(path_to_model, map_location=self.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()

            print(f"Loaded model from {path_to_model}")
            return model

        print(f"Error: Could not find {path_to_model}")

class FrameSkip(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        total_speed = 0.0
        n = 0
        terminated = truncated = False
        info = {}
        last_obs = None
        for _ in range(self._skip):
            last_obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            total_speed += float(info.get("speed", 0.0))
            n += 1
            if terminated or truncated:
                break
        # Average speed across the skipped frames so the speed bonus represents
        # "how fast was the car going during this action", not just the final frame.
        info["speed"] = total_speed / max(n, 1)
        return last_obs, total_reward, terminated, truncated, info

class SAC_Benchmarker(Benchmarker):
    def __init__(self, device):
        super().__init__(device)

    def initialize_run_env(self):
        env = gym.make("CarRacing-v3", continuous=True, render_mode=None)
        env = FrameSkip(env, skip=4)
        env = gym.wrappers.GrayscaleObservation(env, keep_dim=False) # Converts to grayscale
        env = gym.wrappers.FrameStackObservation(env, stack_size=4) # Stacks the last 4 frames as channels of the image for history, so the channel dimension is now 4 rather than 3 for rgb 
        return env

    def load_model(self, path_to_model):
        model = SAC().to(self.device)
        
        if os.path.exists(path_to_model):
            checkpoint = torch.load(path_to_model, map_location=self.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()

            print(f"Loaded model from {path_to_model}")
            return model

        print(f"Error: Could not find {path_to_model}")

    def benchmark_model(self, path_to_model):
        env = self.initialize_run_env()
        model = self.load_model(path_to_model)

        obs, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            # Preprocess the observation exactly like training
            # (N, H, W, C) -> (1, C, H, W) and normalize
            with torch.no_grad():
                obs_tensor = torch.from_numpy(obs).to(self.device).float() # Converts to tensor (C, H, W)
                obs_tensor = obs_tensor.unsqueeze(0) # Adds a batch dimension (1, C, H, W)
                
                # Get action from model
                # We use the 'mean' for evaluation to get the 'best' behavior
                mean, std = model(obs_tensor)
                
                # Clip for safety and move to CPU
                action = torch.clamp(mean, -1, 1).cpu().numpy()[0]

            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += reward

        env.close()

        return total_reward

    