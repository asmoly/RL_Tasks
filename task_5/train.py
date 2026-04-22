import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

PATH_TO_MODEL = "ppo_carracing"
LOG_NAME = "First_Run"

def make_env():
    return gym.make("CarRacing-v3", render_mode=None)

def main():
    #env = gym.make("CarRacing-v3", render_mode=None)
    env = SubprocVecEnv([make_env for _ in range(32)]) # This is for speed, makes 8 environment in parralel

    try:
        model = PPO.load(PATH_TO_MODEL, env=env, device="cuda")
        print("Loaded model from path")
    except:
        model = PPO("CnnPolicy", env, verbose=1, device="cuda", tensorboard_log="./ppo_carracing_tensorboard/")
        print("Created new model")

    print("Training model")
    model.learn(total_timesteps=500_000, tb_log_name=LOG_NAME, reset_num_timesteps=False)

    print("Saved Model")
    model.save("ppo_carracing")

if __name__ == "__main__":
    main()