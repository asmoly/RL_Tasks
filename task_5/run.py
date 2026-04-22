import gymnasium as gym
from stable_baselines3 import PPO

PATH_TO_MODEL = "ppo_carracing"

def main():
    env = gym.make("CarRacing-v3", render_mode="human")

    model = PPO.load(PATH_TO_MODEL, env=env, device="cuda")
    print("Loaded model from path")

    observation, info = env.reset(seed=42)
    while True:
        action, _state = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        #vec_env.render("human")

        if terminated or truncated:
            observation, info = env.reset()

    env.close()

if __name__ == "__main__":
    main()