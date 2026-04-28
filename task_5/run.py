import gymnasium as gym
from stable_baselines3 import SAC

PATH_TO_MODEL = "sac_carracing_924000_steps.zip"

def main():
    env = gym.make("CarRacing-v3", render_mode="human")

    model = SAC.load(PATH_TO_MODEL, env=env, device="cuda", tensorboard_log="./SAC_RUN_LOGS/")
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