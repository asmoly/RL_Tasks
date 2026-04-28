import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback


PATH_TO_MODEL = "sac_carracing"
LOG_DIRECTORY = "SAC_LOGS"
LOG_NAME = "First_run"

def make_env():
    env = gym.make("CarRacing-v3", render_mode=None, continuous=True)
    return env

def main():
    env = SubprocVecEnv([make_env for _ in range(12)])

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./SAC_MODELS/",
        name_prefix="sac_carracing",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    try:
        model = SAC.load(PATH_TO_MODEL, env=env, device="cuda")
        print("Loaded model from path")
    except:
        model = SAC("CnnPolicy",
                env,  
                verbose=1,
                device="cuda",          # or "cpu"
                buffer_size=200000,     # default is fine, but 200k is safer for pixels
                learning_starts=5000,   # warm-up before learning
                batch_size=256,
                learning_rate=3e-4,
                ent_coef="auto",
                train_freq=1,
                gradient_steps=1,
                tensorboard_log=f"./{LOG_DIRECTORY}/")

        print("Created new model")

    print("Training model")
    model.learn(total_timesteps=1000000, tb_log_name=LOG_NAME, reset_num_timesteps=False, callback=checkpoint_callback)

    print("Saved Model")
    model.save("sac_carracing")

if __name__ == "__main__":
    main()
