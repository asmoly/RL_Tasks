import math
import torch
import os
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from model import SAC
from replay_buffer import ReplayBuffer

PATH_TO_MODEL = "saves\sac_car_racing_iter_7000.pth"
PATH_TO_LOGS = "runs/sac_car_racing_v12"

LR = 1e-4
SAVE_FREQUENCY = 1000

LAMBDA = 0.99
TAU = 0.005 # Soft update coefficient

SAMPLE_BATCH_SIZE = 512
NUM_WARMUP_STEPS = 1000
NUM_ITERATIONS = 4000000
NUM_ENVS = 16

UPDATES_PER_ITERATION = 4

BUFFER_CAPACITY = 200000
STATE_SHAPE = (4, 96, 96)
ACTION_DIM = 3

SPEED_BONUS = 0.15 # Reward bonus for going faster
# This is what we want the entropy to be
TARGET_ENTROPY = 1.0
# This is a hard floor for alpha since we want to keep some exploration alive
MIN_ALPHA = 0.1

# This function will slightly shift the input images by a certain padding (number of pixels)
# this gives a bit more randomness to the model and prevents the crtici values from flattening
def random_shift(images, pad=4):
    n, c, h, w = images.shape # Gets the image dimensions
    padded = torch.nn.functional.pad(images, (pad, pad, pad, pad), mode="replicate")
    shifts = torch.randint(0, 2 * pad + 1, size=(n, 2), device=images.device)
    ys = torch.arange(h, device=images.device).view(1, h, 1).float()
    xs = torch.arange(w, device=images.device).view(1, 1, w).float()
    
    ys = ((ys + shifts[:, 0].view(n, 1, 1).float()) / (h + 2 * pad - 1)) * 2 - 1
    xs = ((xs + shifts[:, 1].view(n, 1, 1).float()) / (w + 2 * pad - 1)) * 2 - 1
    grid = torch.stack([xs.expand(n, h, w), ys.expand(n, h, w)], dim=-1)
    return torch.nn.functional.grid_sample(padded, grid, mode="nearest", align_corners=True)

# Saves model, as well as parameteres
def save_model(model, actor_opt, critic_opt, alpha_opt, log_alpha, iteration, name="sac_car_racing"):
    # If a saves folder doesn't exist already create one
    if not os.path.exists("saves"):
        os.makedirs("saves")

    filename = f"saves/{name}_iter_{iteration}.pth" # Construct filename
    checkpoint = { # These are the aspects of the model and optimizers we want to save
        "iteration": iteration,
        "model_state_dict": model.state_dict(),
        "actor_opt_state_dict": actor_opt.state_dict(),
        "critic_opt_state_dict": critic_opt.state_dict(),
        "alpha_opt_state_dict": alpha_opt.state_dict(),
        "log_alpha": log_alpha.detach().cpu(),
    }

    torch.save(checkpoint, filename) # Save
    print(f"Model saved to {filename}")

def load_model(model, actor_opt, critic_opt, filename, device):
    # If the filepath doesn't exist it just returns iteration 0, a new fresh
    if not os.path.exists(filename):
        return 0, torch.zeros(1, requires_grad=True, device=device), None

    # Loads the model checkpoint
    checkpoint = torch.load(filename, map_location=device)

    # Then it loads in parameters saved in the checkpoint into the model and optimizers
    model.load_state_dict(checkpoint["model_state_dict"])
    actor_opt.load_state_dict(checkpoint["actor_opt_state_dict"])
    critic_opt.load_state_dict(checkpoint["critic_opt_state_dict"])

    # Gets teh log alpha value
    log_alpha_value = checkpoint["log_alpha"].detach().to(device).view(1)
    log_alpha = log_alpha_value.clone().requires_grad_(True)

    alpha_opt_state_dict = checkpoint.get("alpha_opt_state_dict")

    print(f"Loaded checkpoint: {filename}")
    return checkpoint["iteration"], log_alpha, alpha_opt_state_dict

def initialize_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

def initialize_tensorboard(path_to_logs):
    writer = SummaryWriter(log_dir=path_to_logs)
    return writer

# This is a wrapper for the environment so that it returns the info of the car as well (specifically speed) for reward shaping
class SpeedInfoWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action) # Takes step
        car = self.env.unwrapped.car # Unwraps the car
        if car is not None:
            # Calculates and sets the speed of the car
            vx, vy = car.hull.linearVelocity[0], car.hull.linearVelocity[1]
            info["speed"] = float(math.sqrt(vx * vx + vy * vy))
        else:
            info["speed"] = 0.0

        return obs, reward, terminated, truncated, info # Extra info variable contaiing speed

# This is a wrapper for frame skipping
# basically one action will last the specified frames
# this is because usually just one frame is too little to actually tell the difference
# in velocity, rotation, etc..
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

# This wrapper will truncate teh episode when its been earning negative orlow rewards for a while
# This prevents the model getting stuck in bad local optimums
class NoProgressTermination(gym.Wrapper):
    def __init__(self, env, window=30, threshold=-8.0, grace=30):
        super().__init__(env)
        self._window = window         # Number of recent steps to look at
        self._threshold = threshold   # If sum over window < threshold -> truncate
        self._grace = grace           # Skip this many steps at episode start
        self._reward_history = []
        self._step_count = 0

    def reset(self, **kwargs):
        self._reward_history = []
        self._step_count = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._step_count += 1
        self._reward_history.append(float(reward))
        if len(self._reward_history) > self._window:
            self._reward_history.pop(0)
        if (
            self._step_count > self._grace
            and len(self._reward_history) >= self._window
            and sum(self._reward_history) < self._threshold
        ):
            truncated = True
        return obs, reward, terminated, truncated, info

# This defines a function on how to create the environemnt
def make_env():
    env = gym.make("CarRacing-v3", continuous=True) # Continuous=True is important since controls are continuous
    env = SpeedInfoWrapper(env) # Exposes car speed in info dict for reward shaping
    env = FrameSkip(env, skip=4) # Add the frame skip wrapper
    env = NoProgressTermination(env, window=30, threshold=-8.0, grace=30)
    env = gym.wrappers.GrayscaleObservation(env, keep_dim=False) # Converts to grayscale
    env = gym.wrappers.FrameStackObservation(env, stack_size=4) # Stacks the last 4 frames as channels of the image for history, so the channel dimension is now 4 rather than 3 for rgb 
    return env

# Creates a specific amount of environments in parrallel
def initialize_env(num_envs=NUM_ENVS):
    envs = gym.vector.AsyncVectorEnv([make_env for _ in range(num_envs)])
    return envs

def initialize_model(device):
    model = SAC().to(device)
    return model

def initialize_optimizer(model):
   # This is the optimizer for the actor
   # it only optimizes the actor encoder, action mean, and the log std head for the actor
    actor_optimizer = torch.optim.Adam(
        list(model.actor_encoder.parameters()) +
        list(model.action_mean.parameters()) +
        list(model.actor_log_std_head.parameters()),
        lr=LR
    )

    # This is the critic optimizer
    critic_optimizer = torch.optim.Adam(
        list(model.critic_encoder.parameters()) +
        list(model.critic_a.parameters()) +
        list(model.critic_b.parameters()),
        lr=LR
    )

    return actor_optimizer, critic_optimizer

def buffer_step(device, envs, model, replay_buffer, current_obs, episode_returns=None, episode_lengths=None, finished_episodes=None, action_override=None):
    # If an action override is provided then we use it directly
    # what action override does, is it will give a range for the actions we want to be initializing with
    # for example moving forwards and not steering too much, this gives us good data in the bufffer to start
    if action_override is not None:
        action_np = action_override.astype(np.float32, copy=False)
    else:
        # Get the predicted action for the observation
        with torch.no_grad():
            action, _ = model.forward(current_obs) # Normalizes internally

        # What the policy actually outputs, all actions are [-1, 1]
        action_np = action.cpu().numpy()

    # This creates a seperate action where the gas and brake get normalized to [0, 1]
    env_action = action_np.copy()
    env_action[:, 1:] = (env_action[:, 1:] + 1) / 2

    # Take a step in the env with the normalized action
    next_obs_np, reward, terminated, truncated, info = envs.step(env_action)
    dones = terminated | truncated  # This creates an array where every element is terminated[n] or truncated[n]

    # This tracks the episode returns/rewards
    if episode_returns is not None:
        episode_returns += reward
        episode_lengths += 1
        for i in range(len(dones)):
            if dones[i]:
                finished_episodes.append((float(episode_returns[i]), int(episode_lengths[i])))
                episode_returns[i] = 0.0
                episode_lengths[i] = 0

    # This is the reward shaping
    # Gets the speed of the car in the batch frames
    speeds = np.asarray(info.get("speed", np.zeros(current_obs.shape[0])), dtype=np.float32)
    shaped_reward = reward + SPEED_BONUS*speeds # Adds the extra speed reward
    scaled_reward = shaped_reward*0.1  # This is just basic normilization

    # Handles when the episode is done, and getting the final observation
    # the reason there are two different versions is because older or newer version of gymnasium
    # use different names
    final_obs_key = None
    final_mask_key = None
    if "final_obs" in info:
        final_obs_key, final_mask_key = "final_obs", "_final_obs"
    elif "final_observation" in info:
        final_obs_key, final_mask_key = "final_observation", "_final_observation"

    # Basically goes through batch and then if it is the final observation it will use the gymnasium final_obs_key as the next obs
    # otherwise it will just go through normally using the initial output of the step function
    for i in range(current_obs.shape[0]):
        if (
            dones[i] # If episode is over (1 - done)
            and final_obs_key is not None
            and info.get(final_mask_key) is not None
            and info[final_mask_key][i]
        ):
            real_next_obs = info[final_obs_key][i]
        else:
            real_next_obs = next_obs_np[i]

        # Info gets added to replay buffer
        replay_buffer.add(
            current_obs[i].cpu().numpy(),
            action_np[i], # Un normalized action [-1, 1]
            scaled_reward[i], 
            real_next_obs, 
            dones[i]
        )

    return torch.from_numpy(next_obs_np).to(device).float()


def train(device, envs, model, actor_optimizer, critic_optimizer, writer, replay_buffer):
    target_entropy = TARGET_ENTROPY # This is wat we want our entropy to be
    log_alpha = torch.zeros(1, requires_grad=True, device=device) # Initializes log alpha so that it is differentiable

    # Loads the model and gets the start iteration
    start_iteration = 0
    alpha_opt_state_dict = None
    if PATH_TO_MODEL is not None:
        start_iteration, log_alpha, alpha_opt_state_dict = load_model(
            model, actor_optimizer, critic_optimizer, PATH_TO_MODEL, device
        )

    # Creates an optimizer for alpha
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=LR)

    if alpha_opt_state_dict is not None:
        alpha_optimizer.load_state_dict(alpha_opt_state_dict)

    # This creates out target model (Copy of live model)
    # This model is what is actually used to calculate the loss in the training loop
    # but this model doesn't change as fast as the live model
    from copy import deepcopy
    target_model = deepcopy(model).to(device)
    for p in target_model.parameters():
        p.requires_grad = False  # Target networks never learn via gradients
    
    # Reseting environment and converting to tensor
    obs, _ = envs.reset()
    obs = torch.from_numpy(obs).to(device).float()

    # This creates our matricies to store the episode returns and lengths
    episode_returns = np.zeros(NUM_ENVS, dtype=np.float64)
    episode_lengths = np.zeros(NUM_ENVS, dtype=np.int64)
    finished_episodes = []  # rolling buffer of (return, length) for completed episodes

    # Adds some starter data to the buffer
    for i in range(0, NUM_WARMUP_STEPS):
        # This generates ovverride data
        warmup_actions = np.empty((NUM_ENVS, ACTION_DIM), dtype=np.float32)
        warmup_actions[:, 0] = np.random.uniform(-0.5, 0.5, size=NUM_ENVS)  # steering
        warmup_actions[:, 1] = np.random.uniform(-0.4, 1.0, size=NUM_ENVS)  # gas -> [0.3, 1.0]
        warmup_actions[:, 2] = -1.0                                          # brake -> 0
        
        # Takes the buffer step
        obs = buffer_step(
            device, envs, model, replay_buffer, obs,
            episode_returns, episode_lengths, finished_episodes,
            action_override=warmup_actions,
        )
    # Discard warmup episode stats — we only care about returns during real training.
    finished_episodes.clear()
    # Creates the parametrs we want to optimize for both optimizers
    actor_params = (
        list(model.actor_encoder.parameters())
        + list(model.action_mean.parameters())
        + list(model.actor_log_std_head.parameters())
    )
    critic_params = (
        list(model.critic_encoder.parameters())
        + list(model.critic_a.parameters())
        + list(model.critic_b.parameters())
    )

    # Loops from start iteration to max iteration
    for iteration in range(start_iteration + 1, NUM_ITERATIONS):
        obs = buffer_step(device, envs, model, replay_buffer, obs, episode_returns, episode_lengths, finished_episodes) # Take one step and add to buffer

        for _ in range(UPDATES_PER_ITERATION):
            # Gets a sample from the replay buffer
            states, actions, rewards, next_states, dones = replay_buffer.sample(SAMPLE_BATCH_SIZE)
            # Converts the info to tensors and puts on GPU
            states = torch.FloatTensor(states).to(device)
            actions = torch.FloatTensor(actions).to(device)
            next_states = torch.FloatTensor(next_states).to(device)
            # Converts rewards and dones to tensors
            rewards = torch.FloatTensor(rewards).to(device).view(-1, 1)
            dones = torch.FloatTensor(dones).to(device).view(-1, 1)

            alpha = log_alpha.exp() # Get the current alpha value

            # Gets shifted images
            states_aug = random_shift(states)
            next_states_aug = random_shift(next_states)

            # This is the critic update
            with torch.no_grad():
                # Gets the next state and log prob from the live model
                next_dist = model.get_action_dist(next_states_aug) # Gets a distribution
                next_actions = next_dist.rsample() # Samples action form the dist (rsample maintains gradients)
                next_log_prob = next_dist.log_prob(next_actions).sum(-1, keepdim=True) # Calculates the log prob from the width of the dist

                # Target critic Q values
                next_critic_a_out, next_critic_b_out = target_model.get_q_values(next_states_aug, next_actions)

                # Calculates the target from the loss formulas
                target = torch.min(next_critic_a_out, next_critic_b_out) - alpha*next_log_prob
                target = rewards + (1 - dones)*LAMBDA*target

            # Current Q values
            curr_critic_a_out, curr_critic_b_out = model.get_q_values(states_aug, actions)

            # Calculate the loss for each critic just using MSE loss
            critic_a_loss = 0.5*(curr_critic_a_out - target).pow(2).mean()
            critic_b_loss = 0.5*(curr_critic_b_out - target).pow(2).mean()
            critic_loss = critic_a_loss + critic_b_loss # Add the losses

            # Optimize the critic
            critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic_params, 1.0)
            critic_optimizer.step()

            # If the critic loss goes to inifinity it just ends the training
            if not torch.isfinite(critic_loss):
                raise RuntimeError(
                    f"Critic loss became non-finite at iteration {iteration}: "
                    f"critic_loss={critic_loss.item()}, target stats=[min={target.min().item():.3g}, "
                    f"max={target.max().item():.3g}, mean={target.mean().item():.3g}], "
                    f"alpha={alpha.item():.3g}"
                )


            # This is the actor update
            dist = model.get_action_dist(states) # Get a distribution from the states (uses actor_encoder)
            new_actions = dist.rsample() # Get the actions
            new_log_prob = dist.log_prob(new_actions).sum(-1, keepdim=True) # Get the log prob

            # Q values for the actor loss
            actor_curr_critic_a_out, actor_curr_critic_b_out = model.get_q_values(states, new_actions)

            # Calculate the actor loss
            actor_loss = (alpha.detach()*new_log_prob - torch.min(actor_curr_critic_a_out, actor_curr_critic_b_out)).mean()

            # Optimize the actor
            actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor_params, 1.0)
            actor_optimizer.step()

            # Calculate loss for alpha and optimize it
            alpha_loss = -(log_alpha * (new_log_prob + target_entropy).detach()).mean()
            alpha_optimizer.zero_grad()
            alpha_loss.backward()
            alpha_optimizer.step()
            # Floor on log_alpha so alpha never decays below MIN_ALPHA. Without this the
            # auto-alpha update will happily push alpha to ~0 and the policy collapses.
            with torch.no_grad():
                log_alpha.clamp_(min=math.log(MIN_ALPHA))

            # This will move the target model towards the live model at a rate of TAU
            with torch.no_grad():
                for p, p_t in zip(model.parameters(), target_model.parameters()):
                    p_t.data.mul_(1 - TAU).add_(p.data, alpha=TAU)

        # Save model every SAVE_FREQUENCY steps
        if iteration % SAVE_FREQUENCY == 0:
            # Log to tensor board
            writer.add_scalar("Loss/Actor", actor_loss.item(), iteration)
            writer.add_scalar("Loss/Critic", critic_loss.item(), iteration)
            writer.add_scalar("Loss/Alpha", alpha_loss.item(), iteration)
            writer.add_scalar("Alpha", alpha.item(), iteration)

            # This logs the episode returns
            if len(finished_episodes) > 0:
                returns = np.array([r for r, _ in finished_episodes], dtype=np.float64)
                lengths = np.array([l for _, l in finished_episodes], dtype=np.float64)
                writer.add_scalar("Episode/Return_mean", returns.mean(), iteration)
                writer.add_scalar("Episode/Return_max", returns.max(), iteration)
                writer.add_scalar("Episode/Length_mean", lengths.mean(), iteration)
                writer.add_scalar("Episode/Count_in_window", len(returns), iteration)
                ep_ret_str = f"ep_ret_mean={returns.mean():.2f}, ep_ret_max={returns.max():.2f}, ep_len_mean={lengths.mean():.0f}, ep_count={len(returns)}"
                finished_episodes.clear()
            else:
                ep_ret_str = "no episodes finished in window"

            writer.flush() # Force scalars onto disk so TensorBoard can see them in near-realtime on Windows
            print(
                f"Iteration: {iteration}, "
                f"Actor Loss: {actor_loss.item():.4f}, "
                f"Critic Loss: {critic_loss.item():.4f}, "
                f"Alpha: {alpha.item():.4f}, "
                f"{ep_ret_str}"
            )

            save_model(model, actor_optimizer, critic_optimizer, alpha_optimizer, log_alpha, iteration)



def main():
    print("Initializing variables")
    writer = initialize_tensorboard(PATH_TO_LOGS)
    device = initialize_device()
    envs = initialize_env()
    replay_buffer = ReplayBuffer(BUFFER_CAPACITY, STATE_SHAPE, ACTION_DIM)

    model = initialize_model(device)
    actor_optimizer, critic_optimizer = initialize_optimizer(model)
    print("Initialized variables")

    print("Beggining training")
    train(device, envs, model, actor_optimizer, critic_optimizer, writer, replay_buffer)
    print("Finished training")

    envs.close()
    writer.close()

if __name__ == "__main__":
    main()