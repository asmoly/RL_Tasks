import torch
import os
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from model import SAC
from replay_buffer import ReplayBuffer

PATH_TO_MODEL = None  # Set to a checkpoint path (e.g. "saves/sac_car_racing_iter_69000.pth") to resume
PATH_TO_LOGS = "runs/sac_car_racing_v2"

LR = 1e-4
SAVE_FREQUENCY = 500

LAMBDA = 0.99
TAU = 0.005 # Soft update coefficient

SAMPLE_BATCH_SIZE = 256
NUM_WARMUP_STEPS = 2056
NUM_ITERATIONS = 2000000
NUM_ENVS = 16

BUFFER_CAPACITY = 120000
STATE_SHAPE = (4, 96, 96)
ACTION_DIM = 3

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
    # If the filepath doesn't exist it just returns iteration 0 and a new fresh log_alpha
    if not os.path.exists(filename):
        return 0, torch.zeros(1, requires_grad=True, device=device)

    # Loads the model checkpoint
    checkpoint = torch.load(filename, map_location=device)

    # Then it loads in parameters saved in the checkpoint into the model and optimizers
    model.load_state_dict(checkpoint["model_state_dict"])
    actor_opt.load_state_dict(checkpoint["actor_opt_state_dict"])
    critic_opt.load_state_dict(checkpoint["critic_opt_state_dict"])

    # Gets teh log alpha value
    log_alpha_value = checkpoint["log_alpha"].detach().to(device).view(1)
    log_alpha = log_alpha_value.clone().requires_grad_(True)

    print(f"Loaded checkpoint: {filename}")
    return checkpoint["iteration"], log_alpha # Returns the iteration and the log_alpha

def initialize_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

def initialize_tensorboard(path_to_logs):
    writer = SummaryWriter(log_dir=path_to_logs)
    return writer

# This defines a function on how to create the environemnt
def make_env():
    env = gym.make("CarRacing-v3", continuous=True) # Continuous=True is important since controls are continuous
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
    # The actor optimizer only optimizes the action_mean model, and the actor_log_std_head model
    # if you were to put the encoder in here it would be able to learn to reward itself
    actor_optimizer = torch.optim.Adam(
        list(model.action_mean.parameters()) + 
        list(model.actor_log_std_head.parameters()), 
        lr=LR
    )

    # The critic optimizer optimizes the encoder and both the critics
    critic_optimizer = torch.optim.Adam(
        list(model.encoder.parameters()) +
        list(model.critic_a.parameters()) + 
        list(model.critic_b.parameters()), 
        lr=LR
    )

    return actor_optimizer, critic_optimizer

def buffer_step(device, envs, model, replay_buffer, current_obs):
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

    scaled_reward = reward*0.1  # Normalize reward

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
    target_entropy = -ACTION_DIM # This is wat we want our entropy to be
    log_alpha = torch.zeros(1, requires_grad=True, device=device) # Initializes log alpha so that it is differentiable

    # Loads the model and gets the start iteration
    start_iteration = 0
    if PATH_TO_MODEL is not None:
        start_iteration, log_alpha = load_model(
            model, actor_optimizer, critic_optimizer, PATH_TO_MODEL, device
        )

    # Creates an optimizer for alpha
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=LR)

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

    # Adds some starter data to the buffer
    for i in range(0, NUM_WARMUP_STEPS):
        obs = buffer_step(device, envs, model, replay_buffer, obs)
    # Creates the parametrs we want to optimize for both optimizers
    actor_params = (
        list(model.action_mean.parameters())
        + list(model.actor_log_std_head.parameters())
    )
    critic_params = (
        list(model.encoder.parameters())
        + list(model.critic_a.parameters())
        + list(model.critic_b.parameters())
    )

    # Loops from start iteration to max iteration
    for iteration in range(start_iteration + 1, NUM_ITERATIONS):
        obs = buffer_step(device, envs, model, replay_buffer, obs) # Take one step and add to buffer

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

        # This is the critic update
        with torch.no_grad():
            # Gets the next state and log prob from the live model
            next_dist = model.get_action_dist(next_states) # Gets a distribution
            next_actions = next_dist.rsample() # Samples action form the dist (rsample maintains gradients)
            next_log_prob = next_dist.log_prob(next_actions).sum(-1, keepdim=True) # Calculates the log prob from the width of the dist

            # Runs the states through the critics
            next_states_encoded = target_model.encoder(next_states/255.0) # Encode the states with the CNN
            # Get the outputs of the critics (concatinates teh state with the corresponding actions)
            next_critic_a_out = target_model.critic_a(torch.cat([next_states_encoded, next_actions], dim=-1))
            next_critic_b_out = target_model.critic_b(torch.cat([next_states_encoded, next_actions], dim=-1))

            # Calculates the target from the loss formulas
            target = torch.min(next_critic_a_out, next_critic_b_out) - alpha*next_log_prob
            target = rewards + (1 - dones)*LAMBDA*target

        # Gets the critic outputs for the current states
        curr_states_encoded = model.encoder(states/255.0)
        curr_critic_a_out = model.critic_a(torch.cat([curr_states_encoded, actions], dim=-1))
        curr_critic_b_out = model.critic_b(torch.cat([curr_states_encoded, actions], dim=-1))

        # Calculate the loss for each critic just using MSE loss
        critic_a_loss = 0.5*(curr_critic_a_out - target).pow(2).mean()
        critic_b_loss = 0.5*(curr_critic_b_out - target).pow(2).mean()
        critic_loss = critic_a_loss + critic_b_loss # Add the losses

        # Optimize the critic
        critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic_params, 1.0)
        critic_optimizer.step()


        # This is the actor update
        dist = model.get_action_dist(states, detach_encoder=True) # Get a distribution from the states
        new_actions = dist.rsample() # Get the actions
        new_log_prob = dist.log_prob(new_actions).sum(-1, keepdim=True) # Get the log prob

        # Get the critic output for the current states
        with torch.no_grad():
            actor_states_encoded = model.encoder(states/255.0)
        actor_curr_critic_a_out = model.critic_a(torch.cat([actor_states_encoded, new_actions], dim=-1))
        actor_curr_critic_b_out = model.critic_b(torch.cat([actor_states_encoded, new_actions], dim=-1))

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

        # This will move the target model towards the live model at a rate of TAU
        with torch.no_grad():
            for p, p_t in zip(model.parameters(), target_model.parameters()):
                p_t.data.mul_(1 - TAU).add_(p.data, alpha=TAU)

        # Log to tensor board
        writer.add_scalar("Loss/Actor", actor_loss.item(), iteration)
        writer.add_scalar("Loss/Critic", critic_loss.item(), iteration)
        writer.add_scalar("Loss/Alpha", alpha_loss.item(), iteration)
        writer.add_scalar("Alpha", alpha.item(), iteration)

        if iteration % 100 == 0:
            print(
                f"Iteration: {iteration}, "
                f"Actor Loss: {actor_loss.item():.4f}, "
                f"Critic Loss: {critic_loss.item():.4f}, "
                f"Alpha: {alpha.item():.4f}"
            )

        # Save model every SAVE_FREQUENCY steps
        if iteration % SAVE_FREQUENCY == 0:
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