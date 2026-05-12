import torch
import os
import torch.optim as optim
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

from model import SAC
from replay_buffer import ReplayBuffer

PATH_TO_MODEL = None
SAVE_FREQUENCY = 5
PATH_TO_LOGS = "runs/sac_car_racing_v1"

LR = 1e-4

LAMBDA = 0.99

SAMPLE_BATCH_SIZE = 256
NUM_WARMUP_STEPS = 1024
NUM_ITERATIONS = 2000000

BUFFER_CAPACITY = 1000000
STATE_SHAPE = (4, 96, 96)
ACTION_DIM = 3

# Saves model, as well as parameteres
def save_model(model, optimizer, iteration, name="ppo_car_racing"):
    if not os.path.exists("saves"):
        os.makedirs("saves")
        
    filename = f"saves/{name}_iter_{iteration}.pth"
    
    checkpoint = {
        "iteration": iteration,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    
    torch.save(checkpoint, filename)
    print(f"Model saved to {filename}")

def load_model(model, optimizer, filename, device):
    if os.path.exists(filename):
        checkpoint = torch.load(filename, map_location=device)
        
        model.load_state_dict(checkpoint["model_state_dict"]) # Resores weights
        
        # Restore the optimizer state
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            
        print(f"Successfully loaded checkpoint: {filename} (Iteration {checkpoint['iteration']})")
        return checkpoint["iteration"]
    else:
        print(f"No checkpoint found at {filename}")
        return 0


def initialize_device():
    try:
        device = torch.device(0)
    except:
        device = torch.device("cpu")

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
def initialize_env(num_envs=16):
    envs = gym.vector.AsyncVectorEnv([make_env for _ in range(num_envs)])
    return envs

def initialize_model(device):
    model = SAC().to(device)
    return model

def initialize_optimizer(model):
    # This optimizer is for the actor
    actor_optimizer = torch.optim.Adam(
        list(model.encoder.parameters()) + 
        list(model.action_mean.parameters()) + 
        list(model.actor_log_std_head.parameters()), 
        lr=LR
    )

    # This optimizer is for the two critics
    critic_optimizer = torch.optim.Adam(
        list(model.critic_a.parameters()) + 
        list(model.critic_b.parameters()), 
        lr=LR
    )

    return actor_optimizer, critic_optimizer

def buffer_step(device, envs, model, replay_buffer, current_obs):
    action, log_prob = model.forward(current_obs)
    action[:, 1:] = (action[:, 1:] + 1)/2 # This shrinks the throttle and brake to a range of 0 and 1

    next_obs, reward, terminated, truncated, info = envs.step(action)
    next_obs = torch.from_numpy(current_obs).to(device).float()
    
    replay_buffer.add(current_obs, action, reward, next_obs, terminated or truncated) # Adds the step to the buffer

    if terminated or truncated:
        next_obs, _ = envs.reset()
        next_obs = torch.from_numpy(current_obs).to(device).float()

    return next_obs

def train(device, envs, model, actor_optimizer, critic_optimizer, start_iteration, writer, replay_buffer):
    # 1. Create a Target Model (Physical copy for stable targets)
    from copy import deepcopy
    target_model = deepcopy(model).to(device)
    for p in target_model.parameters():
        p.requires_grad = False  # Target networks never learn via gradients
    
    # This sets up the auto alpha (just the weight for log probability, it also gets optimized)
    target_entropy = -ACTION_DIM 
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=LR)
    
    # Reseting environment and converting to tensor
    obs, _ = envs.reset()
    obs = torch.from_numpy(obs).to(device).float()
    TAU = 0.005 # Soft update coefficient

    # Adds some starter data to the buffer
    for i in range(0, NUM_WARMUP_STEPS):
        obs = buffer_step(device, envs, model, replay_buffer, obs)

    for iteration in range(0, NUM_ITERATIONS):
        obs = buffer_step(device, envs, model, replay_buffer, obs) # Take one step and add to buffer
        
        states, actions, rewards, next_states, dones = replay_buffer.sample(SAMPLE_BATCH_SIZE) # Get sample batch
        
        # Converts rewards and dones to tensors
        rewards = torch.FloatTensor(rewards).to(device)
        dones = torch.FloatTensor(dones).to(device)

        alpha = log_alpha.exp() # Get the current alpha value

        with torch.no_grad():
            # Get the actions and log probs from inputting next_states into the model
            next_dist = model.get_action_dist(next_states) # Get distribution
            next_actions = next_dist.rsample() # Sample from dist (rsample keeps gradients)
            next_log_prob = next_dist.log_prob(next_actions).sum(-1, keepdim=True) # Gets the log prob from the dsitribution base don how wide it is

            # Encodes the next states using the CNN
            next_states_encoded = target_model.encoder(next_states/255.0)

            # Gets the critic values for the concatinated encoded states and their corresponding actions
            next_critic_a_out = target_model.critic_a(torch.cat([next_states_encoded, next_actions], dim=-1))
            next_critic_b_out = target_model.critic_b(torch.cat([next_states_encoded, next_actions], dim=-1))

            target = torch.min(next_critic_a_out, next_critic_b_out) - alpha*next_log_prob
            
            # (1 - dones) is used for the mask, it is 1 when the episode is done, so we do 1 - target so that when its done weights go to 0
            # The rest of the formula is just the standard loss function
            target = rewards + (1 - dones)*LAMBDA*target


        curr_states_encoded = model.encoder(states/255.0)
        curr_critic_a_out = model.critic_a(torch.cat([curr_states_encoded, actions], dim=-1))
        curr_critic_b_out = model.critic_b(torch.cat([curr_states_encoded, actions], dim=-1))
        
        critic_a_loss = 0.5(curr_critic_a_out - target)**2
        critic_b_loss = 0.5(curr_critic_b_out - target)**2

        critic_loss = critic_a_loss + critic_b_loss

        # Optimize the critic models
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        # CRITIC OPTIMIZATION DONE
        # MOVING ON TO ACTOR

        # Get the distribution for the current state
        dist = model.get_action_dist(states)
        new_actions = dist.rsample() # Get new actions form the distribution
        new_log_prob = dist.log_prob(new_actions).sum(-1, keepdim=True) # And the log prob

        actor_curr_states_encoded = model.encoder(states/255.0) # Encodes the current states (we re encode so gradients flow to the actor, not the critic)
        
        # Gets the critic outputs from the encoded states
        actor_curr_critic_a_out = model.critic_a(torch.cat([actor_curr_states_encoded, new_actions], dim=-1))
        actor_curr_critic_b_out = model.critic_b(torch.cat([actor_curr_states_encoded, new_actions], dim=-1))
        
        # Calculates the loss function
        actor_loss = (alpha.detach()*new_log_prob - torch.min(actor_curr_critic_a_out, actor_curr_critic_b_out)).mean()

        # Optimizes the actor
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        # Optimizes the alpha variable
        alpha_loss = -(log_alpha*(new_log_prob + target_entropy).detach()).mean() # Loss for alpha (Looks like it is just optimizing towards randomness)
        alpha_optimizer.zero_grad()
        alpha_loss.backward()
        alpha_optimizer.step()

        # This nudges the target model towards the live model
        # If we just used the live model it would basically learn to reward itself because the loss isn't too grounded in the actual reward
        # So we have a stable target model which laggs behind and the live model learns from it
        with torch.no_grad():
            for p, p_t in zip(model.parameters(), target_model.parameters()):
                p_t.data.copy_(TAU * p.data + (1 - TAU) * p_t.data)

        # Tensorboard logging
        if iteration % 100 == 0:
            writer.add_scalar("Loss/Actor", actor_loss.item(), iteration)
            writer.add_scalar("Loss/Critic", critic_loss.item(), iteration)
            writer.add_scalar("Alpha", alpha.item(), iteration)



def main():
    print("Initializing variables")
    writer = initialize_tensorboard(PATH_TO_LOGS)
    device = initialize_device()
    envs = initialize_env()
    replay_buffer = ReplayBuffer(BUFFER_CAPACITY, STATE_SHAPE, ACTION_DIM)

    model = initialize_model(device)
    actor_optimizer, critic_optimizer = initialize_optimizer(model)

    start_iteration = 0
    if PATH_TO_MODEL != None:
        start_iteration = load_model(model, optimizer, PATH_TO_MODEL, device)

    print("Initialized variables")

    print("Beggining training")
    train(device, envs, model, actor_optimizer, critic_optimizer, start_iteration, writer, replay_buffer)
    print("Finished training")

    envs.close()
    writer.close()

if __name__ == "__main__":
    main()