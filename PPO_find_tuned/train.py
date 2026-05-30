import torch
import os
import torch.optim as optim
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

from model import PPO

PATH_TO_MODEL = "saves\ppo_car_racing_iter_734.pth"
SAVE_FREQUENCY = 1
PATH_TO_LOGS = "runs/ppo_car_racing_v4"

TOTAL_ITERATIONS = 2000
ROLLOUT_STEPS = 4096
MINI_BATCH_SIZE = 512
EPOCHS = 10
LR = 1e-4 # Initial LR

EPSILON = 0.1 # Make 0.1 at the start of training
CRITIC_WEIGHT = 0.5
ENTROPY_WEIGHT = 0.01 # Initial entropy coefficient

# These are teh params for optimizing the LR
META_LR = 1e-7 # This is the learning rate for optimizing the learning rate
LR_MIN = 1e-7 # Floor for the learning rate
LR_MAX = 1e-2  # Ceiling for the learning rate

# Saves model, as well as parameteres
# Also it saves the current LR (This is for the LR optimization)
def save_ppo_model(model, optimizer, current_lr, iteration, name="ppo_car_racing"):
    if not os.path.exists("saves"):
        os.makedirs("saves")
        
    filename = f"saves/{name}_iter_{iteration}.pth"
    
    checkpoint = {
        "iteration": iteration,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "current_lr": current_lr,
    }
    
    torch.save(checkpoint, filename)
    print(f"Model saved to {filename}")

# Loads model and all its parameters to the optimizer and model
# Also loads LR
def load_ppo_model(model, optimizer, filename, device):
    if os.path.exists(filename):
        checkpoint = torch.load(filename, map_location=device)
        
        model.load_state_dict(checkpoint["model_state_dict"]) # Resores weights
        
        # Restore the optimizer state
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Restore the adapted LR if present, else fall back to the constant LR
        current_lr = checkpoint.get("current_lr", LR)
        for g in optimizer.param_groups:
            g["lr"] = current_lr
            
        print(f"Successfully loaded checkpoint: {filename} (Iteration {checkpoint['iteration']})")
        return checkpoint["iteration"], current_lr
    else:
        print(f"No checkpoint found at {filename}")
        return 0, LR

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

def initialize_env(num_envs=16):
    envs = gym.vector.AsyncVectorEnv([make_env for _ in range(num_envs)])
    return envs

def initialize_model(device):
    model = PPO().to(device)
    return model

def initialize_optimizer(model):
    optimizer = optim.Adam(model.parameters(), lr=LR, eps=1e-5)
    return optimizer

# Flattens all the parameter gradients in the model into a single 1D tensor
# This is used for the hypergradient descent for optimizing the LR
def get_flat_grad(model):
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.detach().flatten()) # Appens the flattened gradient
        else:
            grads.append(torch.zeros(p.numel(), device=p.device)) # Appends a 0 flat vector if it doens't exist
    
    return torch.cat(grads) # Combines them all together

def collect_rollout(envs, model, current_obs, num_steps, device):
    obs_batch = []
    actions_batch = []
    logprobs_batch = []
    values_batch = []
    rewards_batch = []
    dones_batch = []

    for _ in range(num_steps):
        obs_batch.append(current_obs) # State
        
        with torch.no_grad():
            # Current_obs shape: (num_envs, 4, 96, 96)
            mean, std, value = model(current_obs)
            
            dist = torch.distributions.Normal(mean, std) # Creates the normal distribution
            action = dist.sample() # Samples from that distribution

            logprob = dist.log_prob(action).sum(axis=-1) # Gets the probability of each action from the distribution, and sums them
            # This basically gives you the probability that the model chose that action

        env_action = torch.clamp(action, -1, 1).cpu().numpy() # Clip the actions to match what the environment expects
        next_obs, reward, terminated, truncated, info = envs.step(env_action)

        current_obs = torch.from_numpy(next_obs).to(device).float()
        
        # This appends everything we need for the loss function
        #obs_batch.append(current_obs) # State
        actions_batch.append(action) # Action
        logprobs_batch.append(logprob) # Probability of that action
        # This gets flattened because often we get the dimension (num_env, 1, 1) do to the nature of linear layers
        values_batch.append(value.flatten()) # Predicted Rewards
        rewards_batch.append(torch.tensor(reward).to(device) * 0.1) # Actual reward (also normalizing so that the rewards aren't huge)
        dones_batch.append(torch.tensor(terminated | truncated).to(device)) # Done

    return {
        "obs": torch.stack(obs_batch),       # Shape: (steps, num_envs, 3, 96, 96)
        "actions": torch.stack(actions_batch),
        "logprobs": torch.stack(logprobs_batch),
        "values": torch.stack(values_batch),
        "rewards": torch.stack(rewards_batch),
        "dones": torch.stack(dones_batch),
        "last_obs": current_obs
    }
    

def calculate_advantage(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
    # rewards, values, and dones have shape (steps, num_envs)

    num_steps = len(rewards) # This gets the length of the rollout being proccessed
    advantages = torch.zeros_like(rewards) # Create a 0 tensor of same shape as rewards
    previous_advantages = 0 # Stores all previous advantages, so that the first action will have highest weight
    
    # We walk backwards through time
    for t in reversed(range(num_steps)):
        if t == num_steps - 1: # If it is the last value in the tensor, then we use the parameter as the next value
            next_value = last_value # Next value is the predicted value of the state after the rollout
        else:
            next_value = values[t + 1] # Otherwise we can just get it from the values tensor
            
        mask = 1.0 - dones[t].float() # Dones is a mask where 0 is running, 1 is done, so we do 1 - dones to get mask for current value
        
        # Calculates the error (from the formulas for PPO in my powerpoint)
        # If delta is positive then this action resulted in a better reward than what the critic expected
        # if delta is negative it was worse
        delta = rewards[t] + (gamma*next_value*mask) - values[t]
        
        # 4. Apply the GAE smoothing factor (lambda)
        # This links the current delta with the 'future' deltas we already calculated
        previous_advantages = delta + (gamma * lam * mask * previous_advantages)
        advantages[t] = previous_advantages
        
    # Returns are what the Critic should have predicted (Value + Advantage)
    returns = advantages + values
    
    return advantages, returns


def train(device, envs, model, optimizer, current_lr, start_iteration, writer):
    # Reseting environment and converting to tensor
    obs, _ = envs.reset()
    obs = torch.from_numpy(obs).to(device).float()

    # State that persists ACROSS minibatches and iterations for the hypergradient update
    # We need the previous step's gradient g_{t-1} to compute <g_t, g_{t-1}>
    prev_flat_grad = None

    for iteration in range(start_iteration + 1, TOTAL_ITERATIONS):
        data = collect_rollout(envs, model, obs, ROLLOUT_STEPS, device)
        obs = data["last_obs"] # Save for the next iteration

        with torch.no_grad():
            _, _, last_value = model(obs) # Gets the value for the state right after the rollout
            advantages, returns = calculate_advantage(data["rewards"], data["values"], data["dones"], last_value.flatten())

        # Collapses all data (coverts (steps, num_envs) to (steps*num_envs))
        b_obs = data["obs"].reshape((-1, 4, 96, 96))
        b_actions = data["actions"].reshape((-1, 3))
        b_logprobs = data["logprobs"].reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = data["values"].reshape(-1)

        batch_size = b_obs.shape[0] # Gets the batch Size

        total_loss = 0
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        for epoch in range(EPOCHS):
            indices = torch.randperm(batch_size) # Shuffles indicies for better learning

            for start in range(0, batch_size, MINI_BATCH_SIZE): # Loops through the mini batches within the large batch
                end = start + MINI_BATCH_SIZE
                mb_idx = indices[start:end] # This gets a slice of random data of size MINI_BATCH_SIZE

                new_mean, new_std, new_value = model(b_obs[mb_idx]) # Runs the data through the model
                new_dist = torch.distributions.Normal(new_mean, new_std) # Creates a distribution based on the mean and std
                new_logprobs = new_dist.log_prob(b_actions[mb_idx]).sum(-1) # Gets the probability of the action that was actually taken from the models distribution
                
                # Entrypy measures how certain the model is
                # If the distribution is wide than it will be high
                # Low entropy means the model is very confident 
                entropy = new_dist.entropy().sum(-1).mean()


                # This calculates the PPO ratio, from my slides
                # Ratio = New Probability / Old Probability
                logratio = new_logprobs - b_logprobs[mb_idx] # Takes the new models lop probs and divides by the old models log probs
                ratio = torch.exp(logratio) # e^logratio to get rid of the log

                # --- C. Clipped Actor Loss ---
                # Use normalized advantages for stability
                mb_advantages = b_advantages[mb_idx] # Gets the advantages from the data
                mb_advantages = (mb_advantages - mb_advantages.mean())/(mb_advantages.std() + 1e-8) # Normalizes the advantages
                # Basically centering the advatages then dividing by the spread

                # l_clip_1 and l_clip_2 are from the loss formula for the actor part
                l_clip_1 = ratio*mb_advantages
                l_clip_2 = torch.clamp(ratio, 1 - EPSILON, 1 + EPSILON)*mb_advantages

                # This is the actor loss formula from my slides, 
                # We do negative, because normally you need to maximize the loss, but our optimizer only minimizes
                # And we take the mean to get a concret float as the loss
                actor_loss = -torch.min(l_clip_1, l_clip_2).mean() # Then we are taking the minimum of them

                # This is the loss for the critic, again can be found in my slides
                # This is just mse loss
                # In formula this should be negative but again we are minimizing so we don't need to worry about that
                critic_loss = 0.5*(new_value.flatten() - b_returns[mb_idx]).pow(2).mean()
                # 0.5 is to get rid of the 2 power which comes down when differentiation happens (makes gradient cleaner)

                # Total loss equation
                loss = actor_loss + (CRITIC_WEIGHT*critic_loss) - (ENTROPY_WEIGHT*entropy)

                optimizer.zero_grad()
                loss.backward()
                # Clip gradients to stop the model from "exploding"
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

                # LR OPTIMIZER
                flat_grad = get_flat_grad(model) # Grab the current gradients
                if prev_flat_grad is not None: # If the prev gradient exists
                    # We get the dot product between the gradients to calculate a coefficient of their difference
                    dot = torch.dot(flat_grad, prev_flat_grad).item()
                    hyper_grad = -dot # We then negate it

                    # Perform gradient descent on the LR
                    current_lr = current_lr - META_LR*hyper_grad

                    # We then clamp the LR within our parameters
                    current_lr = max(LR_MIN, min(LR_MAX, current_lr))

                    # Add the current learning rate to the optimizer
                    for g in optimizer.param_groups:
                        g["lr"] = current_lr

                # Set the new prev gradient as the current one
                prev_flat_grad = flat_grad

                optimizer.step()

                total_loss += loss.item()
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.item()

        avg_total_loss = total_loss/((batch_size/MINI_BATCH_SIZE)*EPOCHS)
        avg_actor_loss = total_actor_loss/((batch_size/MINI_BATCH_SIZE)*EPOCHS)
        avg_critic_loss = total_critic_loss/((batch_size/MINI_BATCH_SIZE)*EPOCHS)
        avg_entropy = total_entropy/((batch_size/MINI_BATCH_SIZE)*EPOCHS)

        writer.add_scalar("Loss/Actor", avg_actor_loss, iteration)
        writer.add_scalar("Loss/Critic", avg_critic_loss, iteration)
        writer.add_scalar("Loss/Total", avg_total_loss, iteration)
        writer.add_scalar("Entropy", avg_entropy, iteration)
        writer.add_scalar("Hyper/LR", current_lr, iteration)

        print(f"Iteration: {iteration}, Total Loss = {avg_total_loss}, Actor Loss = {avg_actor_loss}, Critic Loss = {avg_critic_loss}, LR = {current_lr:.2e}")

        if iteration%SAVE_FREQUENCY == 0:
            save_ppo_model(model, optimizer, current_lr, iteration)



def main():
    print("Initializing variables")
    writer = initialize_tensorboard(PATH_TO_LOGS)
    device = initialize_device()
    envs = initialize_env()

    model = initialize_model(device)
    optimizer = initialize_optimizer(model)

    start_iteration = 0
    current_lr = LR
    if PATH_TO_MODEL != None:
        start_iteration, current_lr = load_ppo_model(model, optimizer, PATH_TO_MODEL, device)

    print("Initialized variables")

    print("Beggining training")
    train(device, envs, model, optimizer, current_lr, start_iteration, writer)
    print("Finished training")

    envs.close()
    writer.close()

if __name__ == "__main__":
    main()