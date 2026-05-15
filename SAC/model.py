import torch
import torch.nn as nn
from torchrl.modules import TanhNormal

class SAC(nn.Module):
    def __init__(self, input_channels=4, action_dim=3):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4), # 96x96 -> 23x23
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),            # 23x23 -> 10x10
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),            # 10x10 -> 8x8
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*8*8, 512),
            nn.ReLU()
        )

        self.action_mean = nn.Linear(512, action_dim)

        # This initializes the biases for the throttle and brake so that the car initialy presses teh gas, and doesn't press the brake
        with torch.no_grad():
            self.action_mean.bias[1] = 2.0 # 2 run through sigmoid is about 0.88 which is a good initial gas value
            self.action_mean.bias[2] = -2.0 # This initializes teh breaks to a negative bias so it doesn't just brake

        self.actor_log_std_head = nn.Linear(512, action_dim)  # std is the confidence of the action, lower means higher confidence
        # Action is later sampled using a normal distribution from the mean and the std 
        # Also it is predicting the log of the std, later we take the exp(log(std)) which garantues it to be positive

        self.critic_a = nn.Linear(512 + action_dim, 1) # Critic predicts expected reward from current frame until the end of the episode
        self.critic_b = nn.Linear(512 + action_dim, 1)
        # We have two critics in the SAC algorithm because we want to take the min of them to prevent over optimisitc predictions
        # It takes in the image features as well as teh action

    def get_action_dist(self, obs, detach_encoder=False):
        features = self.encoder(obs/255.0)
        
        if detach_encoder: # This is used for the actor update since we don't want to optimize the encoder
            features = features.detach()
        
        # Gets the mean and logstd for the distribution from the models
        mean = self.action_mean(features) 
        log_std = self.actor_log_std_head(features)

        log_std = torch.clamp(log_std, -20, 2) # Stability trick from the paper
        std = torch.exp(log_std) # Since we are predicting log(std) we do e^prediction to get just the std

        # This will shrink our distribution to -1 to 1
        # (Don't forget to later convert gas and brake to 0 to 1 using (throttle/brake + 1)/2 )
        dist = TanhNormal(mean, std)
        
        return dist

    def forward(self, obs):
        dist = self.get_action_dist(obs) # Get the output distribution
    
        # Sample form the distribution
        # The rsample function allows you to sample from the distribution while still keeping gradients because just .sample() wipes the gradients
        # The formula for rsample is action = mean + std*epsilon with epsilon being some noise/randomness
        action = dist.rsample() 
        
        # This returns how likely the model was to choose the action based on how wide the distribution is
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        
        return action, log_prob

    # Used in run script to just run the encoder and action to get a value
    def get_mean_action(self, obs):
        features = self.encoder(obs/255.0)
        mean = self.action_mean(features)
        return torch.tanh(mean)
        