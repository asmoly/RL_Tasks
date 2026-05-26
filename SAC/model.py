import torch
import torch.nn as nn
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform


def _make_encoder(input_channels, feature_dim):
    return nn.Sequential(
        nn.Conv2d(input_channels, 32, kernel_size=8, stride=4), # 96x96 -> 23x23
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2),            # 23x23 -> 10x10
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1),            # 10x10 -> 8x8
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(64*8*8, feature_dim),
        nn.LayerNorm(feature_dim), # Normalizes the output so that critic loss doesn't go to inf
        nn.ReLU()
    )


class SAC(nn.Module):
    def __init__(self, input_channels=4, action_dim=3):
        super().__init__()

        feature_dim = 512 # Output of encoders

        # We make two seperate encoders for the actor and the critic
        self.actor_encoder = _make_encoder(input_channels, feature_dim)
        self.critic_encoder = _make_encoder(input_channels, feature_dim)

        # This is the actor model
        self.action_mean = nn.Linear(feature_dim, action_dim)

        # Initialize the biases of the throttle and brake so the model begins by driving forwar and not braking
        with torch.no_grad():
            self.action_mean.bias[1] = 0.5   # gas
            self.action_mean.bias[2] = -0.5  # brake

        self.actor_log_std_head = nn.Linear(feature_dim, action_dim)  # std is the confidence of the action, lower means higher confidence
        # Action is later sampled using a normal distribution from the mean and the std
        # Also it is predicting the log of the std, later we take the exp(log(std)) which garantues it to be positive

        self.critic_a = nn.Linear(feature_dim + action_dim, 1) # Critic predicts expected reward from current frame until the end of the episode
        self.critic_b = nn.Linear(feature_dim + action_dim, 1)
        # We have two critics in the SAC algorithm because we want to take the min of them to prevent over optimisitc predictions
        # It takes in the image features as well as teh action

    def get_action_dist(self, obs):
        features = self.actor_encoder(obs/255.0)

        # Gets the mean and logstd for the distribution from the models
        mean = self.action_mean(features)
        log_std = self.actor_log_std_head(features)

        # Clamp the log_std for normalization purposes
        log_std = torch.clamp(log_std, -5, 0)
        std = torch.exp(log_std) # Since we are predicting log(std) we do e^prediction to get just the std

        base_dist = Normal(mean, std) # This first gets the un normalized distribution
        dist = TransformedDistribution(base_dist, [TanhTransform(cache_size=1)]) # Normalizes the dist using TanhTransform (Prevents critic from going to inf)

        return dist

    # This runs the critic encoder and then outputs the values for both the critics (simplifies the training loop)
    def get_q_values(self, obs, actions):
        features = self.critic_encoder(obs/255.0)
        q1 = self.critic_a(torch.cat([features, actions], dim=-1))
        q2 = self.critic_b(torch.cat([features, actions], dim=-1))
        return q1, q2

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
        features = self.actor_encoder(obs/255.0)
        mean = self.action_mean(features)
        return torch.tanh(mean)
