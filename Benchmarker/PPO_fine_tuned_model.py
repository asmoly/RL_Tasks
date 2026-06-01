import torch
import torch.nn as nn
from tensordict.nn.distributions import NormalParamExtractor
from tensordict.nn import TensorDictModule
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator

# For PPO it is best practice to initialize the weights orthogonally
def layer_init(layer, gain=2 ** 0.5, bias=0.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, bias)
    return layer

class PPOft(nn.Module):
    def __init__(self, input_channels=4, action_dim=3):
        super().__init__()
        
        self.encoder = nn.Sequential(
            layer_init(nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)), # 96x96 -> 23x23
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),            # 23x23 -> 10x10
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),            # 10x10 -> 8x8
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64*8*8, 512)),
            nn.ReLU() 
        )

        self.steering_mean = layer_init(nn.Linear(512, 1), gain=0.01) # Range: [-1, 1]
        self.throttle_mean = layer_init(nn.Linear(512, 2), gain=0.01) # Range: [0, 1] (Gas and Brake)

        # This initializes the biases for the throttle and brake so that the car initialy presses teh gas, and doesn't press the brake
        with torch.no_grad():
            self.throttle_mean.bias[0] = 2.0 # 2 run through sigmoid is about 0.88 which is a good initial gas value
            self.throttle_mean.bias[1] = -2.0 # This initializes the breaks to a negative bias so it doesn't just brake

        self.actor_log_std = nn.Parameter(torch.full((1, action_dim), -0.5)) # std is the confidence of the action, lower means higher confidence
        # Action is later sampled using a normal distribution from the mean and the std 
        # Also it is predicting the log of the std, later we take the exp(log(std)) which garantues it to be positive
        # It is initialized with a tensor [[0, 0, 0]] becaue e^0 = 1 so the std will start at 1

        self.critic = layer_init(nn.Linear(512, 1), gain=1.0) # Critic predicts expected reward from current frame until the end of the episode

    def forward(self, obs):
        features = self.encoder(obs/255.0) # Normalize pixel values
        
        # mean = self.actor_mean(features) # Get mean from actor
        steer = torch.tanh(self.steering_mean(features)) # tanh making it -1 to 1
        throttle = torch.sigmoid(self.throttle_mean(features)) # sigmoid makes it 0 to 1
        
        mean = torch.cat([steer, throttle], dim=-1) # Concatinates the two heads

        std = torch.exp(self.actor_log_std).expand_as(mean) # Does e^(log(std)) to get the std, when expanding the [1, 3] std to the batch dimension of the mean
        
        # Gets the predicted reward from the critic
        value = self.critic(features)
        
        return mean, std, value
        