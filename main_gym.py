from snake.snake_gym import Snake
from gymnasium.wrappers import FrameStackObservation
import torch
import torch.nn as nn
import numpy as np
import os

stack_size = 4

model_path = f"{os.getcwd()}/models/ppo/agent-v9.pth"

env = Snake(render_mode="human")
env = FrameStackObservation(env, stack_size=stack_size)

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, n_hid, n_out):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(stack_size, 32, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(),
            
            layer_init(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)),
            nn.ReLU(),         
               
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)),
            nn.ReLU(),
            
            nn.Flatten(),
            layer_init(nn.Linear(64 * 8 * 8, n_hid)),
            nn.ReLU(),  
        )
        
        self.actor = layer_init(nn.Linear(n_hid, n_out), std=0.01)
        self.critic = layer_init(nn.Linear(n_hid, 1), std=1)                

    def get_value(self, x):
        return self.critic(self.network(x))
    
    def get_action(self, x):
        hidden = self.network(x)
        logits = self.actor(hidden)
        
        action = torch.argmax(logits, dim=1)
        
        return action

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x)
        logits = self.actor(hidden)
        probs = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)
    
action_size = env.action_space.n

agent = Agent(512, action_size)
agent.load_state_dict(torch.load(model_path))
agent.eval()

obs, _ = env.reset()
total_reward = 0
done = False

while True:
        
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        action = agent.get_action(obs_tensor)
    
    obs, reward, terminated, truncated, info = env.step(action)
        
    total_reward += reward
    
    done = terminated or truncated
    
    if done:
        obs, _ = env.reset()
        print(total_reward)
        total_reward = 0