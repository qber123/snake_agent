from snake.snake_gym import Snake
from gymnasium.wrappers import FrameStackObservation
import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np
import os

stack_size = 4

model_path = f"{os.getcwd()}/models/ppo_cnn_transformer/agent-v1.pth"

env = Snake(render_mode="human")
env = FrameStackObservation(env, stack_size=stack_size)

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class CNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(stack_size, 32, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(),
            
            layer_init(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)),
            nn.ReLU(),         
                
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)),
            nn.ReLU(),  
        )
        
    def forward(self, x):
        return self.network(x)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        
    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class Agent(nn.Module):
    def __init__(self, n_hid, n_out):
        super().__init__()
        
        self.cnn = CNNEncoder()
        
        self.transformer = Transformer(dim=64, depth=2, heads=4, dim_head=16, mlp_dim=128)
        
        self.fc = nn.Sequential(
            layer_init(nn.Linear(64, n_hid)),
            nn.ReLU()
        )
        
        self.pos_embedding = nn.Parameter(torch.randn(1, 64, 64))
        
        self.actor = layer_init(nn.Linear(n_hid, n_out), std=0.01)
        self.critic = layer_init(nn.Linear(n_hid, 1), std=1)                

    def encode(self, x):
        x = self.cnn(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = x + self.pos_embedding
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x

    def get_value(self, x):
        return self.critic(self.encode(x))
    
    def get_action(self, x):
        hidden = self.encode(x)
        logits = self.actor(hidden)
        
        action = torch.argmax(logits, dim=1)
        
        return action.item()

    def get_action_and_value(self, x, action=None):
        hidden = self.encode(x)
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