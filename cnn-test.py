import pygame
import sys
from snake.snake import Snake
import torch
import torch.nn as nn
import numpy as np
import os

game = Snake(is_cnn=True)

pygame.init()

screen = pygame.display.set_mode((600, 600))
clock = pygame.time.Clock()

base_dir = os.getcwd()
model_path = f"{base_dir}/models/snake-agent.pth"

class SnakeAgent(nn.Module):
    def __init__(self, n_out):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(7, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            
            nn.MaxPool2d(2)
        )
        
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, n_out)
        )

    def forward(self, x):
        x = self.conv(x)
        return self.head(x)
    

def choose_action(model, obs, epsilon=0.0):
    if np.random.rand() < epsilon:
        return np.random.randint(0, 4)
    
    obs = torch.tensor(obs, dtype=torch.float32, device="cpu")
    obs = obs.unsqueeze(0)
    with torch.no_grad():
        out = model(obs)
    return int(torch.argmax(out).item())

agent = SnakeAgent(n_out=4)
agent.load_state_dict(torch.load(model_path))

obs = game.reset()

total_reward = 0
max_reward = 0

while True:
    action = choose_action(agent, obs)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
                
    obs, reward, done = game.update(action)

    total_reward += reward

    print(f"reward: {total_reward}")

    if(done):
        print(f"finished with: {total_reward}")
        max_reward = max(total_reward, max_reward)
        total_reward = 0
        game.reset()
        

    screen.fill((0, 0, 0))  
    game.draw(screen, 20)    

    pygame.display.flip()    
    clock.tick(10)