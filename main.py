import pygame
import sys
from snake.snake import Snake
import torch
import torch.nn as nn
import numpy as np
import os

game = Snake()

pygame.init()

screen = pygame.display.set_mode((600, 600))
clock = pygame.time.Clock()

base_dir = os.getcwd()
model_path = f"{base_dir}/models/snake-agent.pth"

class SnakeAgent(nn.Module):
    def __init__(self, n_input, n_hid, n_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_input, n_hid),
            nn.ReLU(),            
            nn.Linear(n_hid, n_hid),
            nn.ReLU(),
            nn.Linear(n_hid, n_out)
        )
    def forward(self, state):
        return self.net(state)
    

def choose_action(model, obs, epsilon=0.0):
    if np.random.rand() < epsilon:
        return np.random.randint(0, 4)
    
    obs = torch.tensor(obs, dtype=torch.float32, device="cpu")
    obs = obs.unsqueeze(0)
    with torch.no_grad():
        out = model(obs)
    return int(torch.argmax(out).item())

agent = SnakeAgent(n_input=13, n_hid=128, n_out=4)
agent.load_state_dict(torch.load(model_path))

obs = game.reset()

while True:
    action = choose_action(agent, obs)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
                
    obs, reward, done = game.update(action)

    if(done): game.reset()

    screen.fill((0, 0, 0))  
    game.draw(screen, 20)    

    pygame.display.flip()    
    clock.tick(10)           