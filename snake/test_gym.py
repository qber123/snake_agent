import pygame
import sys
from gymnasium.utils.env_checker import check_env
from snake_gym import Snake

env = Snake(render_mode="human")
check_env(env)

obs, info = env.reset()

for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

env.close()