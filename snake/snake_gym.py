import gymnasium as gym
import numpy as np
from collections import deque
import pygame


class Snake(gym.Env):
    def __init__(self, render_mode: str = None, truncation_steps: int = None):
        super().__init__()
        
        if render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((600, 600))
            self.clock = pygame.time.Clock()
        
        self.width = 30
        self.height = 30
        self.field = np.zeros((self.width, self.height), dtype=np.float32)
        
        self.x_apple = self.width // 2
        self.y_apple = self.height // 2 - 2
        self.field[self.x_apple][self.y_apple] = 3

        self.snake = deque([
            (self.width // 2, self.height // 2),
            (self.width // 2, self.height // 2 + 1),
            (self.width // 2, self.height // 2 + 2)
        ])
        
        for i, (x, y) in enumerate(self.snake):
            self.field[x, y] = 1 if i == 0 else 2

        self.is_game_over = False
        self.is_food = False
        self.direction = (0, -1)
        
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=3.0,
            shape=(30, 30),
            dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(4)
        self.render_mode = render_mode
        self.max_steps = truncation_steps
        self.steps = 0
        self.reward = 0
        
    def spawn_apple(self):
        while True:
            x = np.random.randint(0, self.width)
            y = np.random.randint(0, self.height)

            if (x, y) not in self.snake:
                self.x_apple = x
                self.y_apple = y
                break

    def count_distance(self, snake_pos):
        snake_pos = np.array(snake_pos)
        distance_manhatten = np.abs((self.x_apple - snake_pos[0])) + np.abs((self.y_apple - snake_pos[1])) 
        return distance_manhatten

    def count_reward(self, old_pos, new_pos):
        reward = 0
        # old_distance = self.count_distance(old_pos)
        # new_distance = self.count_distance(new_pos)

        if(self.is_food): reward += 1

        # reward += 0.1 * (old_distance - new_distance)

        self.reward += reward
        return reward

    def is_collision(self, pos):
        x, y = pos

        return (
            x < 0 or
            x >= self.width or
            y < 0 or
            y >= self.height or
            pos in self.snake
        )

    def form_obs(self):
        observation = np.zeros((30, 30), dtype=np.float32)

        hx, hy = self.snake[0]

        for x, y in self.snake:
            observation[x, y] = 1

        observation[hx, hy] = 2
        observation[self.x_apple, self.y_apple] = 3
        return observation
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.field = np.zeros((self.width, self.height), dtype=np.float32)
                
        self.x_apple = self.width // 2
        self.y_apple = self.height // 2 - 2
        self.field[self.x_apple][self.y_apple] = 3

        self.snake = deque([
            (self.width // 2, self.height // 2),
            (self.width // 2, self.height // 2 + 1),
            (self.width // 2, self.height // 2 + 2)
        ])
        
        for i, (x, y) in enumerate(self.snake):
            self.field[x, y] = 1 if i == 0 else 2

        self.is_game_over = False
        self.is_food = False
        self.direction = (0, -1)
        self.reward = 0
        self.steps = 0

        observation = self.form_obs()
        info = {}

        return observation, info

    def step(self, action):
        info = {}
        
        self.steps += 1

        truncated = (
            self.max_steps is not None
            and self.steps >= self.max_steps
            and not self.is_game_over
        )
        
        
        if self.is_game_over:
            observation = self.form_obs()
            return observation, 0, True, truncated, info
        
        match action:
            case 0: snake_dir = (-1, 0) 
            case 1: snake_dir = (0, -1) 
            case 2: snake_dir = (1, 0) 
            case 3: snake_dir = (0, 1)
            case _: snake_dir = self.direction 
        
        if(self.direction[0] + snake_dir[0] == 0 and self.direction[1] + snake_dir[1] == 0):
            snake_dir = self.direction

        self.direction = snake_dir

        old_pos = self.snake[0]
        new_pos = (old_pos[0] + self.direction[0], old_pos[1] + self.direction[1])

        if self.is_collision(new_pos):
            self.is_game_over = True
            observation = self.form_obs()
            return observation, 0, True, truncated, info

        if (new_pos) != (self.x_apple, self.y_apple):
            self.snake.pop()
        
        if (new_pos) == (self.x_apple, self.y_apple):
            self.is_food = True
            
        self.snake.appendleft(new_pos)

        self.field = np.zeros((self.width, self.height), dtype=np.float32)

        for i, (x, y) in enumerate(self.snake):
            self.field[x, y] = 1 if i == 0 else 2

        reward = self.count_reward(old_pos, new_pos)

        if self.is_food:
            self.spawn_apple()
            self.is_food = False

        self.field[self.x_apple][self.y_apple] = 3
        
        observation = self.form_obs()
        done = self.is_game_over
        
        if self.render_mode == "human":
            self.render()
        
        return observation, reward, done, truncated, info
    
    def close(self):
        if self.render_mode == "human":
            pygame.quit()
            self.is_game_over = True
    
    def render(self):
        if self.render_mode != "human" or self.is_game_over:
            return        
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return
        
        cell_size = 20
        
        for x in range(self.width):
            for y in range(self.height):
                value = self.field[x, y]

                if value == 0:
                    color = (0, 0, 0)
                elif value == 1:
                    color = (0, 255, 0)
                elif value == 2:
                    color = (0, 180, 0)
                elif value == 3:
                    color = (255, 0, 0)

                pygame.draw.rect(
                    self.screen,
                    color,
                    (x * cell_size, y * cell_size, cell_size, cell_size),
                )
        pygame.display.flip()
        self.clock.tick(40)