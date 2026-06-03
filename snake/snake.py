import numpy as np
from collections import deque
import pygame

class Snake:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.field = np.zeros((width, height))
        
        self.x_apple = np.random.randint(0, width)
        self.y_apple = np.random.randint(0, height)
        self.field[self.x_apple][self.y_apple] = 3

        self.snake = deque([
            (width // 2, height // 2),
            (width // 2, height // 2 + 1),
            (width // 2, height // 2 + 2)
        ])
        
        for i, (x, y) in enumerate(self.snake):
            self.field[x, y] = 1 if i == 0 else 2

        self.is_game_over = False
        self.is_food = False
        self.direction = (0, -1)

    def spawn_apple(self):
        while True:
            x = np.random.randint(0, self.width)
            y = np.random.randint(0, self.height)

            if (x, y) not in self.snake:
                self.x_apple = x
                self.y_apple = y
                break

    def count_distance(self, snake_pos):
        distance_vector = np.array([self.x_apple, self.y_apple]) - np.array(snake_pos)
        distance = np.linalg.norm(distance_vector)
        return distance

    def count_reward(self, old_pos, new_pos):
        reward = 0
        old_distance = self.count_distance(old_pos)
        new_distance = self.count_distance(new_pos)

        if(new_distance < old_distance): reward += 50
        if(self.is_food): reward += 200
        if(self.is_game_over): reward += -10000

        return reward

    def reset(self):
        self.__init__(self.width, self.height)
        return self.field

    def update(self, action):
        if self.is_game_over:
            return self.field, 0, True
        
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

        if new_pos in self.snake or new_pos[0] >= self.width or new_pos[0] < 0 or new_pos[1] >= self.height or new_pos[1] < 0:
            self.is_game_over = True
            return self.field, -10000, True

        if (new_pos) != (self.x_apple, self.y_apple):
            self.snake.pop()
        
        if (new_pos) == (self.x_apple, self.y_apple):
            self.is_food = True
            
        self.snake.appendleft(new_pos)

        self.field = np.zeros((self.width, self.height))

        for i, (x, y) in enumerate(self.snake):
            self.field[x, y] = 1 if i == 0 else 2

        if self.is_food:
            self.spawn_apple()
            self.is_food = False

        self.field[self.x_apple][self.y_apple] = 3
        
        observation = self.field
        reward = self.count_reward(old_pos, new_pos)
        done = self.is_game_over
        
        return observation, reward, done

    def draw(self, screen, cell_size):
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
                    screen,
                    color,
                    (x * cell_size, y * cell_size, cell_size, cell_size),
                )

        
    