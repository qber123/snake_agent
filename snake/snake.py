import numpy as np
from collections import deque
import pygame

class Snake:
    def __init__(self):
        self.width = 30
        self.height = 30
        self.field = np.zeros((self.width, self.height))
        
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

    def spawn_apple(self):
        while True:
            x = np.random.randint(0, self.width)
            y = np.random.randint(0, self.height)

            if (x, y) not in self.snake:
                self.x_apple = x
                self.y_apple = y
                break

    def count_distance(self, snake_pos):
        # distance_vector = np.array([self.x_apple, self.y_apple]) - np.array(snake_pos)
        # distance = np.linalg.norm(distance_vector)
        snake_pos = np.array(snake_pos)
        distance_manhatten = np.abs((self.x_apple - snake_pos[0])) + np.abs((self.y_apple - snake_pos[1])) 
        return distance_manhatten

    def count_reward(self, old_pos, new_pos):
        reward = 0
        old_distance = self.count_distance(old_pos)
        new_distance = self.count_distance(new_pos)

        if(self.is_food): reward += 10
        if(self.is_game_over): reward += -10

        reward += 0.1 * (old_distance - new_distance)

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
        hx, hy = self.snake[0]
        ax, ay = self.x_apple, self.y_apple
        dx, dy = self.direction
        observation = np.zeros(13, dtype=np.float32)
        
        # direction information
        match (dx, dy):
            case (-1, 0): observation[0] = 1
            case (0, -1): observation[1] = 1
            case (1, 0): observation[2] = 1
            case (0, 1): observation[3] = 1 
               
        # food relative pos
        if ax > hx: observation[4] = 1 # right
        else: observation[5] = 1 # left
        if ay > hy: observation[6] = 1 # up
        else: observation[7] = 1 # down
        
        # distance to apple
        observation[8] = (ax - hx) / self.width # dx
        observation[9] = (ay - hy) / self.height # dy
        
        left_dir = (dy, -dx)
        right_dir = (-dy, dx)
        
        front_pos = (
            hx + dx,
            hy + dy
        )

        left_pos = (
            hx + left_dir[0],
            hy + left_dir[1]
        )

        right_pos = (
            hx + right_dir[0],
            hy + right_dir[1]
        )
        
        danger_front = int(self.is_collision(front_pos))
        danger_left = int(self.is_collision(left_pos))
        danger_right = int(self.is_collision(right_pos))
        
        # dangers
        observation[10] = danger_front
        observation[11] = danger_left
        observation[12] = danger_right
        
        return observation

    def reset(self):
        self.__init__()
        observation = self.form_obs()
        return observation

    def update(self, action):
        if self.is_game_over:
            observation = self.form_obs()
            return observation, 0, True
        
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
            observation = self.form_obs()
            return observation, -1, True

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
        
        observation = self.form_obs()
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

        
    