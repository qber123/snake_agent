import numpy as np
from collections import deque

class Snake:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.field = np.zeros((width, height), dtype=np.uint8)
        
        self.x_apple = np.random.randint(0, width, dtype=np.uint8)
        self.y_apple = np.random.randint(0, height, dtype=np.uint8)
        self.field[self.x_apple][self.y_apple] = 3

        self.snake = deque([
            (width // 2, height // 2),
            (width // 2, height // 2 + 1),
            (width // 2, height // 2 + 2)
        ])
        
        first = True
        for (x, y) in self.snake:
            if first:
                self.field[x][y] = 1
                first = False
            else:
                self.field[x][y] = 2

        self.is_game_over = False
        self.is_food = False

    def spawn_apple(self):
        self.x_apple = np.random.randint(0, self.width, dtype=np.uint8)
        self.y_apple = np.random.randint(0, self.height, dtype=np.uint8)
        if (self.x_apple, self.y_apple) in self.snake:
            self.spawn_apple()

    def count_distance(self, snake_pos):
        distance_vector = (self.x_apple, self.y_apple) - snake_pos
        distance = np.sqrt(distance_vector[0] + distance_vector[1], dtype=np.uint8)
        return distance

    def count_reward(self, old_pos, new_pos):
        reward = 0
        old_distance = self.count_distance(old_pos)
        new_distance = self.count_distance(new_pos)

        if(new_distance < old_distance): reward += 50
        if(self.is_food): reward += 200
        if(self.is_game_over): reward += -10000

        return reward

    def update(self, action):
        snake_dir = ()
        match action:
            case 0: snake_dir = (-1, 0) # left
            case 1: snake_dir = (0, 1) # up
            case 2: snake_dir = (1, 0) # right
            case 3: snake_dir = (0, -1) # down

        new_pos = self.snake[0] + snake_dir

        if new_pos in self.snake or new_pos[0] > self.width or new_pos[0] < 0 or new_pos[1] > self.height or new_pos[1] < 0:
            self.is_game_over = True

        elif (self.snake[0] + snake_dir) != (self.x_apple, self.y_apple):
            self.is_food = True
            self.snake.pop()
            
        self.snake.appendleft(new_pos)

        first = True
        for (x, y) in self.snake:
            if first:
                self.field[x][y] = 1
                first = False
            else:
                self.field[x][y] = 2

        if self.is_food:
            self.spawn_apple()
            self.is_food = False

        self.field[self.x_apple][self.y_apple] = 3

        
    