import pygame
import sys
from snake import Snake

game = Snake()

pygame.init()
screen = pygame.display.set_mode((600, 600))

clock = pygame.time.Clock()

while True:
    action = -1

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                action = 0
            elif event.key == pygame.K_UP:
                action = 1
            elif event.key == pygame.K_RIGHT:
                action = 2
            elif event.key == pygame.K_DOWN:
                action = 3
                
    obs, reward, done = game.update(action)

    if(done): game.reset()

    screen.fill((0, 0, 0))  
    game.draw(screen, 20)    

    pygame.display.flip()    
    clock.tick(10)           