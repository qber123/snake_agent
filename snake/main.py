import pygame
import sys
from snake import Snake

game = Snake(40, 40)

pygame.init()

screen = pygame.display.set_mode((1200, 800))

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
                game.update(action)
            elif event.key == pygame.K_UP:
                action = 1
                game.update(action)
            elif event.key == pygame.K_RIGHT:
                action = 2
                game.update(action)
            elif event.key == pygame.K_DOWN:
                action = 3
                game.update(action)

    screen.fill((0, 0, 0))  
    game.draw(screen, 10)    

    pygame.display.flip()    
    clock.tick(10)           