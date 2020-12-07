"""
An AI learns to play my favorite childhood video game
"""

import pygame
import neat
import os
import random

# pygame window dimensions
WIN_WIDTH = 600
WIN_HEIGHT = 800

pygame.font.init()

# loading images used in pygame
CAR_IMG = pygame.transform.rotate(pygame.transform.scale(pygame.image.load("./img/car.png"), (180, 180)), 90)
TRACK_IMG = pygame.transform.scale(pygame.image.load('./img/track.png'), (600, 800))
ROAD_BLOCK_IMG = pygame.transform.rotate(pygame.transform.scale(pygame.image.load('./img/block.png'), (500, 80)), 0)
WHITE_BG = pygame.image.load('./img/bg.png')
GAME_FONT = pygame.font.SysFont("times new roman", 50)

# generation 0
GEN = 0


class Car:

    """
    user-controlled car class
    """
    IMG = CAR_IMG

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.img = self.IMG
        self.position = self.x
        self.vel = 0

    def move_left(self):
        self.x = self.x - 50  # states how far the car moves horizontally to left

    def move_right(self):
        self.x = self.x + 50  # states how far the car moves horizontally to right

    def get_mask(self):
        """
        get the mask of the car image

        """
        return pygame.mask.from_surface(self.img)

    def draw(self, window):
        """
        draw the user-controlled car in pygame

        """
        window.blit(self.img, (self.x, self.y))


class Road_block:
    """
    roadblock class
    """
    GAP = 200
    VELOCITY = 50  # states how fast the car travels

    def __init__(self, y):
        self.y = y
        self.position = 0
        self.left = 0
        self.right = 0
        self.ROAD_BLOCK_LEFT = ROAD_BLOCK_IMG
        self.ROAD_BLOCK_RIGHT = ROAD_BLOCK_IMG
        self.passed = False
        self.set_position()

    def set_position(self):
        """
        set the position of the gap between the roadblocks

        """
        self.position = random.randrange(30, 300)
        self.left = self.position - self.ROAD_BLOCK_LEFT.get_width()
        self.right = self.position + self.GAP

    def move(self):
        """
        move the roadblock while car is travelling

        """
        self.y += self.VELOCITY

    def draw(self, window):
        """
        draw the roadblocks in pygame

        """
        window.blit(self.ROAD_BLOCK_LEFT, (self.left, self.y))
        window.blit(self.ROAD_BLOCK_RIGHT, (self.right, self.y))

    def collide(self, car):
        """
        return True if car collided with any of the roadblocks

        """
        car_mask = car.get_mask()
        left_mask = pygame.mask.from_surface(self.ROAD_BLOCK_LEFT)
        right_mask = pygame.mask.from_surface(self.ROAD_BLOCK_RIGHT)

        left_offset = (self.left - car.x, self.y - round(car.y))
        right_offset = (self.right - car.x, self.y - round(car.y))

        left_point = car_mask.overlap(left_mask, left_offset)
        right_point = car_mask.overlap(right_mask, right_offset)

        if left_point or right_point:
            return True

        return False


class Track:
    """
    car track class

    """
    VELOCITY = 50  # states how fast the car travels
    HEIGHT = TRACK_IMG.get_height()
    IMG = TRACK_IMG

    def __init__(self, x):
        self.x = x
        self.y1 = 0
        self.y2 = -self.HEIGHT

    def move(self):
        """
        move the track as player travels

        """
        self.y1 += self.VELOCITY
        self.y2 += self.VELOCITY

        if self.HEIGHT - self.y1 < 0:
            self.y1 = self.y2 - self.HEIGHT

        if self.HEIGHT - self.y2 < 0:
            self.y2 = self.y1 - self.HEIGHT

    def draw(self, window):
        """
        draw the track

        """
        window.blit(self.IMG, (self.x, self.y1))
        window.blit(self.IMG, (self.x, self.y2))


def draw_window(window, cars, road_blocks, bg, runtime, generation, car_number):
    """
    draw the game objects in the game window

    """
    window.blit(WHITE_BG, (0, 0))

    bg.draw(window)

    for road_block in road_blocks:
        road_block.draw(window)

    timer = GAME_FONT.render("Time: " + "{:.2f}".format(runtime), True, (36, 36, 36))
    window.blit(timer, (340, 10))

    gen = GAME_FONT.render("Gen: " + str(generation), True, (36, 36, 36))
    window.blit(gen, (26, 10))

    car_num = GAME_FONT.render("Survivor: " + str(car_number), True, (36, 36, 36))
    window.blit(car_num, (26, 60))

    for car in cars:
        car.draw(window)

    pygame.display.update()


def eval_genome(genomes, config):
    """
    executes the simulation of the current population of cars and sets their fitness based
    on the distance they travelled

    """
    global GEN
    nnets = []
    ge = []
    cars = []
    GEN += 1

    for _, genome in genomes:
        nn = neat.nn.FeedForwardNetwork.create(genome, config)
        nnets.append(nn)
        cars.append(Car(260, 600))
        genome.fitness = 0  # start with a fitness level of 0
        ge.append(genome)

    road_blocks = [Road_block(50)]

    background = Track(0)
    play = True
    clock = pygame.time.Clock()
    runtime = 0
    window = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))

    while play:
        add_road_block = False
        tick = 60
        clock.tick(tick)
        runtime += 1/tick
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                play = False
                pygame.quit()
                quit()

        road_block_index = 0

        # to determine to use the 1st or 2nd roadblocks on the screen
        if len(cars) > 0:
            if len(road_blocks) > 1 and cars[0].y + cars[0].img.get_height() < road_blocks[0].y:
                road_block_index = 1
        else:
            play = False
            break

        for ind, car in enumerate(cars):
            ge[ind].fitness += 0.01  # increase the fitness level of surviving cars as they travel in game

            output = nnets[cars.index(car)].activate((car.x - road_blocks[road_block_index].position,
                                                      road_blocks[road_block_index].right - car.x))

            # determine to move left, right or don't move based on the output of the ANN
            if output[0] > output[1] and output[0] >= 0:
                car.move_left()
            elif output[1] >= 0.1:
                car.move_right()

        remove = []
        for road_block in road_blocks:
            road_block.move()
            for ind, car in enumerate(cars):

                # check for collision
                if road_block.collide(car):
                    ge[ind].fitness -= 1  # reduce fitness level of car collided with roadblocks
                    cars.pop(ind)
                    nnets.pop(ind)
                    ge.pop(ind)

                if not road_block.passed and road_block.y > car.y + car.img.get_height():
                    road_block.passed = True
                    add_road_block = True

            if road_block.y > 800:
                remove.append(road_block)

        # add roadblock and increase car fitness level for passing a roadblock
        if add_road_block:
            for genome in ge:
                genome.fitness += 1

            road_blocks.append(Road_block(150))

        for rem in remove:
            road_blocks.remove(rem)

        for ind, car in enumerate(cars):
            if car.x + car.img.get_width() >= 500 or car.x < 0:
                ge[ind].fitness -= 1
                cars.pop(ind)
                nnets.pop(ind)
                ge.pop(ind)

        background.move()
        draw_window(window, cars, road_blocks, background, runtime, GEN, len(cars))


def play(config_path):
    """
    executes the NEAT algorithm to train the ANN to play the video game with parameters
    specified in the config file

    """
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                                neat.DefaultStagnation, config_path)

    population = neat.Population(config)

    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())

    population.run(eval_genome, 50)  # run for 50 generations

    # save the best genome
    # winner = population.run(eval_genome, 50)
    # with open("winner.pkl", "wb") as output:
    #     pickle.dump(winner, output)


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    play(config_path)
