import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import math

FPS = 60

road1_vertices = [
                [-600, 50, 0],
                [1000, 50, 0],
                [-600, 0, 0],
                [200, 0, 0],
                [400, 0, 0],
                [1000, 0, 0]
]
road1_edges = [
            [0, 1],
            [2, 3],
            [4, 5]
]
road2_vertices = [
                [-600, -200, 0],
                [200, 0, 0],
                [-600, -250, 0],
                [400, 0, 0],
]
road2_edges = [
            [0, 1],
            [2, 3],
]
cars_edges = [
            [0, 1],
            [2, 3],
            [0, 2],
            [1, 3]
]

def addToColumn(array, column_number, value):
    for i in range(len(array)):
        array[i][column_number] = array[i][column_number] + value
    return array

class Sim():
    def __init__(self, x, x_R2):
        self.car_vertices = []
        for j in range(len(x[0])):
            self.car_vertices.append([
                                [x[0][j], 35, 0],
                                [x[0][j], 15, 0],
                                [x[0][j] - 20, 35, 0],
                                [x[0][j] - 20, 15, 0]
            ])
        self.hero_vertices = [
                            [math.cos(math.radians(14.0362434679)) * x_R2[0][0] + 10, - math.sin(math.radians(14.0362434679)) * (400 - x_R2[0][0]) + 35, 0],
                            [math.cos(math.radians(14.0362434679)) * x_R2[0][0] + 10, - math.sin(math.radians(14.0362434679)) * (400 - x_R2[0][0]) - 20 + 35, 0],
                            [math.cos(math.radians(14.0362434679)) * x_R2[0][0] - 20 + 10, - math.sin(math.radians(14.0362434679)) * (400 - x_R2[0][0]) + 35, 0],
                            [math.cos(math.radians(14.0362434679)) * x_R2[0][0] - 20 + 10, - math.sin(math.radians(14.0362434679)) * (400 - x_R2[0][0]) - 20 + 35, 0]
        ]
#        self.hero_vertices
        pass
    
    def draw_cars(self, x):
        glLineWidth(1)
        glBegin(GL_LINES)
        glColor3f(1, 0, 0)
        for edge in cars_edges:
            for vertex in edge:
                glVertex3fv(self.hero_vertices[vertex])
        for j in range(len(x[0])):
            for edge in cars_edges:
                for vertex in edge:
                    glVertex3fv(self.car_vertices[j][vertex])
        glEnd()
    
    def move(self, x, x_R2, t):
#        print("self.car_vertices[0] before movement = ", self.car_vertices[0])
        for j in range(len(x[0])):
            epsilon = x[t + 1][j] - x[t][j]
#            print("x[%d][%d] - x[%d][%d] = %f - %f" % (t + 1, j, t, j, x[t + 1][j], x[t][j]))
            self.car_vertices[j] = addToColumn(self.car_vertices[j], 0, epsilon)
#        print("self.car_vertices[0] after movement = ", self.car_vertices[0])
#        input()
#        print("length of self.hero_vertices = ", len(self.hero_vertices))
        epsilon = math.cos(math.radians(14.0362434679)) * (x_R2[t + 1][0] - x_R2[t][0])
        delta = math.sin(math.radians(14.0362434679)) * (x_R2[t + 1][0] - x_R2[t][0])
        self.hero_vertices = addToColumn(self.hero_vertices, 0, epsilon)
        self.hero_vertices = addToColumn(self.hero_vertices, 1, delta)

def draw_road():
    glLineWidth(2)
    glBegin(GL_LINES)
    glColor3f(0, 0, 1)
    for edge in road1_edges:
        for vertex in edge:
            glVertex3fv(road1_vertices[vertex])
    for edge in road2_edges:
        for vertex in edge:
            glVertex3fv(road2_vertices[vertex])
    glEnd()
    pass

def sim_main(x, x_R2, x_last):
    pygame.init()
    display = (1000, 500)
    pygame.display.set_mode(display, DOUBLEBUF|OPENGL)
    
    gluPerspective(45, (display[0]/display[1]), .1, 1000)
    glTranslatef(-200, 0, -999)
    
    clock = pygame.time.Clock()

    sim = Sim(x, x_R2)
    time = x_last
    for t in range(time):
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        sim.move(x, x_R2, t)

        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        draw_road()
        sim.draw_cars(x)
        # updates display
        pygame.display.flip()


    
# sim_main()