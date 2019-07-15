import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np

# wednesday 2pm

road1_vertices = [
                [-20, 1, 0],
                [20, 1, 0],
                [-20, 0, 0],
                [0, 0, 0],
                [1, 0, 0],
                [20, 0, 0]
]
road1_edges = [
            [0, 1],
            [2, 3],
            [4, 5]
]

hero_vertices = [
            [-20, .6, 0],
            [-20, .4, 0],
            [-19.8, .6, 0],
            [-19.8, .4, 0]
]
cars_edges = [
            [0, 1],
            [2, 3],
            [0, 2],
            [1, 3]
]

class Sim():
    def __init__(self, x, x_R2):
        self.car_vertices = []
        for j in range(len(x[0])):
            self.car_vertices.append([
                                [x[0][j]/800, .6, 0],
                                [x[0][j]/800, .4, 0],
                                [x[0][j]/800 - .2, .6, 0],
                                [x[0][j]/800 - .2, .4, 0]
            ])
        self.hero_vertices = [
                            [x_R2[0][0]/800, - 4.4, 0],
                            [x_R2[0][0]/800, - 4.6, 0],
                            [x_R2[0][0]/800 - .2, - 4.4, 0],
                            [x_R2[0][0]/800 - .2, - 4.6, 0]
        ]
#        self.hero_vertices
        pass
    
    def draw_cars(self):
        glLineWidth(1)
        glBegin(GL_LINES)
        glColor3f(1, 0, 0)
        for edge in self.cars_edges:
            for vertex in edge:
                glVertex3fv(self.hero_vertices[vertex])
        for j in range(len(x[0])):
            for edge in self.cars_edges:
                for vertex in edge:
                    glVertex3fv(self.car_vertices[j][vertex])
        glEnd()
    
#    def move(self, x, x_R2, t):
#        epsilon = x[t + 1][0] - x[t][0]
#        for j in range(len(x[0])):
#            self.car_vertices = list(map(lambda vert: (vert[0] + epsilon, vert[1], vert[2]), self.car_vertices))
#        epsilon = x_R2[t + 1][0] - x_R2[t][0]
#        self.hero_vertices = list(map(lambda vert: (vert[0] + math.cos(30) * epsilon, vert[1] + math.sin(30) * epsilon, vert[2] + z), self.hero_vertices))

def draw_road():
    glLineWidth(1)
    glBegin(GL_LINES)
    glColor3f(0, 0, 1)
    for edge in road1_edges:
        for vertex in edge:
            glVertex3fv(road1_vertices[vertex])
    glEnd()
    pass

def sim_main(x, x_R2):
    pygame.init()
    display = (1000, 500)
    pygame.display.set_mode(display, DOUBLEBUF|OPENGL)
    
    gluPerspective(45, (display[0]/display[1]), .1, 50)
    
    glTranslatef(0, 0, -25)
    
    sim = Sim(x, x_R2)

    clock = pygame.time.Clock()

    time = len(x)
    for t in range(time - 1):
        clock.tick(3)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        sim.move(x, x_R2, t)

        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        draw_road()
        sim.draw_cars()
        # updates display
        pygame.display.flip()


    
# sim_main()