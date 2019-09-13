import gym
#from gym import error, spaces, utils
#from gym.utils import seeding
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from simulation import sim_main

cz_length = 400
iz_length = 30
vStart = 20.0
vf = 20
t_sim = 50
dt = .1
uMax = 3.5
uMin = -3.5
vMax = 50
vMin = 10
#vDes = 30
decDes = -3
vehNum = 10
#vehNum = 10
vehNum_R2 = 1
vehLength = 0
Q = 1100
h_min = 2.2369

def columns(matrix, i):
    return [row[i] for row in matrix]

def makePlots(time, position, velocity, acceleration, tfinal, road_number, vehicle_number, original_road, x_R2, v_R2, u_R2, tf_R2, self_i):
    # plots position
    R1_count = 0
    for j in range(vehicle_number):
        if original_road[j] == "road 1":
            road_color = "blue"
            if R1_count < 1:
                plt.plot(columns(time, 0)[:self_i], columns(position, j)[:self_i], color = road_color, label = '%s' % (original_road[j]))
                R1_count += 1
            else:
                plt.plot(columns(time, 0)[:self_i], columns(position, j)[:self_i], color = road_color)
    road_color = 'red'
    plt.plot(columns(time, 0)[:self_i], columns(x_R2, 0)[:self_i], color = road_color, label = "road 2")
    plt.xlabel("time")
    plt.ylabel("position")
    plt.legend()
    plt.show()
    # plots velocity
    R1_count = 0
    for j in range(vehicle_number):
        if original_road[j] == "road 1":
            road_color = "blue"
            if R1_count < 1:
                plt.plot(columns(time, j)[:self_i], columns(velocity, j)[:self_i], color = road_color, label = '%s' % (original_road[j]))
                R1_count += 1
            else:
                plt.plot(columns(time, j)[:self_i], columns(velocity, j)[:self_i], color = road_color)
    road_color = 'red'
    plt.plot(columns(time, 0)[:self_i], columns(v_R2, 0)[:self_i], color = road_color, label = "road 2")
    plt.xlabel("time")
    plt.ylabel("velocity")
    plt.legend()
    plt.show()
    # plots acceleration
    R1_count = 0
    for j in range(vehicle_number):
        if original_road[j] == "road 1":
            road_color = "blue"
            if R1_count < 1:
                plt.plot(columns(time, j)[:self_i], columns(acceleration, j)[:self_i], color = road_color, label = '%s' % (original_road[j]))
                R1_count += 1
            else:
                plt.plot(columns(time, j)[:self_i], columns(acceleration, j)[:self_i], color = road_color)
    road_color = 'red'
    plt.plot(columns(time, 0)[:self_i], columns(u_R2, 0)[:self_i], color = road_color, label = "road 2")
    plt.xlabel("time")
    plt.ylabel("acceleration")
    plt.legend()
    plt.show()

def vehicleGen():
    H = 3600/Q
    time = []
    x0 = []
    for i in range(vehNum):
        R = random.random()
        h = (H - h_min) * (-math.log(1 - R)) + H - h_min
        if i == 0:
            time.append(h)
        else:
            time.append(time[i - 1] + h)
    for i in range(vehNum):
        x0.append(-time[i] * vStart)
    return x0
        
def gippsFirst(x0, v0, t, i):
    if x0 >= 0:
        vDes = vf
    else:
        vDes = vStart
    va = v0 + 2.5 * uMax * dt * (1 - (v0/vDes)) * math.sqrt(0.025 + (v0/vDes))
    ua = (va - v0)/dt
    xa = x0 + v0 * dt + 0.5 * ua * dt**2
    
    if ua < -5:
        ua = -5
        va = v0 + ua * dt
        xa = x0 + v0 * dt + ua * dt**2
    #print("czLength/current speed = ", czLength/va)
    tf = t + (cz_length - xa)/va
    #print("tf = ", tf)
    return xa, va, ua, tf

def gippsRest(x, v, t, i, j):
    if x[i - 1][j] >= 0:
        vDes = vf
    else:
        vDes = vStart
#    fd1 = 10
    v1 = v[i - 1][j] + 2.5 * uMax * dt * (1 - (v[i - 1][j]/vDes)) * math.sqrt(0.025 + (v[i - 1][j]/vDes))
#    v2 = uMin * dt + math.sqrt(uMin**2 * dt**2 - (uMin * (2 * (x[i][j - 1] - x[i - 1][j] - (vehLength + fd1)) - v[i - 1][j] * dt - (v[i][j - 1]**2/decDes))))
    v2 = 100
    
    v[i][j] = min(v1, v2)
    u = (v[i][j] - v[i - 1][j])/dt
    x[i][j] = x[i - 1][j] + v[i - 1][j] * dt + 0.5 * u * dt**2
    tf = t[i][j] + (cz_length - x[i][j])/v[i][j]
    
    # max and min constraints
    if u >= uMax:
        u = uMax
        v[i][j] = v[i - 1][j] + u * dt
        x[i][j] = x[i - 1][j] + v[i - 1][j] * dt + .5 * u * dt
    if u <= uMin:
        u = uMin
        v[i][j] = v[i - 1][j] + u * dt
        x[i][j] = x[i - 1][j] + v[i - 1][j] * dt + .5 * u * dt
    if v[i][j] >= vMax or v[i][j] <= vMin:
        v[i][j] = v[i - 1][j] + u * dt
#        x[i][j] = x[i - 1][j] + v[i - 1][j] * dt + .5 * u* dt
    return x[i][j], v[i][j], u, tf

    # sets constants to control vehicles
def RTControl(t0, tf, x0, v0):
    #print("beginning of RTControl")
    A = np.array([[t0**3/6, t0**2/2, t0, 1], [t0**2/2, t0, 1, 0], [tf**3/6, tf**2/2, tf, 1], [tf**2/2, tf, 1, 0]])
    Y = np.array([x0, v0, cz_length, vf])
    X = np.linalg.solve(A, Y)
    X = X.tolist()
    return X[0], X[1], X[2], X[3]

# inside control zone
def calcDuringCz(tf, t, x, v, u, i, j):
    a, b, c, d = 0, 0, 0, 0
#    if j == 0:
#        tf[i][j] = t[i][j] + (cz_length - x[i - 1][j])/v[i - 1][j]
#    else:
    tf[i][j] = tf[i][j - 1] + iz_length/vf
    a, b, c, d = RTControl(t[i - 1][j], tf[i][j], x[i - 1][j], v[i - 1][j])
    x[i][j] = (a * t[i][j]**3)/6 + (b * t[i][j]**2)/2 + c * t[i][j] + d
    #print(x[i][j])
    v[i][j] = (a * t[i][j]**2)/2 + b * t[i][j] + c
    u[i][j] = a * t[i][j] + b
    
    # max and min acceleration
    if u[i][j] >= uMax:
        u[i][j] = uMax
        v[i][j] = v[i - 1][j] + u[i][j] * dt
        x[i][j] = x[i - 1][j] + v[i - 1][j] * dt + .5 * u[i][j] * dt
    if u[i][j] <= uMin:
        u[i][j] = uMin
        v[i][j] = v[i - 1][j] + u[i][j] * dt
        x[i][j] = x[i - 1][j] + v[i - 1][j] * dt + .5 * u[i][j] * dt
    if v[i][j] >= vMax or v[i][j] <= vMin:
        v[i][j] = v[i - 1][j] + u[i][j] * dt
        x[i][j] = x[i - 1][j] + v[i - 1][j] * dt + .5 * u[i][j] * dt
#    if tf[i][j] - t0[i][j] <= 2 * dt:
#        print("inside")
#        v[i][j] = vf
#        u[i][j] = 0
#        x[i][j] = x[i - 1][j] + v[i][j] * dt
    return tf, x, v, u

# calculate fuel consumption
def calcFC(x, v, u, dt, l_vehNum):
    b0 = 0.1569;
    b1 = 0.02450;
    b2 = -7.415e-4;
    b3 = 5.975e-5;    
    c0 = 0.07224;
    c1 = 0.09681; 
    c2 = 0.001075;
    FC = [[0 for i in range(l_vehNum)] for j in range(int(t_sim/dt))]
    totalFC = 0
    for i in range(1, int(t_sim/dt) - 2):
        for j in range(l_vehNum):
            if x[i][j] >= 0 and x[i][j] <= cz_length + iz_length:
                #print("i = %d, j = %d" % (i, j))
                #print("x[%d][%d] = %f" % (i, j, x[i][j]))
                if u[i][j] == 0:
                    FC[i][j] = (b0 + b1 * v[i][j] + b2 * v[i][j]**2 + b3 * v[i][j]**3) * dt
                    #print(FC[i][j])
                if u[i][j] > 0:
                    FC[i][j] = (b0 + b1 * v[i][j] + b2 * v[i][j]**2 + b3 * v[i][j]**3 + u[i][j] * (c0 + c1 * v[i][j] + c2 * v[i][j]**2)) * dt
                    #print(FC[i][j])
                if u[i][j] < 0:
                    FC[i][j] = 0
                    #print(FC[i][j])
                #print("FC[%d][%d] = %f" % (i, j, FC[i][j]))
                totalFC = totalFC + FC[i][j]
                #print("totalFC = %d" % (totalFC))
                #input()
    totalFC = totalFC/1000
    #totalFC = totalFC * 0.00235214583
    return totalFC

def reward(tf_returned, action):
    A = 1
    r = .02
    score = 0
    distance = []
    for i in range(1, vehNum + 1):
        distance.append(abs(tf_returned[0] - tf_returned[i]))
    
    min1 = min(distance)
    distance.remove(min1)
    min2 = min(distance)
    
    score = 100 * (1 - .5 * (A * r**min1 + A * r**min2))
#    if action != 0:
#        score -= 10
    score = 2 * (score - 50)
    return score

class MergeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        # give car on secondary road random position, but give the rest of the cars a fixed position
        rand = np.random.rand(1)
        initial_position_R2 = [-(rand[0] * 670 + 105)]
    #    initial_position_R2 = [-220]
        self.vehNum_merged, self.first_vehNum, self.first_vehNum_R2 = 0, 0, 0
    #    initial_position_R1 = vehicleGen()
        initial_position_R1 = [-100, -175, -250, -325, -400, -475, -550, -625, -700, -775]
        
##        initial_position_R1 = [-100, -200, -260, -280, -290]
#        initial_position_R2 = [-130]
#        self.vehNum_merged, self.first_vehNum, self.first_vehNum_R2 = 0, 0, 0
#        initial_position_R1 = vehicleGen()
##        print("initial position = ", initial_position_R1)
        self.t = [[0 for i in range(vehNum + vehNum_R2)] for j in range(int(t_sim/dt))]
        self.x = [[0 for i in range(vehNum)] for j in range(int(t_sim/dt))]
        self.x_R2 = [[0 for i in range(vehNum_R2)] for j in range(int(t_sim/dt))]
        self.x_merged = [[0 for i in range(vehNum)] for j in range(int(t_sim/dt))]
        for j in range(vehNum):
            self.x[0][j] = initial_position_R1[j]
        for j in range(vehNum_R2):
            self.x_R2[0][j] = initial_position_R2[j]
        
        self.v = [[vStart for i in range(vehNum)] for j in range(int(t_sim/dt))]
        self.tf = [[0 for i in range(vehNum)] for j in range(int(t_sim/dt))]
        self.u = [[0 for i in range(vehNum)] for j in range(int(t_sim/dt))]
        
        self.v_R2 = [[vStart for i in range(vehNum_R2)] for j in range(int(t_sim/dt))]
        self.tf_R2 = [[0 for i in range(vehNum_R2)] for j in range(int(t_sim/dt))]
        self.u_R2 = [[0 for i in range(vehNum_R2)] for j in range(int(t_sim/dt))]
        
        self.v_merged = [[vStart for i in range(vehNum + vehNum_R2)] for j in range(int(t_sim/dt))]
        self.tf_merged = [[0 for i in range(vehNum + vehNum_R2)] for j in range(int(t_sim/dt))]
        self.u_merged = [[0 for i in range(vehNum + vehNum_R2)] for j in range(int(t_sim/dt))]
#        x_merged = []
#        v_merged = []
#        tf_merged = []
#        t_merged = []
#        u_merged = []
        self.original_road = []
        for i in range(vehNum):
            self.original_road.append("road 1")
        self.i = 1
        
    def step(self, action):
        if action == 0:
            action = .2
        elif action == 1:
            action = -.2
#        elif action == 2:
#            action = 0
    # FIRST ROAD
        for j in range(self.first_vehNum, vehNum):
#            print("inside first road")
            self.t[self.i][j] = (self.i + 1) * dt
            # before control zone FIRST ROAD
#            if x[i - 1][j] < 0:
#                if j == first_vehNum:
#                    x[i][j], v[i][j], u[i][j], tf[i][j] = gippsFirst(x[i - 1][j], v[i - 1][j], t[i][j], i)
#                    #tf[i][j] = tf[i - 1][j]
#                else:
#                    x[i][j], v[i][j], u[i][j], tf[i][j] = gippsRest(x, v, t, i, j)
                #tf[i][j] = tf[0][j] + t[i][j]
                # in control zone
            if self.x[self.i][j] >= -1000:
                self.x_merged[self.i][j], self.x_merged[self.i - 1][j] = self.x[self.i][j], self.x[self.i - 1][j]
                self.v_merged[self.i][j], self.v_merged[self.i - 1][j] = self.v[self.i][j], self.v[self.i - 1][j]
                self.u_merged[self.i][j], self.u_merged[self.i - 1][j] = self.u[self.i][j], self.u[self.i - 1][j]
                self.tf_merged[self.i][j], self.tf_merged[self.i - 1][j] = self.tf[self.i][j], self.tf[self.i - 1][j]
#                original_road.append("road 1")
                self.first_vehNum += 1
                self.vehNum_merged += 1
#                print("after first road")
        
        # SECOND ROAD
        for j in range(vehNum_R2):
            self.t[self.i][j] = (self.i + 1) * dt
            # before control zone SECOND ROAD
#            if self.x_R2[self.i - 1][j] < 0:
#                if j == first_vehNum_R2:
#                    x_R2[self.i][j], v_R2[self.i][j], u_R2[self.i][j], tf_R2[self.i][j] = gippsFirst(x_R2[self.i - 1][j], v_R2[self.i - 1][j], t[self.i][j], self.i)
#                    #tf_R2[i][j] = tf_R2[i - 1][j]
#                else:
#                    x_R2[self.i][j], v_R2[self.i][j], u_R2[self.i][j], tf_R2[self.i][j] = gippsRest(x_R2, v_R2, t, self.i, j)
                #tf_R2[i][j] = tf_R2[0][j] + t_R2[i][j]
                # in control zone
            if self.x_R2[self.i][j] >= -1000:
#                x_merged[i][vehNum_merged], x_merged[i - 1][vehNum_merged] = x_R2[i][j], x_R2[i - 1][j]
#                v_merged[i][vehNum_merged], v_merged[i - 1][vehNum_merged] = v_R2[i][j], v_R2[i - 1][j]
#                u_merged[i][vehNum_merged], u_merged[i - 1][vehNum_merged] = u_R2[i][j], u_R2[i - 1][j]
#                tf_merged[i][vehNum_merged], tf_merged[i - 1][vehNum_merged] = tf_R2[i][j], tf_R2[i - 1][j]
#                original_road.append("road 2")
#                vehNum_merged += 1
#                first_vehNum_R2 += 1
#                x_R2[i][j], v_R2[i][j], u_R2[i][j], tf_R2[i][j] = gippsFirst(x_R2[i - 1][j], v_R2[i - 1][j], t[i][j], i)
                self.u_R2[self.i][j] = action
                self.v_R2[self.i][j] = self.v_R2[self.i - 1][j] + self.u_R2[self.i][j] * dt
                self.x_R2[self.i][j] = self.x_R2[self.i - 1][j] + self.v_R2[self.i - 1][j] * dt + .5 * self.u_R2[self.i][j] * dt ** 2
                
                self.tf_R2[self.i][j] = self.t[self.i][j] + (cz_length - self.x_R2[self.i][j])/self.v_R2[self.i][j]
                if self.x_R2[self.i][j] >= 400:
                    self.tf_R2[self.i][j] = self.tf_R2[self.i - 1][j]
#                self.tf_R2[self.i][j] = 20
        # MERGED ROAD
        for j in range(self.vehNum_merged):
            self.t[self.i][j] = (self.i + 1) * dt
            #before control zone MERGED ROAD
#            if self.x_merged[self.i - 1][j] >= 0 and self.x_merged[self.i - 1][j] <= cz_length - 3:
#                if j == 0:
#                    self.x_merged[self.i][j], self.v_merged[self.i][j], self.u_merged[self.i][j], self.tf_merged[self.i][j] = gippsFirst(self.x_merged[self.i - 1][j], self.v_merged[self.i - 1][j], self.t[self.i][j], self.i)
#                else:
#                    self.x_merged[self.i][j], self.v_merged[self.i][j], self.u_merged[self.i][j], self.tf_merged[self.i][j] = gippsRest(self.x_merged, self.v_merged, self.t, self.i, j)
#            after control zone
#            print("self.x_merged = ", self.x_merged)
            if self.x_merged[self.i - 1][j] >= -1000:
                if j == 0:
                    self.x_merged[self.i][j], self.v_merged[self.i][j], self.u_merged[self.i][j], self.tf_merged[self.i][j] = gippsFirst(self.x_merged[self.i - 1][j], self.v_merged[self.i - 1][j], self.t[self.i][j], self.i)
                    #tf[i][j] = tf[i - 1][j]
                else:
                    self.x_merged[self.i][j], self.v_merged[self.i][j], self.u_merged[self.i][j], self.tf_merged[self.i][j] = gippsRest(self.x_merged, self.v_merged, self.t, self.i, j)
                if self.x_merged[self.i][j] >= 400:
                    self.tf_merged[self.i][j] = self.tf_merged[self.i - 1][j]
        
        
        tf_returned = []
        tf_returned.append(self.tf_R2[self.i][0])
        for j in range(len(self.tf_merged[0]) - 1):
            tf_returned.append(self.tf_merged[self.i][j])
        score = reward(tf_returned, action)
        
        if self.x_R2[self.i][0] >= 400 or self.i == 499:
            terminal = True
        else:
            terminal = False
        self.i += 1
#        print("length of x_merged = ", self.x_merged[0][8])
        return tf_returned, score, terminal
    
    def reset(self):
        
        # give car on secondary road random position, but give the rest of the cars a fixed position
        rand = np.random.rand(1)
        initial_position_R2 = [-(rand[0] * 670 + 105)]
    #    initial_position_R2 = [-220]
        self.vehNum_merged, self.first_vehNum, self.first_vehNum_R2 = 0, 0, 0
    #    initial_position_R1 = vehicleGen()
        initial_position_R1 = [-100, -175, -250, -325, -400, -475, -550, -625, -700, -775]
        
        self.t = [[0 for i in range(vehNum + vehNum_R2)] for j in range(int(t_sim/dt))]
        self.x = [[0 for i in range(vehNum)] for j in range(int(t_sim/dt))]
        self.x_R2 = [[0 for i in range(vehNum_R2)] for j in range(int(t_sim/dt))]
        self.x_merged = [[0 for i in range(vehNum)] for j in range(int(t_sim/dt))]
        for j in range(vehNum):
            self.x[0][j] = initial_position_R1[j]
        for j in range(vehNum_R2):
            self.x_R2[0][j] = initial_position_R2[j]
        
        self.v = [[vStart for i in range(vehNum)] for j in range(int(t_sim/dt))]
        self.tf = [[0 for i in range(vehNum)] for j in range(int(t_sim/dt))]
        self.u = [[0 for i in range(vehNum)] for j in range(int(t_sim/dt))]
        
        self.v_R2 = [[vStart for i in range(vehNum_R2)] for j in range(int(t_sim/dt))]
        self.tf_R2 = [[0 for i in range(vehNum_R2)] for j in range(int(t_sim/dt))]
        self.u_R2 = [[0 for i in range(vehNum_R2)] for j in range(int(t_sim/dt))]
        
        self.v_merged = [[vStart for i in range(vehNum + vehNum_R2)] for j in range(int(t_sim/dt))]
        self.tf_merged = [[0 for i in range(vehNum + vehNum_R2)] for j in range(int(t_sim/dt))]
        self.u_merged = [[0 for i in range(vehNum + vehNum_R2)] for j in range(int(t_sim/dt))]
#        x_merged = []
#        v_merged = []
#        tf_merged = []
#        t_merged = []
#        u_merged = []
#        self.original_road = []
#        for i in range(vehNum):
#            self.original_road.append("main road")
        self.i = 0
        tf_returned = []
        tf_returned.append(self.t[self.i][j] + dt + (cz_length - self.x_R2[self.i][j])/self.v_R2[self.i][j])
        for j in range(len(self.tf_merged[0]) - 1):
            tf_returned.append(self.t[self.i][j] + dt + (cz_length - self.x[self.i][j])/self.v[self.i][j])
#        print("tf_returned = ", tf_returned)
        self.i = 1
        return tf_returned
    def render(self, mode='human'):
        makePlots(self.t, self.x_merged, self.v_merged, self.u_merged, self.tf_merged, "merged road", vehNum, self.original_road, self.x_R2, self.v_R2, self.u_R2, self.tf_R2, self.i)
#        sim_main(self.x_merged, self.x_R2, self.i - 1)
    def close(self):
        ...

#x = MergeEnv()