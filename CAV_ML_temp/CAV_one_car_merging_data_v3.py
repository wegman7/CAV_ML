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
vehNum_R2 = 1
vehLength = 5
Q = 1100
h_min = 2.2369

PLOT = False
SIM = False
N = 1200

def columns(matrix, i):
    return [row[i] for row in matrix]

def makePlots(time, position, velocity, acceleration, tfinal, road_number, vehicle_number, original_road, x_R2, v_R2, u_R2, tf_R2, i):
    
    # plots position
    R1_count, R2_count = 0, 0
    for j in range(vehicle_number):
        if original_road[j] == "road 1":
            road_color = "blue"
            if R1_count < 1:
                plt.plot(columns(time, 0)[:i], columns(position, j)[:i], color = road_color, label = '%s' % (original_road[j]))
                R1_count += 1
            else:
                plt.plot(columns(time, 0)[:i], columns(position, j)[:i], color = road_color)
    road_color = 'red'
    plt.plot(columns(time, 0)[:i], columns(x_R2, 0)[:i], color = road_color, label = "road 2")
    plt.xlabel("time")
    plt.ylabel("position")
    plt.legend()
    plt.show()
    
    # plots velocity
    R1_count, R2_count = 0, 0
    for j in range(vehicle_number):
        if original_road[j] == "road 1":
            road_color = "blue"
            if R1_count < 1:
                plt.plot(columns(time, j)[:i], columns(velocity, j)[:i], color = road_color, label = '%s' % (original_road[j]))
                R1_count += 1
            else:
                plt.plot(columns(time, j)[:i], columns(velocity, j)[:i], color = road_color)
    road_color = 'red'
    plt.plot(columns(time, 0)[:i], columns(v_R2, 0)[:i], color = road_color, label = "road 2")
    plt.xlabel("time")
    plt.ylabel("velocity")
    plt.legend()
    plt.show()
    
    # plots acceleration
    R1_count, R2_count = 0, 0
    for j in range(vehicle_number):
        if original_road[j] == "road 1":
            road_color = "blue"
            if R1_count < 1:
                plt.plot(columns(time, j)[:i], columns(acceleration, j)[:i], color = road_color, label = '%s' % (original_road[j]))
                R1_count += 1
            else:
                plt.plot(columns(time, j)[:i], columns(acceleration, j)[:i], color = road_color)
    road_color = 'red'
    plt.plot(columns(time, 0)[:i], columns(u_R2, 0)[:i], color = road_color, label = "road 2")
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
    tf = t + (cz_length - xa)/va
    
    return xa, va, ua, tf

def gippsRest(x, v, t, i, j):
    if x[i - 1][j] >= 0:
        vDes = vf
    else:
        vDes = vStart
    fd1 = 10
    v1 = v[i - 1][j] + 2.5 * uMax * dt * (1 - (v[i - 1][j]/vDes)) * math.sqrt(0.025 + (v[i - 1][j]/vDes))
    # v2 = uMin * dt + math.sqrt(uMin**2 * dt**2 - (uMin * (2 * (x[i][j - 1] - x[i - 1][j] - (vehLength + fd1)) - v[i - 1][j] * dt - (v[i][j - 1]**2/decDes))))
    v2 = 100
    v[i][j] = min(v1, v2)
    u = (v[i][j] - v[i - 1][j])/dt
    x[i][j] = x[i - 1][j] + v[i - 1][j] * dt + 0.5 * u * dt**2
    tf = t[i][j] + (cz_length - (x[i][j]))/v[i][j]
    
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
def calcDuringCz(tf, t, x, v, u, i, j, tf_target):
    a, b, c, d = 0, 0, 0, 0
    
    tf[i][j] = tf_target + 0/vf
    a, b, c, d = RTControl(t[i - 1][j], tf[i][j], x[i - 1][j], v[i - 1][j])
    x[i][j] = (a * t[i][j]**3)/6 + (b * t[i][j]**2)/2 + c * t[i][j] + d
    v[i][j] = (a * t[i][j]**2)/2 + b * t[i][j] + c
    u[i][j] = a * t[i][j] + b
    
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
                if u[i][j] == 0:
                    FC[i][j] = (b0 + b1 * v[i][j] + b2 * v[i][j]**2 + b3 * v[i][j]**3) * dt
                if u[i][j] > 0:
                    FC[i][j] = (b0 + b1 * v[i][j] + b2 * v[i][j]**2 + b3 * v[i][j]**3 + u[i][j] * (c0 + c1 * v[i][j] + c2 * v[i][j]**2)) * dt
                if u[i][j] < 0:
                    FC[i][j] = 0
                totalFC = totalFC + FC[i][j]
    totalFC = totalFC/1000
    #totalFC = totalFC * 0.00235214583
    return totalFC

def mainAlg():
    initial_position_R2 = [-220]
    first_vehNum, first_vehNum_R2, vehNum_merged = 0, 0, 0
    initial_position_R1 = vehicleGen()
    
    t = [[0 for i in range(vehNum + vehNum_R2)] for j in range(int(t_sim/dt))]
    x = [[0 for i in range(vehNum)] for j in range(int(t_sim/dt))]
    x_R2 = [[0 for i in range(vehNum_R2)] for j in range(int(t_sim/dt))]
    x_merged = [[0 for i in range(vehNum + vehNum_R2)] for j in range(int(t_sim/dt))]
    for j in range(vehNum):
        x[0][j] = initial_position_R1[j]
    for j in range(vehNum_R2):
        x_R2[0][j] = initial_position_R2[j]
        
    v = [[vStart for i in range(vehNum)] for j in range(int(t_sim/dt))]
    tf = [[0 for i in range(vehNum)] for j in range(int(t_sim/dt))]
    u = [[0 for i in range(vehNum)] for j in range(int(t_sim/dt))]
    
    v_R2 = [[vStart for i in range(vehNum_R2)] for j in range(int(t_sim/dt))]
    tf_R2 = [[0 for i in range(vehNum_R2)] for j in range(int(t_sim/dt))]
    u_R2 = [[0 for i in range(vehNum_R2)] for j in range(int(t_sim/dt))]
    
    original_road = []
    tf_data = []
#    tf_data_one_timestep = [0 for i in range(vehNum + 1)]
    action_data = []
    
    # main loop
    for i in range(1, int(t_sim/dt) - 1):
        tf_data_one_timestep = []
        # FIRST ROAD
        for j in range(first_vehNum, vehNum):
            t[i][j] = (i + 1) * dt
            
        for j in range(vehNum):
            t[i][j] = (i + 1) * dt
            # before control zone MERGED ROAD
            if x[i - 1][j] <= 400:
                if j == 0:
                    x[i][j], v[i][j], u[i][j], tf[i][j] = gippsFirst(x[i - 1][j], v[i - 1][j], t[i][j], i)
                else:
                    x[i][j], v[i][j], u[i][j], tf[i][j] = gippsRest(x, v, t, i, j)
            # after control zone
            else:
                if j == 0:
                    x[i][j], v[i][j], u[i][j], tf[i][j] = gippsFirst(x[i - 1][j], v[i - 1][j], t[i][j], i)
                    #tf[i][j] = tf[i - 1][j]
                else:
                    x[i][j], v[i][j], u[i][j], tf[i][j] = gippsRest(x, v, t, i, j)
                tf[i][j] = tf[i - 1][j]
                
        if x[i][j] >= 0 and x[i][j] <= cz_length - 3:
            original_road.append("road 1")
        
        # calculate desired tf of car on secondary road
        for j in range(vehNum):
            if x[i][j] <= x_R2[i - 1][0]:
                tf_preceeding = tf[i][j]
                break
        for j in range(vehNum - 1, -1, -1):
            if x[i][j] >= x_R2[i - 1][0]:
                tf_succeeding = tf[i][j]
                break
        tf_target = (tf_preceeding + tf_succeeding)/2
        tf_R2[0][0] = t[0][0] + (cz_length - (x_R2[0][0]))/v_R2[0][0]
        if tf_R2[i - 1][0] < tf_target:
            action = -.2
        else:
            action = .2
            
            
        # SECOND ROAD
        for j in range(first_vehNum_R2, vehNum_R2):
            t[i][j] = (i + 1) * dt
            # in control zone
#            if x_R2[i][j] <= cz_length:
#                tf_R2, x_R2, v_R2, u_R2 = calcDuringCz(tf_R2, t, x_R2, v_R2, u_R2, i, j, tf_target)
            u_R2[i][j] = action
            v_R2[i][j] = v_R2[i - 1][j] + u_R2[i][j] * dt
            x_R2[i][j] = x_R2[i - 1][j] + v_R2[i - 1][j] * dt + .5 * u_R2[i][j] * dt ** 2
            tf_R2[i][0] = t[i][j] + (cz_length - (x_R2[i][j]))/v_R2[i][j]
        
#        tf_data_one_timestep[0] = tf_R2[i][0]
#        for j in range(vehNum):
#            tf_data_one_timestep[j + 1] = tf[i][j]
        
        tf_data_one_timestep.append(tf_R2[i][0])
        for j in range(vehNum):
            tf_data_one_timestep.append(tf[i][j])
        tf_data_one_timestep.append(action)
        tf_data.append(tf_data_one_timestep)
        
        if x_R2[i][0] >= 398:
            break
#    totalFC = calcFC(x, v, u, dt, cz_length, vehNum)
#    totalFC_R2 = calcFC(x_R2, v_R2, u_R2, dt, cz_length, vehNum_R2)
#    totalFC_merged = calcFC(x_merged, v_merged, u_merged, dt, cz_length, vehNum + vehNum_R2)
    return t, x, v, u, tf, original_road, x_R2, v_R2, u_R2, tf_R2, i, tf_data

def writeFile(tf_data_cumulative):
    for s in range(len(tf_data_cumulative)):
        if s % 10 == 0:
            print("%d / %d" % (s, len(tf_data_cumulative)))
        for i in range(len(tf_data_cumulative[s])):
            if s == 0 and i == 0:
                sorted_data = np.array([tf_data_cumulative[s][i]])
            else:
                sorted_data = np.append(sorted_data, [tf_data_cumulative[s][i]], 0)
    np.savetxt("trial_5_1200_sims.txt", sorted_data)
    
#    real_data = np.loadtxt("trial_1.txt")
    print(len(sorted_data))
#    print("\n\n", real_data)
    pass

def nTrials():
    tf_data_cumulative = []
    for n in range(N):
        time, x, v, u, tf, original_road, x_R2, v_R2, u_R2, tf_R2, i, tf_data = mainAlg()
        tf_data_cumulative.append(tf_data)
        if PLOT:
            makePlots(time, x, v, u, tf, "merged road", vehNum, original_road, x_R2, v_R2, u_R2, tf_R2, i)
        if SIM:
            sim_main(x, x_R2)
    writeFile(tf_data_cumulative)



nTrials()