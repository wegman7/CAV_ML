import gym
import gym_merge
import numpy as np
import matplotlib.pyplot as plt
#from keras.layers import Dense
#from keras.models import Sequential
#from keras.optimizers import Adam
from keras.models import load_model

render = True

class NeuralNetwork():
    def  __init__(self):
        self.network = load_model('imitation_10_cars.h5')
    
    def predict(self, state):
        action = self.network.predict(state)
        return action

def preProcessData(state):
    state_new = []
    if state[0] > state[1] and state[0] < state[2] and state[0] < (state[1] + state[2])/2:
        state_new.append([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    elif state[0] > state[1] and state[0] < state[2] and state[0] > (state[1] + state[2])/2:
        state_new.append([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    
    elif state[0] > state[2] and state[0] < state[3] and state[0] < (state[2] + state[3])/2:
        state_new.append([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    elif state[0] > state[2] and state[0] < state[3] and state[0] > (state[2] + state[3])/2:
        state_new.append([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        
    elif state[0] > state[3] and state[0] < state[4] and state[0] < (state[3] + state[4])/2:
        state_new.append([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    elif state[0] > state[3] and state[0] < state[4] and state[0] > (state[3] + state[4])/2:
        state_new.append([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        
    elif state[0] > state[4] and state[0] < state[5] and state[0] < (state[4] + state[5])/2:
        state_new.append([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    elif state[0] > state[4] and state[0] < state[5] and state[0] > (state[4] + state[5])/2:
        state_new.append([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        
    elif state[0] > state[5] and state[0] < state[6] and state[0] < (state[5] + state[6])/2:
        state_new.append([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    elif state[0] > state[5] and state[0] < state[6] and state[0] > (state[5] + state[6])/2:
        state_new.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        
    elif state[0] > state[6] and state[0] < state[7] and state[0] < (state[6] + state[7])/2:
        state_new.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    elif state[0] > state[6] and state[0] < state[7] and state[0] > (state[6] + state[7])/2:
        state_new.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        
    elif state[0] > state[7] and state[0] < state[8] and state[0] < (state[7] + state[8])/2:
        state_new.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    elif state[0] > state[7] and state[0] < state[8] and state[0] > (state[7] + state[8])/2:
        state_new.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        
    elif state[0] > state[8] and state[0] < state[9] and state[0] < (state[8] + state[9])/2:
        state_new.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    elif state[0] > state[8] and state[0] < state[9] and state[0] > (state[8] + state[9])/2:
        state_new.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
        
    elif state[0] > state[9] and state[0] < state[10] and state[0] < (state[9] + state[10])/2:
        state_new.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
    elif state[0] > state[9] and state[0] < state[10] and state[0] > (state[9] + state[10])/2:
        state_new.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
        
    elif state[0] > state[10] and state[0] < state[11] and state[0] < (state[10] + state[11])/2:
        state_new.append([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
    elif state[0] > state[10] and state[0] < state[11] and state[0] > (state[10] + state[11])/2:
        state_new.append([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    else:
        print("fix this")
    state_new = np.array(state_new)
    return state_new

def merge():
    env = gym.make('merge-v0')
    model = NeuralNetwork()
    while True:
        state = env.reset()
        state = preProcessData(state)
        step_axis = []
        score_axis = []
        step = 0
        while True:
            step += 1
            action = np.argmax(model.network.predict(state))
            state, reward, terminal = env.step(action)
            state = preProcessData(state)
            step_axis.append(step)
            score_axis.append(reward)
            if terminal:
                if render:
                    env.render()
                plt.plot(step_axis, score_axis)
                plt.xlabel("steps")
                plt.ylabel("reward")
                plt.show()
                break

merge()