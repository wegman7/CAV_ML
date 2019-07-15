import gym
import gym_merge
from simulation import sim_main
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

STATE_SIZE = 11
ACTION_SIZE = 2
LEARNING_RATE = .2

class NeuralNetwork:
    def __init__(self):
        self.network = Sequential()
        self.network.add(Dense(128, input_dim=STATE_SIZE, activation='relu',
                        kernel_initializer='he_uniform'))
        self.network.add(Dense(256, input_dim=STATE_SIZE, activation='relu',
                        kernel_initializer='he_uniform'))
        self.network.add(Dense(512, input_dim=STATE_SIZE, activation='relu',
                        kernel_initializer='he_uniform'))
        self.network.add(Dense(256, input_dim=STATE_SIZE, activation='relu',
                        kernel_initializer='he_uniform'))
        self.network.add(Dense(128, input_dim=STATE_SIZE, activation='relu',
                        kernel_initializer='he_uniform'))
        self.network.add(Dense(ACTION_SIZE, activation='softmax',
                        kernel_initializer='he_uniform'))
        self.network.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=LEARNING_RATE))

    def train(self, state, action):
#        print(state)
#        input()
        for i in range(len(state)):
            min_state_value = min(state[i][0])
            max_state_value = max(state[i][0])
#            print(state[i][0])
#            print(max(state[i][0]))
            for j in range(len(state[0])):
                state[i][j] = (state[i][j] - min_state_value)/(max_state_value - min_state_value)
#            print(state[i])
            self.network.fit(state[i], action[i], verbose = 0)
            if i % 1000 == 0:
                print("%d / %d" % (i, len(state)))

def collect_data():
    state = []
    action = []
    data = np.loadtxt("trial_5_1200_sims.txt")
    print(data.shape)
    for i in range(len(data)):
        state_one_timestep = []
        for j in range(len(data[0]) - 1):
            state_one_timestep.append(data[i][j])
        state.append([state_one_timestep])
        if data[i][len(data[0]) - 1] == .2:
            action.append([[1, 0]])
        else:
            action.append([[0, 1]])
#        action[0].append(data[i][len(data[0]) - 1])
    state = np.array(state)
    action = np.array(action)
    return state, action
    
def main():
    state, action = collect_data()
#    print(len(state))
#    print(action)
    
    x = NeuralNetwork()
    x.train(state, action)
    x.network.save('model_3_3000_sims_5_layers_512_middle_layer_cross_entropy.h5')

main()