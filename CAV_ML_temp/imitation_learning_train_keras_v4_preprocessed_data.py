import gym
import gym_merge
from simulation import sim_main
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import l2
from sklearn.model_selection import train_test_split

STATE_SIZE = 20
ACTION_SIZE = 2
LEARNING_RATE = .0001

class NeuralNetwork:
    def __init__(self):
#        self.network = Sequential()
#        self.network.add(Dense(32, input_shape=(2,), activation="relu"))
#        self.network.add(Dense(32, activation="relu"))
#        self.network.add(Dense(2, activation="linear"))
#        self.network.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))
        self.network = Sequential()
        self.network.add(Dense(32, input_dim=STATE_SIZE, activation='linear'))
#        self.network.add(Dense(64, input_dim=STATE_SIZE, activation='relu'))
#        self.network.add(Dense(32, input_dim=STATE_SIZE, activation='linear'))
        self.network.add(Dense(ACTION_SIZE, input_dim = 32, activation='softmax'))
        self.network.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=LEARNING_RATE))
        self.network.summary()
        
    def train(self, state, action):
        self.network.fit(state, action, batch_size = 16, epochs = 6, verbose = 1)

def collect_data():
    state = []
    action = []
    data = np.loadtxt("trial_9_ten_cars_3000_sims.txt")
    print(data.shape)
    for i in range(len(data)):
        state_one_timestep = []
        for j in range(len(data[0]) - 1):
            state_one_timestep.append(data[i][j])
        state.append(state_one_timestep)
        if data[i][len(data[0]) - 1] == .2:
            action.append([1.0, 0.0])
        else:
            action.append([0.0, 1.0])
    state = np.array(state)
    action = np.array(action)
    # normalize data
#    for i in range(len(state)):
#        for j in range(len(state[0])):
#            state[i][j] = (state[i][j] - 15)/(60 - 15)
    return state, action

def preProcessData(state, action):
    state_new = []
    for i in range(len(state)):
        if state[i][0] > state[i][1] and state[i][0] < state[i][2] and state[i][0] < (state[i][1] + state[i][2])/2:
            state_new.append([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        elif state[i][0] > state[i][1] and state[i][0] < state[i][2] and state[i][0] > (state[i][1] + state[i][2])/2:
            state_new.append([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        
        elif state[i][0] > state[i][2] and state[i][0] < state[i][3] and state[i][0] < (state[i][2] + state[i][3])/2:
            state_new.append([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        elif state[i][0] > state[i][2] and state[i][0] < state[i][3] and state[i][0] > (state[i][2] + state[i][3])/2:
            state_new.append([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            
        elif state[i][0] > state[i][3] and state[i][0] < state[i][4] and state[i][0] < (state[i][3] + state[i][4])/2:
            state_new.append([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        elif state[i][0] > state[i][3] and state[i][0] < state[i][4] and state[i][0] > (state[i][3] + state[i][4])/2:
            state_new.append([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            
        elif state[i][0] > state[i][4] and state[i][0] < state[i][5] and state[i][0] < (state[i][4] + state[i][5])/2:
            state_new.append([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        elif state[i][0] > state[i][4] and state[i][0] < state[i][5] and state[i][0] > (state[i][4] + state[i][5])/2:
            state_new.append([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            
        elif state[i][0] > state[i][5] and state[i][0] < state[i][6] and state[i][0] < (state[i][5] + state[i][6])/2:
            state_new.append([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        elif state[i][0] > state[i][5] and state[i][0] < state[i][6] and state[i][0] > (state[i][5] + state[i][6])/2:
            state_new.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            
        elif state[i][0] > state[i][6] and state[i][0] < state[i][7] and state[i][0] < (state[i][6] + state[i][7])/2:
            state_new.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        elif state[i][0] > state[i][6] and state[i][0] < state[i][7] and state[i][0] > (state[i][6] + state[i][7])/2:
            state_new.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
            
        elif state[i][0] > state[i][7] and state[i][0] < state[i][8] and state[i][0] < (state[i][7] + state[i][8])/2:
            state_new.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        elif state[i][0] > state[i][7] and state[i][0] < state[i][8] and state[i][0] > (state[i][7] + state[i][8])/2:
            state_new.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
            
        elif state[i][0] > state[i][8] and state[i][0] < state[i][9] and state[i][0] < (state[i][8] + state[i][9])/2:
            state_new.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        elif state[i][0] > state[i][8] and state[i][0] < state[i][9] and state[i][0] > (state[i][8] + state[i][9])/2:
            state_new.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
            
        elif state[i][0] > state[i][9] and state[i][0] < state[i][10] and state[i][0] < (state[i][9] + state[i][10])/2:
            state_new.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
        elif state[i][0] > state[i][9] and state[i][0] < state[i][10] and state[i][0] > (state[i][9] + state[i][10])/2:
            state_new.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
            
        elif state[i][0] > state[i][10] and state[i][0] < state[i][11] and state[i][0] < (state[i][10] + state[i][11])/2:
            state_new.append([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
        elif state[i][0] > state[i][10] and state[i][0] < state[i][11] and state[i][0] > (state[i][10] + state[i][11])/2:
            state_new.append([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        else:
            print("fix this")
    state_new = np.array(state_new)
    state_train, state_test, action_train, action_test = train_test_split(state_new, action, test_size= .4, random_state = 1)
    return state_train, action_train, state_test, action_test

def test(x, state_test, action_test):
    predictions = x.network.predict(state_test)
    number_correct = 0
    for i in range(len(state_test)):
        if np.argmax(predictions[i]) == np.argmax(action_test[i]):
            number_correct += 1
    accuracy = number_correct/len(state_test)
    return accuracy
    

def main():
    state, action = collect_data()
    state_train, action_train, state_test, action_test = preProcessData(state, action)
    x = NeuralNetwork()
    x.train(state_train, action_train)
    accuracy = test(x, state_test, action_test)
    print("accuracy: ", accuracy)
    x.network.save('imitation_ten_cars.h5')

main()