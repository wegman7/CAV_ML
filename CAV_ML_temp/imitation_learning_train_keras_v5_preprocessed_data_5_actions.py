import gym
import gym_merge
from simulation import sim_main
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import l2
from sklearn.model_selection import train_test_split

STATE_SIZE = 5
ACTION_SIZE = 2
LEARNING_RATE = .0001

class NeuralNetwork:
    def __init__(self):
        self.network = Sequential()
        self.network.add(Dense(32, input_dim=STATE_SIZE, activation='linear'))
        self.network.add(Dense(ACTION_SIZE, input_dim = 32, activation='softmax'))
        self.network.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=LEARNING_RATE))
        self.network.summary()
        
    def train(self, state, action):
        self.network.fit(state, action, batch_size = 16, epochs = 6, verbose = 1)

def collect_data():
    state = []
    action = []
    data = np.loadtxt("trial_10_two_cars_300_sims.txt")
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
    tf_target = []
    for i in range(len(state)):
        tf_target.append([])
        for j in range(1, len(state[0]) - 1):
            tf_target[i].append((state[i][j] + state[i][j + 1])/2)
    
    for i in range(len(state)):
        if state[i][0] < tf_target[i][0] and  state[i][0] < (tf_target[i][0] + state[i][1])/2:
            state_new.append([1, 0, 0, 0, 0])
        elif state[i][0] < tf_target[i][0] - .05 and state[i][0] > (tf_target[i][0] + state[i][1])/2:
            state_new.append([0, 1, 0, 0, 0])
        elif state[i][0] > tf_target[i][0] + .05 and state[i][0] < (tf_target[i][0] + state[i][2])/2:
            state_new.append([0, 0, 1, 0, 0])
        elif state[i][0] > tf_target[i][0] and state[i][0] > (tf_target[i][0] + state[i][2])/2:
            state_new.append([0, 0, 0, 1, 0])
        elif state[i][0] > tf_target[i][0] - .05 and state[i][0] < tf_target[i][0] + .05:
            state_new.append([0, 0, 0, 0, 1])
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
    x.network.save('imitation_two_cars.h5')

main()