import gym
import gym_merge
from simulation import sim_main
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import l2
from sklearn.model_selection import train_test_split

STATE_SIZE = 3
ACTION_SIZE = 2
LEARNING_RATE = .00005

class NeuralNetwork:
    def __init__(self):
        self.network = Sequential()
        self.network.add(Dense(32, input_dim=3, activation="linear"))
        self.network.add(Dense(64, input_dim=32, activation="relu"))
        self.network.add(Dense(32, input_dim=64, activation="linear"))
        self.network.add(Dense(2, input_dim=32, activation="softmax"))
        self.network.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))
#        self.network = Sequential()
#        self.network.add(Dense(32, input_dim=STATE_SIZE, activation='relu'))
#        self.network.add(Dense(64, input_dim=STATE_SIZE, activation='relu'))
#        self.network.add(Dense(32, input_dim=STATE_SIZE, activation='linear'))
#        self.network.add(Dense(ACTION_SIZE, activation='softmax'))
#        self.network.compile(loss='categorical_crossentropy',
#                      optimizer=Adam(lr=LEARNING_RATE))
        self.network.summary()
        
    def train(self, state, action):
        for i in range(len(state)):
            for j in range(len(state[0])):
                state[i][j] = (state[i][j] - 15)/(60 - 15)
        self.network.fit(state, action, batch_size = 256, epochs = 50, verbose = 1)

def collect_data():
    state = []
    action = []
    data = np.loadtxt("trial_7_300_sims_static_state_two_cars_main_raod.txt")
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
    state_train, state_test = train_test_split(state, test_size=.2)
    action_train, action_test = train_test_split(action, test_size=.2)
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
#    accuracy = x.network.evaluate(state_test, action_test)
    accuracy = test(x, state_test, action_test)
    print("accuracy: ", accuracy)
#    x.network.save('model_3_3000_sims_5_layers_512_middle_layer_cross_entropy.h5')

main()