import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
#from tensorflow import feature_column

EPOCHS = 1
LEARNING_RATE = .001

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(11, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 2)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x)
        

def collect_data():
    state = []
    action = []
    data = np.loadtxt("trial_5_1200_sims.txt")
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
#            for j in range(len(state[0])):
#                state[i][j] = (state[i][j] - 15)/(60 - 15)
    return state, action

def preprocessData(state, action):
#    column = []
#    for j in range(len(state[0])):
#        column.append(state[:, 0])
##    crossed_feature = feature_column.crossed_column([column[1], column[2]], hash_bucket_size = 5)
#    crossed_feature = feature_column.crossed_column([[1, 1, 0], [0, 0, 1]], hash_bucket_size = 3)
    
    train_state, test_state = train_test_split(state, test_size=0.2)
    train_state, val_state = train_test_split(train_state, test_size=0.2)
    
    train_action, test_action = train_test_split(action, test_size=.2)
    train_action, val_action = train_test_split(train_action, test_size=.2)
    return train_state, test_state, val_state, train_action, test_action, val_action
    
def main():
    state, action = collect_data()
    train_state, test_state, val_state, train_action, test_action, val_action = preprocessData(state, action)
    print(train_state.shape, test_shape.shape, val_state.shape, train_action.shape, test_action.shape, val_action.shape)
    x = NeuralNetwork()
#    optimizer = optim.SGD(net.parameters(), lr = LEARNING_RATE, momentum = .9)
#    criterion = nn.NLLLoss()
#    
#    for epoch in range(EPOCHS):
#        for batch_i in range(len(train)):
            

main()