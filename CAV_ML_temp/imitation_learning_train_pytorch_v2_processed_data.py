import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torch.autograd import Variable
from sklearn.model_selection import train_test_split
#from tensorflow import feature_column

EPOCHS = 1
LEARNING_RATE = .00001

class NeuralNetwork(nn.Module):
    def __init__(self):
#        nn.Module.__init__(self)
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(20, 32)
#        self.fc1 = nn.Linear(11, 32)
        self.fc2 = nn.Linear(32, 32)
#        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
#        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x))
        return x

def collect_data():
    state = []
    action = []
    data = np.loadtxt("trial_6_1200_sims_static_state.txt")
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

def preprocessData(state, action):
    state_new = [0 for i in range(len(state))]
    for i in range(len(state)):
        if state[i][0] > state[i][1] and state[i][0] < state[i][2] and state[i][0] < (state[i][1] + state[i][2])/2:
            state_new[i] = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        elif state[i][0] > state[i][1] and state[i][0] < state[i][2] and state[i][0] > (state[i][1] + state[i][2])/2:
            state_new[i] = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        
        elif state[i][0] > state[i][2] and state[i][0] < state[i][3] and state[i][0] < (state[i][2] + state[i][3])/2:
            state_new[i] = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        elif state[i][0] > state[i][2] and state[i][0] < state[i][3] and state[i][0] > (state[i][2] + state[i][3])/2:
            state_new[i] = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            
        elif state[i][0] > state[i][3] and state[i][0] < state[i][4] and state[i][0] < (state[i][3] + state[i][4])/2:
            state_new[i] = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        elif state[i][0] > state[i][3] and state[i][0] < state[i][4] and state[i][0] > (state[i][3] + state[i][4])/2:
            state_new[i] = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            
        elif state[i][0] > state[i][4] and state[i][0] < state[i][5] and state[i][0] < (state[i][4] + state[i][5])/2:
            state_new[i] = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        elif state[i][0] > state[i][4] and state[i][0] < state[i][5] and state[i][0] > (state[i][4] + state[i][5])/2:
            state_new[i] = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            
        elif state[i][0] > state[i][5] and state[i][0] < state[i][6] and state[i][0] < (state[i][5] + state[i][6])/2:
            state_new[i] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        elif state[i][0] > state[i][5] and state[i][0] < state[i][6] and state[i][0] > (state[i][5] + state[i][6])/2:
            state_new[i] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            
        elif state[i][0] > state[i][6] and state[i][0] < state[i][7] and state[i][0] < (state[i][6] + state[i][7])/2:
            state_new[i] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        elif state[i][0] > state[i][6] and state[i][0] < state[i][7] and state[i][0] > (state[i][6] + state[i][7])/2:
            state_new[i] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
            
        elif state[i][0] > state[i][7] and state[i][0] < state[i][8] and state[i][0] < (state[i][7] + state[i][8])/2:
            state_new[i] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        elif state[i][0] > state[i][7] and state[i][0] < state[i][8] and state[i][0] > (state[i][7] + state[i][8])/2:
            state_new[i] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
            
        elif state[i][0] > state[i][8] and state[i][0] < state[i][9] and state[i][0] < (state[i][8] + state[i][9])/2:
            state_new[i] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        elif state[i][0] > state[i][8] and state[i][0] < state[i][9] and state[i][0] > (state[i][8] + state[i][9])/2:
            state_new[i] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
            
        elif state[i][0] > state[i][9] and state[i][0] < state[i][10] and state[i][0] < (state[i][9] + state[i][10])/2:
            state_new[i] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
        elif state[i][0] > state[i][9] and state[i][0] < state[i][10] and state[i][0] > (state[i][9] + state[i][10])/2:
            state_new[i] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
            
        elif state[i][0] > state[i][10] and state[i][0] < state[i][11] and state[i][0] < (state[i][10] + state[i][11])/2:
            state_new[i] = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
        elif state[i][0] > state[i][10] and state[i][0] < state[i][11] and state[i][0] > (state[i][10] + state[i][11])/2:
            state_new[i] = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        else:
            print("fix this")

    state_new = np.array(state_new)
    train_state, test_state = train_test_split(state_new, test_size=.2)
#    train_state, test_state = train_test_split(state, test_size=.2)
    train_action, test_action = train_test_split(action, test_size=.2)
    
    train_state = train_state.astype(dtype = 'float32')
    train_state = torch.from_numpy(train_state)
    train_action = train_action.astype(dtype = 'float32')
    train_action = torch.from_numpy(train_action)
    
    test_state = test_state.astype(dtype = 'float32')
    test_state = torch.from_numpy(test_state)
    test_action = test_action.astype(dtype = 'float32')
    test_action = torch.from_numpy(test_action)
    return train_state, test_state, train_action, test_action
    
def main():
    state, action = collect_data()
    train_state, test_state, train_action, test_action = preprocessData(state, action)
    x = NeuralNetwork()
    print(x)
    optimizer = optim.SGD(x.parameters(), lr = LEARNING_RATE, momentum = .9)
    criterion = nn.MSELoss()
    
    # train network
    for epoch in range(EPOCHS):
        for batch_i in range(len(train_state)):
            optimizer.zero_grad()
            net_out = x(train_state[batch_i])
            loss = criterion(net_out, train_action[batch_i])
            loss.backward()
            optimizer.step()
            if batch_i % 1000 == 0:
                print("%d / %d" % (batch_i, len(train_state)))
                print("loss = ", loss)
    
    # test network
    total_correct = 0
    total_correct_alt = 0
    for i in range(len(test_state)):
        net_out = x(test_state[i])
        pred = net_out.data.max(0)[1]
        if pred.item() == 0:
            pred_alt = torch.tensor(1)
        else:
            pred_alt = torch.tensor(0)
        target = torch.argmax(test_action[i])
        if pred.item() == target.item():
            total_correct += 1
        if pred_alt.item() == target.item():
            total_correct_alt += 1
    
#    total_correct = total_correct.item()
#    total_correct_alt = total_correct_alt.item()
    print("total_correct = ", total_correct)
    print("total_correct_alt = ", total_correct_alt)
    print("accuracy: %f" % (total_correct/len(test_state)))
    print("accuracy_alt: %f" % (total_correct_alt/len(test_state)))

main()