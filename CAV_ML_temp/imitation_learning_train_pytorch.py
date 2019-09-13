import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torch.autograd import Variable
from sklearn.model_selection import train_test_split
#from tensorflow import feature_column

EPOCHS = 1
LEARNING_RATE = .0001

class NeuralNetwork(nn.Module):
    def __init__(self):
#        nn.Module.__init__(self)
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(11, 32)
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
    
    train_state = train_state.astype(dtype = 'float32')
    train_state = torch.from_numpy(train_state)
    train_action = train_action.astype(dtype = 'float32')
    train_action = torch.from_numpy(train_action)
    
    test_state = test_state.astype(dtype = 'float32')
    test_state = torch.from_numpy(test_state)
    test_action = test_action.astype(dtype = 'float32')
    test_action = torch.from_numpy(test_action)
    return train_state, test_state, val_state, train_action, test_action, val_action
    
def main():
    state, action = collect_data()
    train_state, test_state, val_state, train_action, test_action, val_action = preprocessData(state, action)
    x = NeuralNetwork()
    print(x)
    optimizer = optim.SGD(x.parameters(), lr = LEARNING_RATE, momentum = .9)
    criterion = nn.MSELoss()
    
    # train network
    for epoch in range(EPOCHS):
        for batch_i in range(80000):
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
    for i in range(10000):
        net_out = x(test_state[i])
#        print(net_out)
        pred = net_out.data.max(0)[1]
#        print("pred before = ", pred)
        if pred.item() == 0:
            pred_alt = torch.tensor(1)
        else:
            pred_alt = torch.tensor(0)
#        print("pred after = ", pred)
        target = torch.argmax(test_action[i])
        if pred.item() == target.item():
            total_correct += 1
        if pred_alt.item() == target.item():
            total_correct_alt += 1
#        total_correct += torch.eq(pred, target)
#        total_correct_alt += torch.eq(pred_alt, target)
#    total_correct = total_correct.item()
#    total_correct_alt = total_correct_alt.item()
    print("total_correct = ", total_correct)
    print("total_correct_alt = ", total_correct_alt)
    print("accuracy: %f" % (total_correct/10000))
    print("accuracy_alt: %f" % (total_correct_alt/10000))

main()