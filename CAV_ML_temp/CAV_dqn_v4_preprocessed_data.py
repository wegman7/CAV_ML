import random
import gym
import gym_merge
import numpy as np
import matplotlib.pyplot as plt
import datetime
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import load_model


ENV_NAME = "merge-v0"

GAMMA = 1
LEARNING_RATE = 0.00001

MEMORY_SIZE = 1000000
BATCH_SIZE = 8

EXPLORATION_MAX = .5
EXPLORATION_MIN = 0.1
EXPLORATION_DECAY = 0.99999
render = True


class DQNSolver:

    def __init__(self, observation_space, action_space):
        self.exploration_rate = EXPLORATION_MAX

        self.memory = deque(maxlen=MEMORY_SIZE)
#        self.model = Sequential()
#        self.model.add(Dense(32, input_shape=(2,), activation="relu"))
#        self.model.add(Dense(32, activation="relu"))
#        self.model.add(Dense(2, activation="softmax"))
#        self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))
        self.model = load_model('imitation_10_cars.h5')

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            # agent acts randomly
            return random.randrange(2)
        # predict the reward value based on current state
        q_values = self.model.predict(state)
        # return the best action based on predicted reward
        return np.argmax(q_values[0])

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        # sample minibatch from the memory
        batch = random.sample(self.memory, BATCH_SIZE)
        # extract information from each memory
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                # if not terminal, predect the future discounted reward
                q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))
            # make agent approximately map the current state of future discounted reward, called q_values
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            # train the neural net with the state and q_values
            self.model.fit(state, q_values, batch_size = 32, verbose=0)
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

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
        state_new.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
    elif state[0] > state[10] and state[0] < state[11] and state[0] > (state[10] + state[11])/2:
        state_new.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    else:
        print("fix this", state[0])
    state_new = np.array(state_new)
    return state_new

def printTimesteps(state, run, reward, dqn_solver, step_axis, score_axis, env):
    print("\n\n\nRun: %d, exploration: %.2f, score: %.2f" % (run, dqn_solver.exploration_rate, reward))
    print("state = ", state)
    print("two layers of 32 nodes, learning rate .001, gamma = 1.0, exploration decay = .999")
#    if run % 20 == 0:
#        plt.plot(step_axis, score_axis)
#        plt.xlabel("steps")
#        plt.ylabel("reward")
#        plt.show()
    if render:
        env.render()
    step_axis = []
    score_axis = []
    pass

def printEpochs(epoch_axis, reward_axis, average_reward_axis, run):
    n = 50
    if run >= n:
        last_n = reward_axis[-n:]
        last_n_average = sum(last_n)/n
        average_reward_axis.append(last_n_average)
    if run % 100 == 0:
        plt.plot(epoch_axis[-(run - (n - 1)):], average_reward_axis, label = 'average reward')
#        plt.plot(epoch_axis, reward_axis, label = 'reward')
        plt.xlabel("epochs")
        plt.ylabel("reward")
        plt.legend()
        plt.show()
#    return average_reward_axis

def merge():
    env = gym.make(ENV_NAME)
    observation_space = 20
    action_space = 2
    dqn_solver = DQNSolver(observation_space, action_space)
    epoch_axis = []
    reward_axis = []
    
    step_axis = []
    score_axis = []
    average_reward_axis = []
    run = 0
    while True:
        run += 1
        state = env.reset()
        state = preProcessData(state)
        state = np.reshape(state, [1, observation_space])
        step = 0
        while True:
            step += 1
            action = dqn_solver.act(state) # .00017
            state_next, reward, terminal = env.step(action) # .00049
            # if our car goes outside of the "bounds" of the cars on the main road
            if state_next[0] < state_next[1] or state_next[0] > state_next[2]:
                printTimesteps(state, run, reward, dqn_solver, step_axis, score_axis, env)
                break
            state_next = preProcessData(state_next)
            state_next = np.reshape(state_next, [1, observation_space])
            dqn_solver.remember(state, action, reward, state_next, terminal) # .000007
            state = state_next
            dqn_solver.experience_replay() # .077
            step_axis.append(step)
            score_axis.append(reward)
            if terminal:
                printTimesteps(state, run, reward, dqn_solver, step_axis, score_axis, env)
                break
        epoch_axis.append(run)
        reward_axis.append(reward)
        printEpochs(epoch_axis, reward_axis, average_reward_axis, run)
#    dqn_solver.model.save('model_3_11.h5')

if __name__ == "__main__":
    merge()