import random
import gym
import gym_merge
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import load_model
from simulation import sim_main

ENV_NAME = "merge-v0"

# maybe try a discount factor over 1.0, to help incentivise ending up further away from the other cars at the END of the simulation
GAMMA = 1
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000000
BATCH_SIZE = 64

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.0000001
EXPLORATION_DECAY = 0.999


class DQNSolver:

    def __init__(self, observation_space, action_space):
        self.exploration_rate = EXPLORATION_MAX

#        self.action_space = action_space
#        self.memory = deque(maxlen=MEMORY_SIZE)
        self.model = load_model('model_3_13.h5')
#        self.model.add(Dense(32, input_shape=(observation_space,), activation="relu"))
#        self.model.add(Dense(32, activation="relu"))
#        self.model.add(Dense(self.action_space, activation="linear"))
#        self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
#        if np.random.rand() < self.exploration_rate:
            # agent acts randomly
#            return random.randrange(self.action_space)
        # predict the reward value based on current state
        q_values = self.model.predict(state)
#        print("inside act method, self.model.predict = ", self.model.predict(state))
        # return the best action based on predicted reward
#        print("and argmax of self.model.predict = ", np.argmax(q_values[0]))
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
#            print("size of q_values = ", q_values.shape)
#            print("q_update = ", q_update)
#            print("q_values (target) = ", q_values)
#            print("state = ", state)
            # train the neural net with the state and q_values
            self.model.fit(state, q_values, verbose=0)
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)


def merge():
    env = gym.make(ENV_NAME)
    observation_space = 6
    action_space = 3
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
        state = np.reshape(state, [1, observation_space])
        step = 0
        while True:
            step += 1
#            env.render()
            action = dqn_solver.act(state)
            state_next, reward, terminal = env.step(action)
            state_next = np.reshape(state_next, [1, observation_space])
#            dqn_solver.remember(state, action, reward, state_next, terminal)
            state = state_next
            if terminal:
                print("\n\n\nRun: %d, exploration: %.2f, score: %.2f" % (run, dqn_solver.exploration_rate, reward))
                print("state = ", state)
                print("two layers of 32 nodes, learning rate .001, gamma = 1.0, exploration decay = .999")
                plt.plot(step_axis, score_axis)
                plt.xlabel("steps")
                plt.ylabel("reward")
                plt.show()
#                env.render()
                step_axis = []
                score_axis = []
#                sim_main(env.x, env.x_R2)
                break
#            dqn_solver.experience_replay()
            
            step_axis.append(step)
            score_axis.append(reward)
        epoch_axis.append(run)
        reward_axis.append(reward)
        
        if run >= 40:
            last_50 = reward_axis[-40:]
            last_50_average = sum(last_50)/40
            average_reward_axis.append(last_50_average)
#        if run % 1 == 0:
#            plt.plot(epoch_axis, reward_axis, label = 'reward')
#            plt.plot(epoch_axis[-(run - 39):], average_reward_axis, label = 'average reward')
#            plt.xlabel("epochs")
#            plt.ylabel("reward")
#            plt.legend()
#            plt.show()


if __name__ == "__main__":
    merge()