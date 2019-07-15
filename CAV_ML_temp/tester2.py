import numpy as np
#from keras.models import load_model
#
#def collect_data():
#    state = []
#    action = []
#    data = np.loadtxt("trial_2.txt")
#    print(data.shape)
#    for i in range(len(data)):
#        state_one_timestep = []
#        for j in range(len(data[0]) - 1):
#            state_one_timestep.append(data[i][j])
#        state.append([state_one_timestep])
#        if data[i][len(data[0]) - 1] == .2:
#            action.append([[1, 0]])
#        else:
#            action.append([[0, 1]])
##        action[0].append(data[i][len(data[0]) - 1])
#    state = np.array(state)
#    action = np.array(action)
#    return state, action
#
#model = load_model('model_1.h5')
#state, action = collect_data()
#print(state[2])
#
#test = model.predict(state[2])
#print(test)

x = np.array([1, 2, 3, 4, 10, 0, 0])
print(np.argmax(x))