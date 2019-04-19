# coding: utf-8
# import os
# os.chdir('test_car/')
from test_car_env import Test_car
env = Test_car(render=False, time_steps=100, num_max_steps=50) # この位置じゃないとエラー出る
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.optimizers import RMSprop, Adam
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy, BoltzmannQPolicy
from rl.memory import SequentialMemory

nb_actions = env.action_space.n

model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(nb_actions, activation='linear'))
# print(model.summary())

memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10, target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
# print(dqn.get_config())
history = dqn.fit(env, nb_steps=2000000, nb_max_episode_steps=51,  verbose=2)

# After training is done, we save the final weights.
dqn.save_weights('dqn_{}_weights.h5f'.format("test_car-v3"), overwrite=True)

# save reward
with open('results_test_car_v3.csv', 'w') as f:
    f.write('episode,reward\n')
    for i, rew in enumerate(history.history['episode_reward']):
        f.write(str(i)+','+str(rew)+'\n')

# show episode reward
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('results_test_car_v3.csv')
reward = df['reward'].to_list()
episode = df['episode'].to_list()

plt.plot(episode, reward, '.')
plt.xlabel("episode")
plt.ylabel("rewards")
plt.show()
