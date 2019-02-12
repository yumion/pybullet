# coding: utf-8
from test_car_env import Test_car

from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

nb_actions = 4

env = Test_car(render=False)

model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(64, activation='relu'))
model.add(Dense(nb_actions, activation='linear'))

# print(model.summary())


memory = SequentialMemory(limit=100000, window_length=1)
policy = EpsGreedyQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

dqn.fit(env, nb_steps=100000, visualize=True, verbose=0)

# After training is done, we save the final weights.
dqn.save_weights('dqn_{}_weights.h5f'.format("test_car-v0"), overwrite=True)
