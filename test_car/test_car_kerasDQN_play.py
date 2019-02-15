# coding: utf-8
from test_car.test_car_env import Test_car
env = Test_car(render=False)


from keras.models import Sequential
from keras.layers import InputLayer, Dense, Reshape, Conv2D, Flatten, MaxPooling2D, BatchNormalization, Dropout
from keras.optimizers import RMSprop

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

nb_actions = 4

model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(64, activation='relu'))
model.add(Dense(nb_actions, activation='relu'))
# print(model.summary())

memory = SequentialMemory(limit=100000, window_length=1)
policy = EpsGreedyQPolicy(eps=0.1)
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10, target_model_update=1e-2, policy=policy)
dqn.compile(optimizer='RMSprop', metrics=['mae'])

dqn.load_weights('dqn_{}_weights.h5f'.format("test_car-v0"))

# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=5, , nb_max_episode_steps=100, visualize=True)
