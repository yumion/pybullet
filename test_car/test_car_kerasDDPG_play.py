# coding :utf-8
# import os
# os.chdir('test_car/')
from test_car_env_box_action import Test_car
env = Test_car(render=False, time_steps=10, num_max_steps=500)

import numpy as np
import gym
from keras.models import Model
from keras.layers import Dense, Flatten, Input, concatenate
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory


def build_actor_model(num_action, observation_shape):
    action_input = Input(shape=(1,)+observation_shape)
    x = Flatten()(action_input)
    x = Dense(16, activation="relu")(x)
    x = Dense(16, activation="relu")(x)
    x = Dense(16, activation="relu")(x)
    x = Dense(num_action, activation="linear")(x)
    actor = Model(inputs=action_input, outputs=x)
    return actor

def build_critic_model(num_action, observation_shape):
    action_input = Input(shape=(num_action,))
    observation_input = Input(shape=(1,)+observation_shape)
    flattened_observation = Flatten()(observation_input)
    x = concatenate([action_input, flattened_observation])
    x = Dense(32, activation="relu")(x)
    x = Dense(32, activation="relu")(x)
    x = Dense(32, activation="relu")(x)
    x = Dense(1, activation="linear")(x)
    critic = Model(inputs=[action_input, observation_input], outputs=x)
    return (critic, action_input)

def build_agent(num_action, observation_shape):
    actor = build_actor_model(num_action, observation_shape)
    critic, critic_action_input = build_critic_model(num_action, observation_shape)
    memory = SequentialMemory(limit=10**5, window_length=1)
    agent = DDPGAgent(
        num_action,
        actor,
        critic,
        critic_action_input,
        memory
    )
    return agent


agent = build_agent(env.action_space.shape[0], env.observation_space.shape)
agent.compile(optimizer="nadam", metrics=["mae"])
agent.load_weights('ddpg_{}_weights.h5f'.format("test_car-v0"))

# Finally, evaluate our algorithm for 5 episodes.
agent.test(env, nb_episodes=5, nb_max_episode_steps=50, visualize=True)
