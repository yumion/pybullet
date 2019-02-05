# coding: utf-8

'''
from pybullet_envs.bullet.racecarGymEnv import RacecarGymEnv

env = RacecarGymEnv(renders=False,isDiscrete=True)

#これらとpybulletのウィンドウ表示(p.connect)を分けないとカーネルが死ぬ
import gym
from baselines import deepq
import datetime


def callback(lcl, glb):
    # stop training if reward exceeds 199
    total = sum(lcl['episode_rewards'][-101:-1]) / 100
    totalt = lcl['t']
    is_solved = totalt > 2000 and total >= -50
    return is_solved

model = deepq.models.mlp([64])

act = deepq.learn(
    env,
    q_func=model,
    lr=1e-3,
    max_timesteps=10000,
    buffer_size=50000,
    exploration_fraction=0.1,
    exploration_final_eps=0.02,
    print_freq=10,
    callback=callback)

print("Saving model to racecar_model.pkl")
act.save("racecar_model.pkl")


'''
from pybullet_envs.bullet.racecarGymEnv import RacecarGymEnv
env = RacecarGymEnv(renders=True,isDiscrete=True)

from baselines import deepq
act = deepq.load("racecar_model.pkl")

episodes = 100
for episode in range(episodes):
    obs, done = env.reset(), False
    print("===episode:",episode,"=============================")
    print("obs: ", obs)
    episode_rew = 0
    while not done:
        env.render()
        obs, rew, done, _ = env.step(act(obs[None])[0])
        episode_rew += rew
    print("Episode reward: ", episode_rew)
