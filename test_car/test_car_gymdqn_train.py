# coding: utf-8
from test_car_env import Test_car
env = Test_car(render=False, time_steps=100, num_max_steps=50)

#これらとpybulletのウィンドウ表示(p.connect)を分けないとカーネルが死ぬ
from baselines import deepq
import datetime

def callback(lcl, glb):
    total = sum(lcl['episode_rewards'][-501:-1]) / 50
    is_solved = total>=0.9
    return is_solved

'''
cnn_to_mlp(convs, hiddens, dueling=False, layer_norm=False)
    - convs: [(int, int, int)]
        list of convolutional layers in form of
        (num_outputs, kernel_size, stride)
    - hiddens: [int]
        list of sizes of hidden layers
'''
model = deepq.models.mlp(hiddens=[64])

act = deepq.learn(
    env,
    q_func=model,
    lr=1e-2,
    max_timesteps=100000,
    buffer_size=10000,
    exploration_fraction=0.1,
    exploration_final_eps=0.02,
    print_freq=1)

print("Saving model to test_car_model.pkl")
act.save("test_car_model.pkl")

p.disconnect()
