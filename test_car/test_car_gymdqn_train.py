from test_car_env import Test_car
env = Test_car(render=True)

#これらとpybulletのウィンドウ表示(p.connect)を分けないとカーネルが死ぬ
from baselines import deepq
import datetime


def callback(lcl, glb):
    # stop training if reward exceeds 199
    total = sum(lcl['episode_rewards'][-101:-1]) / 100
    totalt = lcl['t']
    is_solved = totalt > 2000 and total >= 20
    return is_solved

'''
cnn_to_mlp(convs, hiddens, dueling=False, layer_norm=False)
    - convs: [(int, int, int)]
        list of convolutional layers in form of
        (num_outputs, kernel_size, stride)
    - hiddens: [int]
        list of sizes of hidden layers
'''
model = deepq.models.cnn_to_mlp(convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)], hiddens=[256], dueling=False)

act = deepq.learn(
    env,
    q_func=model,
    lr=1e-2,
    max_timesteps=100000,
    buffer_size=50000,
    exploration_fraction=0.1,
    exploration_final_eps=0.02,
    print_freq=10,
    callback=callback)

print("Saving model to test_car_model.pkl")
act.save("test_car_model.pkl")

p.disconnect()
