# coding: utf-8
"""
An example of QT-Opt.
"""

import argparse
import copy
import json
import os
from pprint import pprint

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym

import machina as mc
from machina.pols import ArgmaxQfPol
from machina.algos import qtopt
from machina.vfuncs import DeterministicSAVfunc, CEMDeterministicSAVfunc
from machina.envs import GymEnv
from machina.traj import Traj
from machina.traj import epi_functional
from machina.samplers import EpiSampler
from machina import logger
from machina.utils import set_device, measure

from simple_net import QTOptNet, FlattenedObservationWrapper, QNetCNN


parser = argparse.ArgumentParser()
parser.add_argument('--log', type=str, default='garbage',
                    help='Directory name of log.')
parser.add_argument('--env_name', type=str,
                    default='bulletKukaCam', help='Name of environment.')
parser.add_argument('--record', action='store_true',
                    default=False, help='If True, movie is saved.')
parser.add_argument('--seed', type=int, default=256)
parser.add_argument('--max_epis', type=int,
                    default=100000000, help='Number of episodes to run.')
parser.add_argument('--max_steps_off', type=int,
                    default=1000000000000, help='Number of episodes stored in off traj.')
parser.add_argument('--num_parallel', type=int, default=4,
                    help='Number of processes to sample.')
parser.add_argument('--cuda', type=int, default=-1, help='cuda device number.')
parser.add_argument('--data_parallel', action='store_true', default=False,
                    help='If True, inference is done in parallel on gpus.')
parser.add_argument('--max_steps_per_iter', type=int, default=20,
                    help='Number of steps to use in an iteration.')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--qf_lr', type=float, default=1e-3,
                    help='Q function learning rate.')
parser.add_argument('--tau', type=float, default=0.0001,
                    help='Coefficient of target function.')
parser.add_argument('--gamma', type=float, default=0.9,
                    help='Discount factor.')
parser.add_argument('--lag', type=int, default=6000,
                    help='Lag of gradient steps of target function2.')
parser.add_argument('--num_iter', type=int, default=2,
                    help='Number of iteration of CEM.')
parser.add_argument('--num_sampling', type=int, default=60,
                    help='Number of samples sampled from Gaussian in CEM.')
parser.add_argument('--num_best_sampling', type=int, default=6,
                    help='Number of best samples used for fitting Gaussian in CEM.')
parser.add_argument('--multivari', action='store_true',
                    help='If true, Gaussian with diagonal covarince instead of Multivariate Gaussian matrix is used in CEM.')
parser.add_argument('--eps', type=float, default=0.2,
                    help='Probability of random action in epsilon-greedy policy.')
parser.add_argument('--loss_type', type=str,
                    choices=['mse', 'bce'], default='mse',
                    help='Choice for type of belleman loss.')
parser.add_argument('--save_memory', action='store_true',
                    help='If true, save memory while need more computation time by for-sentence.')
args = parser.parse_args()


torch.backends.cudnn.enabled = False

# logフォルダ確保
if not os.path.exists(args.log):
    os.mkdir(args.log)
# 引数のパラメータを保存
with open(os.path.join(args.log, 'args.json'), 'w') as f:
    json.dump(vars(args), f)
pprint(vars(args))

# modelの保存場所確保
if not os.path.exists(os.path.join(args.log, 'models')):
    os.mkdir(os.path.join(args.log, 'models'))

# 乱数の種固定
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# GPU or CPU
device_name = 'cpu' if args.cuda < 0 else "cuda:{}".format(args.cuda)
device = torch.device(device_name)
set_device(device)

# logのcsvファイル確保
score_file = os.path.join(args.log, 'progress.csv')
logger.add_tabular_output(score_file)

# Gymのenviromentを生成
from pybullet_envs.bullet.kukaCamGymEnv import KukaCamGymEnv
env = KukaCamGymEnv(renders=False, isDiscrete=False)  # renders=Trueだとmachinaのtrain進まない。相性が悪い？
env = FlattenedObservationWrapper(env)
flattend_observation_space = env.flattend_observation_space

env = GymEnv(env, log_dir=os.path.join(args.log, 'movie'), record_video=args.record)
env.env.seed(args.seed)

# 観測と行動の次元
observation_space = env.observation_space
action_space = env.action_space
print('obs: {0}, act: {1}'.format(observation_space, action_space))

# Q-Network
print('Qnet')
qf_net = QTOptNet(observation_space, action_space)
qf = DeterministicSAVfunc(flattend_observation_space, action_space, qf_net, data_parallel=args.data_parallel) # 決定的行動状態価値関数?q-netの出力の形を少し整える

# target Q network theta1
print('target1_net')
targ_qf1_net = QTOptNet(observation_space, action_space)
targ_qf1_net.load_state_dict(qf_net.state_dict()) # model（重み）をロード(q-netからコピー)
targ_qf1 = CEMDeterministicSAVfunc(
        flattend_observation_space, action_space,
        targ_qf1_net, num_sampling=args.num_sampling,
        num_best_sampling=args.num_best_sampling, num_iter=args.num_iter, multivari=args.multivari,
        data_parallel=args.data_parallel, save_memory=args.save_memory) #CrossEntropy Methodよくわからん

# lagged network
print('lagged_net')
lagged_qf_net = QTOptNet(observation_space, action_space)
lagged_qf_net.load_state_dict(qf_net.state_dict()) # model（重み）をロード(theta1からコピー)
lagged_qf = DeterministicSAVfunc(flattend_observation_space, action_space, lagged_qf_net, data_parallel=args.data_parallel)

# target network theta2
print('target2_net')
targ_qf2_net = QTOptNet(observation_space, action_space)
targ_qf2_net.load_state_dict(lagged_qf_net.state_dict()) # model（重み）をロード(6000stepsごとにlagged netからコピー)
targ_qf2 = DeterministicSAVfunc(flattend_observation_space, action_space, targ_qf2_net, data_parallel=args.data_parallel)

# q-networkの最適化手法
print('optimizer')
optim_qf = torch.optim.Adam(qf_net.parameters(), args.qf_lr)

# epsilon-greedy policy
print('Policy')
pol = ArgmaxQfPol(flattend_observation_space, action_space, targ_qf1, eps=args.eps)

# replay bufferからサンプリング?
print('sampler')
sampler = EpiSampler(env, pol, num_parallel=args.num_parallel, seed=args.seed)

# off-policy experience. Traj=(s,a,r,s')
off_traj = Traj(args.max_steps_off, traj_device='cpu')

total_epi = 0
total_step = 0
total_grad_step = 0 # パラメータ更新回数
num_update_lagged = 0 # lagged netの更新回数
max_rew = -1000

print('start')
while args.max_epis > total_epi:
    with measure('sample'):
        print('sampling')
        # policyにしたがって行動し、経験を貯める（env.stepをone_epiの__init__内で行っている）
        # off-policy
        epis = sampler.sample(pol, max_steps=args.max_steps_per_iter)
    with measure('train'):
        # on-policyのサンプリング
        print('on-policy')
        on_traj = Traj(traj_device='cpu')
        on_traj.add_epis(epis)
        on_traj = epi_functional.add_next_obs(on_traj)
        on_traj.register_epis()
        off_traj.add_traj(on_traj)  # off-policyに加える

        # episodeとstepのカウント
        total_epi += on_traj.num_epi
        step = on_traj.num_step
        total_step += step
        epoch = step

        if args.data_parallel:
            qf.dp_run = True
            lagged_qf.dp_run = True
            targ_qf1.dp_run = True
            targ_qf2.dp_run = True
        # train
        print('train')
        result_dict = qtopt.train(
            off_traj, qf, lagged_qf, targ_qf1, targ_qf2, optim_qf,
            epoch, args.batch_size, args.tau, args.gamma,
            loss_type=args.loss_type)

        # multi-agent並列処理。dp_run=data_parallel run
        if args.data_parallel:
            qf.dp_run = False
            lagged_qf.dp_run = False
            targ_qf1.dp_run = False
            targ_qf2.dp_run = False

    total_grad_step += epoch
    if total_grad_step >= args.lag * num_update_lagged: # 6000stepsごとにlagged netを更新
        logger.log('Updated lagged qf!!')
        lagged_qf_net.load_state_dict(qf_net.state_dict())
        num_update_lagged += 1

    rewards = [np.sum(epi['rews']) for epi in epis]
    mean_rew = np.mean(rewards)
    # logを保存
    logger.record_results(args.log, result_dict, score_file,
                          total_epi, step, total_step,
                          rewards,
                          plot_title=args.env_name)

    if mean_rew > max_rew: # 報酬の最大値が更新されたら保存
        # policy
        torch.save(pol.state_dict(), os.path.join(args.log, 'models', 'pol_max.pkl'))
        # Q関数
        torch.save(qf.state_dict(), os.path.join(args.log, 'models',  'qf_max.pkl'))
        # target Q theta1
        torch.save(targ_qf1.state_dict(), os.path.join(args.log, 'models', 'targ_qf1_max.pkl'))
        # target Q theta 2
        torch.save(targ_qf2.state_dict(), os.path.join(args.log, 'models', 'targ_qf2_max.pkl'))
        # 訓練パラメータ
        torch.save(optim_qf.state_dict(), os.path.join(args.log, 'models', 'optim_qf_max.pkl'))
        # 得られた報酬記録を更新
        max_rew = mean_rew

    # 最後のepisodeのmodelを保存
    torch.save(pol.state_dict(), os.path.join(args.log, 'models', 'pol_last.pkl'))
    torch.save(qf.state_dict(), os.path.join(args.log, 'models', 'qf_last.pkl'))
    torch.save(targ_qf1.state_dict(), os.path.join(args.log, 'models', 'targ_qf1_last.pkl'))
    torch.save(targ_qf2.state_dict(), os.path.join(args.log, 'models', 'targ_qf2_last.pkl'))
    torch.save(optim_qf.state_dict(), os.path.join(args.log, 'models',  'optim_qf_last.pkl'))
    # 初期化
    del on_traj
del sampler
env.close()
