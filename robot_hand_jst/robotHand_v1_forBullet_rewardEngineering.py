# coding: utf-8
import numpy as np
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') # ROSと競合してOpenCVをimportできない
import cv2
import time
from datetime import datetime  # 時刻を取得
import pybullet as p
import pybullet_data
from PIL import Image
import os

'''定数の設定'''
NUM_DIZITIZED = 20  # 各状態の離散値への分割数
NUM_STATES = 2
NUM_ACTIONS = 4
discount = 0.9  # 時間割引率
lr = 0.01  # 学習係数
MAX_STEPS = 50  # 1試行のstep数
NUM_EPISODES = 10000  # 最大試行回数
AREA_THRESH = 10  # 赤色物体面積の閾値．0~100で規格化してある

'''学習するときはFalse，学習済みのモデルを使用するときはTrue'''
# 使うq_tableのファイル名を"trained_q_table.npy"とすること
TEST_MODE = False
'''追加学習するときはTrue'''
ADD_TRAIN_MODE = False

'''pybulletに描画するか'''
RENDER = False

class Agent:
    '''CartPoleのエージェントクラスです、棒付き台車そのものになります'''

    def __init__(self, num_states, num_actions):
        self.brain = Brain(num_states, num_actions)  # エージェントが行動を決定するための頭脳を生成

    def update_Q_function(self, observation, action, reward, observation_next):
        '''Q関数の更新'''
        self.brain.update_Q_table(observation, action, reward, observation_next)

    def get_action(self, observation, step, test=False):
        '''行動の決定'''
        action = self.brain.decide_action(observation, step, test)
        if RENDER: print('action: ', action)
        return action


class Brain:
    '''エージェントが持つ脳となるクラスです、Q学習を実行します'''

    def __init__(self, num_states, num_actions):
        self.num_actions = num_actions  # ロボットハンドの取れる行動数(コマンド数)
        if TEST_MODE or ADD_TRAIN_MODE:  # 保存したQ-tableを使用
            self.q_table = np.load('0507_q_table.npy')
        else:  # Qテーブルを作成。行数は状態を分割数^(4変数)にデジタル変換した値、列数は行動数を示す
            self.q_table = np.random.uniform(low=0, high=1, size=(NUM_DIZITIZED**num_states, num_actions))

    def bins(self, clip_min, clip_max, num):
        '''観測した状態（連続値）を離散値にデジタル変換する閾値を求める'''
        return np.linspace(clip_min, clip_max, num + 1)[1:-1]

    def digitize_state(self, observation):
        '''観測したobservation状態を、離散値に変換する'''
        targetPosInHand_x, targetPosInHand_y = observation
        digitized = [
            np.digitize(targetPosInHand_x, bins=self.bins(-1.0, 1.0, NUM_DIZITIZED)), #　面積の比率
            np.digitize(targetPosInHand_y, bins=self.bins(-1.0, 1.0, NUM_DIZITIZED))
        ]
        return sum([x * (NUM_DIZITIZED**i) for i, x in enumerate(digitized)]) #　6進数で表して計算を圧縮

    def update_Q_table(self, observation, action, reward, observation_next):
        '''QテーブルをQ学習により更新'''
        state = self.digitize_state(observation)  # 状態を離散化
        state_next = self.digitize_state(observation_next)  # 次の状態を離散化
        Max_Q_next = max(self.q_table[state_next][:])
        self.q_table[state, action] = self.q_table[state, action] + lr * (reward + discount * Max_Q_next - self.q_table[state, action])
        self.save_Q_table()  # Q-tableを更新するたびに保存

    def save_Q_table(self):
        '''学習したQ-tableを保存'''
        np.save(datetime.today().strftime("%m%d")+'_q_table', self.q_table)

    def load_Q_table(self):  # NoneTypeで読み込んでしまうため使ってない
        '''学習済みのQ-tableを読み込み'''
        np.load('trained_q_table.npy')

    def decide_action(self, observation, episode, test=False):
        '''ε-greedy法で徐々に最適行動のみを採用する'''
        state = self.digitize_state(observation)
        if episode < 10**4:
            epsilon = -(1 - 0.1) / 10**4 * episode + 1
        else:
            epsilon = 0.1

        if epsilon <= np.random.uniform(0, 1) or test:
            action = np.argmax(self.q_table[state][:])
        else:
            action = np.random.choice(self.num_actions)  # 行動をランダムに返す
        return action


class Environment:
    '''CartPoleを実行する環境のクラスです'''

    def __init__(self):
        self.num_states = NUM_STATES  # 課題の状態の数(面積と重心(x,y)と、それぞれの変化量で6つ)
        self.num_actions = NUM_ACTIONS  # ロボットハンドの行動（前進，後退，右旋回，左旋回，握る，離す，止まる）
        self.agent = Agent(self.num_states, self.num_actions)  # 環境内で行動するAgentを生成
        print('env')
        '''pybullet'''
        if RENDER:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        self.maxForce = 10

    def renderPicture(self, height=320, width=320):
        '''bullet側からカメラ画像を取得'''
        base_pos, orn = p.getBasePositionAndOrientation(self.hand)
        yaw = p.getEulerFromQuaternion(orn)[2] # z軸方向から見た本体の回転角度
        rot_matrix = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]) # 回転行列
        target_relative_vec2D = np.array([2,0]) # 本体から見たtargetの相対位置
        target_abs_vec2D = np.dot(rot_matrix, target_relative_vec2D) # targetの絶対位置

        cam_eye = np.array(base_pos) + np.array([-0.01,-0.020,0.020]) # カメラの座標
        cam_target = np.array(base_pos) + np.append(target_abs_vec2D,0.20) # カメラの焦点の座標（ターゲットの座標）、ｚ軸方向を変える。
        cam_upvec = [0,0,1]

        view_matrix = p.computeViewMatrix(
                cameraEyePosition=cam_eye,
                cameraTargetPosition=cam_target,
                cameraUpVector=cam_upvec)

        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=float(width)/height,
            nearVal=0.1, farVal=100.0)

        (_, _, rgb, _, mask) = p.getCameraImage(
            width=width, height=height, viewMatrix=view_matrix,
            projectionMatrix=proj_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)

        rgb_array = np.array(rgb)
        rgb_array = rgb_array[:,:,:3]
        # mask_array = np.array(mask)

        return rgb_array

    def red_detect(self, img):
        '''赤色のマスク'''
        # HSV色空間に変換
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # 赤色のHSVの値域1
        hsv_min = np.array([0, 127, 0])
        hsv_max = np.array([149, 255, 255])
        mask1 = cv2.inRange(hsv, hsv_min, hsv_max)
        # 赤色のHSVの値域2
        hsv_min = np.array([150, 127, 0])
        hsv_max = np.array([179, 255, 255])
        mask2 = cv2.inRange(hsv, hsv_min, hsv_max)
        return mask1 + mask2

    def calc_area(self, img):
        '''面積計算'''
        img = self.red_detect(img)
        pix_area = cv2.countNonZero(img)  # ピクセル数
        # パーセントを算出
        h, w = img.shape  # frameの面積
        per = round(100 * float(pix_area) / (w * h), 3)  # 0-100で規格化
        if RENDER:
            print('DETECT_AREA: ', per)
        return per

    def calc_center(self, img):
        '''重心座標(x,y)を求める'''
        img = self.red_detect(img)
        mu = cv2.moments(img, False)
        x, y = int(mu["m10"] / (mu["m00"] + 1e-7)), int(mu["m01"] / (mu["m00"] + 1e-7))
        # 重心を丸でくくる
        #cv2.circle(img, (x, y), 4, 100, 2, 4)
        return x, y

    def reset(self):
        '''環境を初期化する'''
        if RENDER:
            print('Environment.reset\n')

        #bulletの世界をリセット
        p.resetSimulation()

        #フィールドを表示
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0,0,-10)
        self.planeId = p.loadURDF("plane.urdf")

        #オブジェクトモデルを表示
        p.setAdditionalSearchPath(os.environ['HOME']+"/atsushi/catkin_ws/src/robotHand_v1/urdf/")
        startPos = [0,0,0.03]
        startOrientation = p.getQuaternionFromEuler([0,0,0])
        self.hand = p.loadURDF("smahoHand.urdf", startPos, startOrientation)
        # 摩擦係数を変更
        p.changeDynamics(self.hand, 4, lateralFriction=0) # 前輪ボール
        p.changeDynamics(self.hand, 2, lateralFriction=100)
        p.changeDynamics(self.hand, 3, lateralFriction=100)
        # ターゲットを表示
        for i in range(1):
            targetX, targetY = np.random.permutation(np.arange(-1,1,0.1))[0:2] # 2x2マス以内にランダムで配置
            targetPos = [targetX, targetY, 0.05]
            self.target = p.loadURDF("target.urdf", targetPos, startOrientation)

        return self.get_env()

    def get_env(self):
        '''環境を認識する'''
        '''カメラで写真をとりOpenCVで面積と重心を取得する'''
        frame = self.renderPicture()
        handpos, handorn = p.getBasePositionAndOrientation(self.hand)
        targetpos, targetorn = p.getBasePositionAndOrientation(self.target)
        invHandPos, invHandOrn = p.invertTransform(handpos, handorn)
        targetPosInHand, targetOrnInHand = p.multiplyTransforms(invHandPos, invHandOrn, targetpos, targetorn)
        # 観測
        observation = (targetPosInHand[0], targetPosInHand[1]) # ハンドからみたターゲットの相対位置(x,y)
        return observation, frame

    def act_env(self, observation, action):
        '''決定したactionに従って、ロボットハンドを動かす'''
        self.selectAction(action)
        for i in range(100):
            p.stepSimulation()
            if RENDER:
                time.sleep(1./240.)

        targetPosInHand_x, targetPosInHand_y = observation
        observation_next, frame = self.get_env()
        done = self.is_done(frame)

        return observation_next, done

    def selectAction(self, action):
        if action == 0:  # 前
            p.setJointMotorControlArray(
                    self.hand, [2,3], p.VELOCITY_CONTROL,
                    targetVelocities=[20,20],
                    forces=np.ones(2)*self.maxForce)
        elif action == 1:  # 後
            p.setJointMotorControlArray(
                    self.hand, [2,3], p.VELOCITY_CONTROL,
                    targetVelocities=[-20,-20],
                    forces=np.ones(2)*self.maxForce)
        elif action == 2:  # 右
            p.setJointMotorControlArray(
                    self.hand, [2,3], p.VELOCITY_CONTROL,
                    targetVelocities=[10,-10],
                    forces=np.ones(2)*self.maxForce)
        elif action == 3:  # 左
            p.setJointMotorControlArray(
                    self.hand, [2,3], p.VELOCITY_CONTROL,
                    targetVelocities=[10,-10],
                    forces=np.ones(2)*self.maxForce)

    def is_done(self, frame):
        '''observationによって終了判定をする'''
        #終了判定は面積が閾値以上&面積の変化なし（重心位置が画像の真ん中？カメラの位置によるけど，とりあえずはなし）
        done = False
        area_sum = self.calc_area(frame)
        if area_sum > AREA_THRESH:
            done = True
            # hold
        return done

    def reward(self, bodyA, bodyB):
        closestPoints = p.getClosestPoints(bodyA, bodyB, 10000)
        numPt = len(closestPoints)
        reward = -1000
        # print(numPt)
        if (numPt > 0):
            reward = -closestPoints[0][8]
            print('reward:', reward)
        return reward

    def run(self, test_interval=10, num_test=10):
        '''実行'''
        if RENDER:
            print('Environment.run')

        for episode in range(NUM_EPISODES):  # 試行数分繰り返す
            # test_intervalごとに性能をテスト
            if (episode % test_interval == 0 and episode != 0) or TEST_MODE:
                # 1episode以降はこっちに分岐
                self.test(num_episodes=num_test)
            else:
                if episode == 0:
                    # はじめにtest log生成
                    with open('test_reward_redArea.csv', 'w') as f:
                        f.write('mean,std\n')

                if not RENDER:
                    print('Episode:', episode)
                observation, frame = self.reset()  # 環境の初期化

                for step in range(MAX_STEPS):  # 1エピソードのループ
                    if RENDER:
                        print('Step: {0} of Episode: {1}'.format(step+1, episode))
                    # 行動を求める
                    action = self.agent.get_action(observation, episode, test=False)
                    # 行動a_tの実行により、s_{t+1}, r_{t+1}を求める
                    observation_next, done = self.act_env(observation, action)
                    # 報酬を与える
                    reward = observation_next[0] # 面積値
                    # reward = self.reward(self.hand, self.target)
                    # step+1の状態observation_nextを用いて,Q関数を更新する
                    self.agent.update_Q_function(observation, action, reward, observation_next)
                    # 観測の更新
                    observation = observation_next
                    # 終了時の処理
                    if done:
                        print('{0} Episode: Finished after {1} time steps'.format(episode, step+1))
                        break

    def test(self, num_episodes):
        '''性能をテスト'''
        test_reward = [] # test時の報酬を格納
        for episode in range(num_episodes):  # 試行数分繰り返す
            print('-*- test mode -*-')
            print('Episode:', episode)
            images = [] # １試行の映像を格納
            observation, frame = self.reset()  # 環境の初期化

            for step in range(MAX_STEPS):  # 1エピソードのループ
                if RENDER:
                    print('Step: {0} of Episode: {1}'.format(step+1, episode))
                # ロボット視点の映像を保存
                frame = self.renderPicture()
                frame = Image.fromarray(frame)
                images.append(frame)
                False# 行動を求める
                action = self.agent.get_action(observation, episode, test=True)
                # 行動a_tの実行により、s_{t+1}, r_{t+1}を求める
                observation_next, done = self.act_env(observation, action)
                # 報酬を与える
                reward = observation_next[0] # 面積値
                # reward = self.reward(self.hand, self.target)
                test_reward.append(reward)
                # 観測の更新
                observation = observation_next
                # 終了時の処理
                if done:
                    print('{0} Episode: Finished after {1} time steps'.format(episode, step+1))
                    images[0].save('test_success_episode.gif', save_all=True, append_images=images[1:], optimize=False, duration=1000) #成功エピソードの映像を保存
                    break
        # testの報酬のログをcsvで保存
        with open('test_reward_redArea.csv', 'a') as f:
            rew_mean = np.array(test_reward).mean() # エピソードで平均
            rew_std = np.array(test_reward).std() # 標準偏差
            f.write(str(rew_mean)+','+str(rew_std)+'\n')
        # 最終エピソードの動き
        images[0].save('0507test_final_episode.gif', save_all=True, append_images=images[1:], optimize=False, duration=1000)


# main
robot_hand_env = Environment()
robot_hand_env.run(test_interval=10, num_test=10)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('test_reward.csv')
reward = df['mean'].to_list()
error = df['std'].to_list()
episode = np.arange(0,len(reward)*10,10)

plt.plot(episode, reward, 'b.')
plt.errorbar(episode, reward, error, ecolor='green', fmt='b.', alpha=0.2)
plt.xlim(-100, episode[-1]+100)
plt.ylim(-1.2, 1.2)
plt.xlabel('episode')
plt.ylabel('reward')
plt.savefig('QL_results.png')
plt.show()
