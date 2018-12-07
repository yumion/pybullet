# coding: utf-8
import numpy as np
import cv2
import time
from datetime import datetime  # 時刻を取得
import pybullet as p
import pybullet_data


'''定数の設定'''
NUM_DIZITIZED = 6  # 各状態の離散値への分割数
discount = 0.99  # 時間割引率
lr = 0.5  # 学習係数
MAX_STEPS = 30  # 1試行のstep数cartpoleは195steps立ち続ければ終わり
NUM_EPISODES = 100000  # 最大試行回数
AREA_THRESH = 80  # 赤色物体面積の閾値．0~100で規格化してある

'''学習するときはFalse，学習済みのモデルを使用するときはTrue'''
# 使うq_tableのファイル名を"trained_q_table.npy"とすること
TEST_MODE = False
'''追加学習するときはTrue'''
ADD_TRAIN_MODE = True


class Agent:
    '''CartPoleのエージェントクラスです、棒付き台車そのものになります'''

    def __init__(self, num_states, num_actions):
        self.brain = Brain(num_states, num_actions)  # エージェントが行動を決定するための頭脳を生成

    def update_Q_function(self, observation, action, reward, observation_next):
        '''Q関数の更新'''
        self.brain.update_Q_table(observation, action, reward, observation_next)

    def get_action(self, observation, step):
        '''行動の決定'''
        action = self.brain.decide_action(observation, step)
        return action


class Brain:
    '''エージェントが持つ脳となるクラスです、Q学習を実行します'''

    def __init__(self, num_states, num_actions):
        self.num_actions = num_actions  # ロボットハンドの取れる行動数(コマンド数)
        if TEST_MODE or ADD_TRAIN_MODE:  # 保存したQ-tableを使用
            self.q_table = np.load('trained_q_table.npy')
        else:  # Qテーブルを作成。行数は状態を分割数^(4変数)にデジタル変換した値、列数は行動数を示す
            self.q_table = np.random.uniform(low=0, high=1, size=(NUM_DIZITIZED**num_states, num_actions))

    def bins(self, clip_min, clip_max, num):
        '''観測した状態（連続値）を離散値にデジタル変換する閾値を求める'''
        return np.linspace(clip_min, clip_max, num + 1)[1:-1]

    def digitize_state(self, observation):
        '''観測したobservation状態を、離散値に変換する'''
        area_sum, area_v = observation
        digitized = [
            np.digitize(area_sum, bins=self.bins(0, 10.0, NUM_DIZITIZED)), #　面積の比率
            np.digitize(area_v, bins=self.bins(-10.0, 10.0, NUM_DIZITIZED))
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

    def decide_action(self, observation, episode):
        '''ε-greedy法で徐々に最適行動のみを採用する'''
        state = self.digitize_state(observation)
        epsilon = 0.5 * (1 / (episode + 1))

        if epsilon <= np.random.uniform(0, 1):
            action = np.argmax(self.q_table[state][:])
        else:
            action = np.random.choice(self.num_actions)  # 0,1の行動をランダムに返す
        return action


class  Environment:
    '''CartPoleを実行する環境のクラスです'''

    def __init__(self):
        self.num_states = 2  # 課題の状態の数(面積と重心(x,y)と、それぞれの変化量で6つ)
        self.num_actions = 4  # ロボットハンドの行動（前進，後退，右旋回，左旋回，握る，離す，止まる）
        self.agent = Agent(self.num_states, self.num_actions)  # 環境内で行動するAgentを生成
        '''pybullet'''
        p.connect(p.GUI)
        self.maxForce = 10

    def renderPicture(self, height=320, width=320):
        '''bullet側からカメラ画像を取得'''
        base_pos, orn = p.getBasePositionAndOrientation(self.car)
        cam_eye = np.array(base_pos) + [0.1,0,0.2]
        cam_target = np.array(base_pos) + [2,0,0.2]
        cam_upvec = [1,0,1]

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
        mask_array = np.array(mask)

        return rgb_array

    def green_detect(self, img):
        '''緑色のマスク'''
        # HSV色空間に変換
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # 緑色のHSVの値域
        hsv_min = np.array([50, 100, 100])
        hsv_max = np.array([70, 255, 255])
        mask = cv2.inRange(hsv, hsv_min, hsv_max)
        return mask

    def calc_area(self, img):
        '''面積計算'''
        img = self.green_detect(img)
        pix_area = cv2.countNonZero(img)  # ピクセル数
        # パーセントを算出
        h, w = img.shape  # frameの面積
        per = round(100 * float(pix_area) / (w * h), 3)  # 0-100で規格化
        print('GREEN_AREA: ', per)
        return pix_area, per

    def reset(self):
        '''環境を初期化する'''
        print('Environment.reset\n')

        #bulletの世界をリセット
        p.resetSimulation()

        #フィールドを表示
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0,0,-10)
        self.planeId = p.loadURDF("plane100.urdf")

        #オブジェクトモデルを表示
        p.setAdditionalSearchPath("../../catkin_ws/src/simple_car/simple_car_description/urdf/")
        self.startPos = [0,0,0]
        self.startOrientation = p.getQuaternionFromEuler([0,0,0])
        self.car = p.loadURDF("test_car.urdf", self.startPos, self.startOrientation)

        # ターゲットを表示
        targetX, targetY = np.random.permutation(np.arange(10))[0:2]
        self.targetPos = [targetX, targetY, 0]
        self.target = p.createCollisionShape(
            p.GEOM_CYLINDER, radius=0.2, height=2, collisionFramePosition=self.targetPos)
        p.createMultiBody(0, self.target)

        # 目標の面積, 重心の位置を取得する
        frame = self.renderPicture()
        _, area_sum = self.calc_area(frame)
        area_v = 0

        observation = (area_sum, area_v)

        return observation, frame

    def get_env(self, area_sum_before):
        '''環境を認識する'''
        '''カメラで写真をとりOpencv2で面積と重心を取得する'''
        frame = self.renderPicture()
        # 赤色の面積とその変化量, 重心の位置とその変化量を取得する
        _, area_sum = self.calc_area(frame)
        area_v = area_sum - area_sum_before
        # 観測量として返す
        observation = (area_sum, area_v)
        return observation, frame

    def act_env(self, observation, action):
        '''決定したactionに従って、ロボットハンドを動かす'''
        if action == 0:  # 前
            self.go()
        elif action == 1:  # 後
            self.back()
        elif action == 2:  # 右
            self.right()
        elif action == 3:  # 左
            self.left()

        for i in range(50):
            p.stepSimulation()
            time.sleep(1./240.)

        area_sum, _ = observation
        observation_next, _ = self.get_env(area_sum)
        done = self.is_done(observation_next)

        return observation_next, done

    def go(self):
        p.setJointMotorControlArray(
                self.car, np.arange(p.getNumJoints(self.car))[1:], p.VELOCITY_CONTROL,
                targetVelocities=[10,10,10,10],
                forces=np.ones(4)*self.maxForce)
    def back(self):
        p.setJointMotorControlArray(
                self.car, np.arange(p.getNumJoints(self.car))[1:], p.VELOCITY_CONTROL,
                targetVelocities=[-10,-10,-10,-10],
                forces=np.ones(4)*self.maxForce)
    def right(self):
        p.setJointMotorControlArray(
                self.car, np.arange(p.getNumJoints(self.car))[1:], p.VELOCITY_CONTROL,
                targetVelocities=[10,6,10,6],
                forces=np.ones(4)*self.maxForce)
    def left(self):
        p.setJointMotorControlArray(
                self.car, np.arange(p.getNumJoints(self.car))[1:], p.VELOCITY_CONTROL,
                targetVelocities=[6,10,6,10],
                forces=np.ones(4)*self.maxForce)
    def stop(self):
        p.setJointMotorControlArray(
                self.car, np.arange(p.getNumJoints(self.car))[1:], p.VELOCITY_CONTROL,
                targetVelocities=[0,0,0,0],
                forces=np.ones(4)*self.maxForce)

    def is_done(self, observation):
        '''observationによって終了判定をする'''
        #終了判定は面積が閾値以上&面積の変化なし（重心位置が画像の真ん中？カメラの位置によるけど，とりあえずはなし）
        done = False
        area_sum, area_v = observation
        if area_sum > AREA_THRESH:
            done = True
        return done

    def run(self):
        '''実行'''
        print('Environment.run')
        complete_episodes = 0  # 連続で取り続けた試行数
        is_episode_final = False  # 最終試行フラグ

        for episode in range(NUM_EPISODES):  # 試行数分繰り返す
            observation, frame = self.reset()  # 環境の初期化

            for step in range(MAX_STEPS):  # 1エピソードのループ
                print('Step: {0} of Episode: {1}'.format(step+1, episode))
                # 行動を求める
                action = self.agent.get_action(observation, episode)
                # 行動a_tの実行により、s_{t+1}, r_{t+1}を求める
                observation_next, done = self.act_env(observation, action)

                # 報酬を与える
                if done:
                    reward = 1  # 目標を掴んだら報酬1を与える
                    print ('reward: +1')
                    complete_episodes += 1  # 連続記録を更新
                else:
                    reward = 0  # 途中の報酬は0
                    print ('reward: 0')

                # step+1の状態observation_nextを用いて,Q関数を更新する
                if TEST_MODE:  # 保存したQ-TABLEを使用する
                    continue
                else:
                    self.agent.update_Q_function(observation, action, reward, observation_next)

                # 観測の更新
                observation = observation_next

                # 終了時の処理
                if done:
                    print('{0} Episode: Finished after {1} time steps'.format(episode, step + 1))
                    break

                # 1episode内でdoneできなかったら罰を与える
                if step == MAX_STEPS-1:
                    reward = -1
                    print('reward: -1')
                    complete_episodes = 0  # 4step以上連続で立ち続けた試行数をリセット

            if is_episode_final is True:  # 最終試行では動画を保存と描画
                Brain(num_states=self.num_states, num_actions=self.num_actions).save_Q_table()  # Q-tableを保存する
                print('finished')
                break

            if complete_episodes >= 10:  # 5連続成功なら
                print('10回連続成功\n次で最終試行')
                is_episode_final = True  # 次の試行を最終試行とする

# main
robot_hand_env = Environment()
robot_hand_env.run()
