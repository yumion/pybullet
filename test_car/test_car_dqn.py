# coding: utf-8
import pybullet as p
import time
import numpy as np
import gym
from gym import spaces
import cv2


class Test_car(gym.Env):

    def __init__(self):
        print("init")
        super().__init__()
        self.episodes = 0
        self.max_steps = 10
        self.height = 64
        self.width = 64
        self.action_space = spaces.Discrete(4) #前後左右
        self.observation_space = spaces.Box(0, 255, [self.height, self.width, 3]) #Boxは連続値
        self.reward_range = [-1,1]
        '''pybullet側の初期設定'''
        p.connect(p.DIRECT)
        p.setAdditionalSearchPath("../catkin_ws/src/simple_car/simple_car_description/urdf/")
        self.maxForce = 10
        self.reset()
        print("init_reset終了")

    def reset(self):

        print("====episode:"+str(self.episodes)+"=================")
        self.episodes += 1
        self.steps = 0

        '''pybullet側'''
        #bulletの世界をリセット
        p.resetSimulation()
        #フィールドを表示
        p.setGravity(0,0,-10)
        self.plane = p.loadURDF("plane100.urdf")

        #オブジェクトモデルを表示
        self.startPos = [0,0,0]
        self.startOrientation = p.getQuaternionFromEuler([0,0,0])
        self.car = p.loadURDF("test_car.urdf", self.startPos, self.startOrientation)

        # ターゲットを表示
        targetX, targetY = np.random.permutation(np.arange(10))[0:2]
        self.targetPos = [targetX, targetY, 0]
        self.target = p.createCollisionShape(
            p.GEOM_CYLINDER, radius=0.2, height=2, collisionFramePosition=self.targetPos)
        p.createMultiBody(0, self.target)

        return self.observation()

    def step(self, action):

        print("---step:"+str(self.steps)+"-------")
        self.steps += 1
        if action == 0:
            #前進
            p.setJointMotorControlArray(
                self.car, np.arange(p.getNumJoints(self.car))[1:], p.VELOCITY_CONTROL,
                targetVelocities=[20,20,20,20],
                forces=np.ones(4)*self.maxForce)
        elif action == 1:
            #右
            p.setJointMotorControlArray(
                self.car, np.arange(p.getNumJoints(self.car))[1:], p.VELOCITY_CONTROL,
                targetVelocities=[20,12,20,12],
                forces=np.ones(4)*self.maxForce)
        elif action == 2:
            #後退
            p.setJointMotorControlArray(
                self.car, np.arange(p.getNumJoints(self.car))[1:], p.VELOCITY_CONTROL,
                targetVelocities=[-20,-20,-20,-20],
                forces=np.ones(4)*self.maxForce)
        elif action == 3:
            #左
            p.setJointMotorControlArray(
                self.car, np.arange(p.getNumJoints(self.car))[1:], p.VELOCITY_CONTROL,
                targetVelocities=[12,20,12,20],
                forces=np.ones(4)*self.maxForce)

        for i in range(200):
            p.stepSimulation()
            #time.sleep(1./240.)

        observation = self.observation()
        done = self.is_done()
        reward = self.reward()

        return observation, reward, done, {}


    def render(self, mode='rgb_array', close=False):

        if mode != "rgb_array":
            return np.array([])
        base_pos, orn = p.getBasePositionAndOrientation(self.car)
        cam_eye = np.array(base_pos) + [0.1,0,0.2]
        cam_target = np.array(base_pos) + [2,0,0.2]
        cam_upvec = [1,0,1]

        view_matrix = p.computeViewMatrix(
                cameraEyePosition=cam_eye,
                cameraTargetPosition=cam_target,
                cameraUpVector=cam_upvec)

        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=float(self.width)/self.height,
            nearVal=0.1, farVal=100.0)

        (_, _, rgb, _, mask) = p.getCameraImage(
            width=self.width, height=self.height, viewMatrix=view_matrix,
            projectionMatrix=proj_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)

        rgb_array = np.array(rgb)
        rgb_array = rgb_array[:,:,:3]
        mask_array = np.array(mask)

        return rgb_array

    def close(self):
        pass

    def seed(self, seed=None):
        pass

    def observation(self):

        rgb_array = self.render()
        rgb_array = rgb_array / 255.0

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
        return per

    def is_done(self):

        frame = self.render()
        self.area = self.calc_area(frame)
        if self.area >= 80:
            done = True
        elif self.steps > self.max_steps:
            done = True
        else:
            done = False
        return done

    def reward(self):

        if self.area >= 80:
            reward = 100
        elif self.steps > self.max_steps:
            reward = -50
        else:
            reward = self.area
        print("reward: ", reward)
        return reward


env = Test_car()

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

model = deepq.models.cnn_to_mlp([(512,5,1)], [256,64,4])


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
