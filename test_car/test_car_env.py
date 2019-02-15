# coding: utf-8
import pybullet as p
import time
import numpy as np
import gym
from gym import spaces
import cv2


class Test_car(gym.Env):

    def __init__(self, render=False, height=320, width=320, num_actions=4, num_states=3):
        print("init")
        super().__init__()
        self.rendering = render
        self.episodes = 0
        self.max_steps = 30
        self.height = height
        self.width = width
        self.action_space = spaces.Discrete(num_actions) #前後左右
        observation_high = np.ones(num_states) * 100  # 観測空間(state)の次元とそれらの最大値
        self.observation_space = spaces.Box(-observation_high, observation_high, dtype=np.float32) #Boxは連続値
        self.reward_range = [-1,1]
        '''pybullet側の初期設定'''
        if self.rendering:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        p.setAdditionalSearchPath("../../catkin_ws/src/simple_car/simple_car_description/urdf/")
        self.maxForce = 10
        self.reset()
        print("init_reset終了")

    def reset(self):

        print("\n====episode:"+str(self.episodes)+"=================")
        self.episodes += 1
        self.steps = 0
        targetX, targetY = np.random.permutation(np.arange(10))[0:2]
        self.targetPos = [targetX, targetY, 0]

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
        self.target = p.createCollisionShape(
            p.GEOM_CYLINDER, radius=0.2, height=2, collisionFramePosition=self.targetPos)
        vis = p.createVisualShape(
            p.GEOM_CYLINDER, radius=0.2, length=2, visualFramePosition=self.targetPos, rgbaColor=[0,255,0,1])
        p.createMultiBody(0, self.target, vis)

        return self.observation()

    def step(self, action):

        if self.rendering:
            print("\n---step:"+str(self.steps)+"-------")
        self.steps += 1
        if action == 0:
            #前進
            p.setJointMotorControlArray(
                self.car, np.arange(p.getNumJoints(self.car))[1:], p.VELOCITY_CONTROL,
                targetVelocities=[10,10,10,10],
                forces=np.ones(4)*self.maxForce)
        elif action == 1:
            #右
            p.setJointMotorControlArray(
                self.car, np.arange(p.getNumJoints(self.car))[1:], p.VELOCITY_CONTROL,
                targetVelocities=[10,6,10,6],
                forces=np.ones(4)*self.maxForce)
        elif action == 2:
            #後退
            p.setJointMotorControlArray(
                self.car, np.arange(p.getNumJoints(self.car))[1:], p.VELOCITY_CONTROL,
                targetVelocities=[-10,-10,-10,-10],
                forces=np.ones(4)*self.maxForce)
        elif action == 3:
            #左
            p.setJointMotorControlArray(
                self.car, np.arange(p.getNumJoints(self.car))[1:], p.VELOCITY_CONTROL,
                targetVelocities=[6,10,6,10],
                forces=np.ones(4)*self.maxForce)

        for i in range(200):
            p.stepSimulation()
            if self.rendering:
                time.sleep(1./240.)

        observation = self.observation()
        done = self.is_done()
        reward = self.reward()

        return observation, reward, done, {}


    def render(self, mode='rgb_array', close=False):

        if mode != "rgb_array":
            return np.array([])
        base_pos, orn = p.getBasePositionAndOrientation(self.car)
        yaw = p.getEulerFromQuaternion(orn)[2] # z軸方向から見た本体の回転角度
        rot_matrix = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]) # 回転行列
        target_relative_vec2D = np.array([2,0]) # 本体から見たtargetの相対位置
        target_abs_vec2D = np.dot(rot_matrix, target_relative_vec2D) # targetの絶対位置

        cam_eye = np.array(base_pos) + np.array([0,0,0.2])
        cam_target = np.array(base_pos) + np.append(target_abs_vec2D, 0.2) # z=0.2は足()
        cam_upvec = [0,0,1]

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
        '''観測'''
        rgb_array = self.render()
        # 面積
        area_sum = self.calc_area(rgb_array)
        # 重心
        center_x, center_y = self.calc_center(rgb_array)
        return area_sum, center_x, center_y

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
        if self.rendering:
            print('GREEN_AREA: ', per)
        return per

    def calc_center(self, img):
        '''重心座標(x,y)を求める'''
        img = self.green_detect(img)
        mu = cv2.moments(img, False)
        x, y = int(mu["m10"] / (mu["m00"] + 1e-7)), int(mu["m01"] / (mu["m00"] + 1e-7))
        return x, y

    def is_done(self):

        frame = self.render()
        self.area = self.calc_area(frame)
        if self.area >= 50:
            done = True
        elif self.steps > self.max_steps:
            done = True
        else:
            done = False
        return done

    def reward(self):

        if self.area >= 50:
            reward = 1
            print("reward: ", reward)
        elif self.steps > self.max_steps:
            reward = -1
            print("reward: ", reward)
        else:
            reward = 0
        return reward
