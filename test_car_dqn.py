
# coding: utf-8

# In[1]:


import pybullet as p
import time
import pybullet_data
import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


class Test_car(gym.Env):
    
    def __init__(self):
        print("init")
        super().__init__()
        self.height = 320
        self.width = 320
        self.action_space = spaces.Discrete(4) #前後左右
        self.observation_space = spaces.Box(0, 255, [self.height, self.width, 3]) #Boxは連続値
        self.reward_range = [-1,1]
        '''pybullet側の初期設定'''
        p.connect(p.GUI)
        self.maxForce = 10
        self._reset()
        print("init_reset終了")
            
    def _reset(self):
        print("reset")
        self.done = False
        self.steps = 0
        targetX, targetY = np.random.permutation(np.arange(10))[0:2]
        self.targetPos = [targetX, targetY, 0]
        
        '''pybullet側'''
        #bulletの世界をリセット
        p.resetSimulation()
        #フィールドを表示
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0,0,-10)
        self.planeId = p.loadURDF("plane.urdf")
        
        #オブジェクトモデルを表示
        p.setAdditionalSearchPath("/home/dl-box/ros_ws/src/test_car_description/urdf/")
        self.startPos = [0,0,0]
        self.startOrientation = p.getQuaternionFromEuler([0,0,0])
        self.car = p.loadURDF("test_car.urdf", self.startPos, self.startOrientation)
        
        # ターゲットを表示
        self.target = p.createCollisionShape(
            p.GEOM_CYLINDER, radius=0.2, height=2, collisionFramePosition=self.targetPos)
        p.createMultiBody(0, self.target)
        
        print("reset end")
        return self.observation()
        
    def _step(self, action):
        print("step")
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
            time.sleep(1./240.)
        
        observation = self.observation()
        reward = self.reward()
        self.done = self.is_done()
        print("step end")
        return observation, reward, self.done, {}
        
        
    def _render(self, mode='rgb_array', close=False):
        print("render")
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
        
        print("render end")
        return rgb_array
    
    def _close(self):
        pass

    def _seed(self, seed=None):
        pass
    
    def observation(self):
        print("observation")
        rgb_array = self._render()
        rgb_array = rgb_array/255.0
        
        return rgb_array
        
    def is_done(self):
        print("is_done")
        distance = p.getClosestPoints(
            bodyA=self.car, bodyB=self.target, distance=100, linkIndexA=0)[0][8]
        if distance <= 0:
            self.done = True
        print("is_done end")
        return self.done
    
    def reward(self):
        print("reward")
        if self.done: reward = 1
        else: reward = -1
        print("reward end")
        return reward
        


# In[3]:


env = Test_car()


# In[4]:


#これらとpybulletのウィンドウ表示(p.connect)を分けないとカーネルが死ぬ
import gym
from baselines import deepq
import datetime


# In[5]:


def callback(lcl, glb):
    # stop training if reward exceeds 199
    total = sum(lcl['episode_rewards'][-101:-1]) / 100
    totalt = lcl['t']
    is_solved = totalt > 2000 and total >= -class Agent:
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
50
    return is_solved


# In[6]:


model = deepq.models.mlp([64])


# In[7]:


act = deepq.learn(
    env,
    q_func=model,
    lr=1e-3,
    max_timesteps=10000,
    buffer_size=50000,
    exploration_fraction=0.1,
    exploration_final_eps=0.02,
    print_freq=1,
    callback=callback
)
print("Saving model to racecar_model.pkl")
act.save("test_car_model.pkl")


# In[8]:


env.observation_space.shape


# In[25]:





# In[4]:


from keras.models import Sequential
from keras.layers import Dense, Reshape, Conv2D, Flatten, MaxPooling2D, BatchNormalization, Dropout
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

nb_actions = 4


# In[5]:


model = Sequential()
model.add(Reshape((320, 320, 3), input_shape=(1,) + env.observation_space.shape))
model.add(Conv2D(16, kernel_size=(7,7), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32, kernel_size=(7,7), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64, kernel_size=(5,5), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(128, kernel_size=(5,5), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(256, kernel_size=(5,5), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_actions, activation='linear'))

# print(model.summary())


# In[6]:


memory = SequentialMemory(limit=100000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])


# In[7]:


dqn.fit(env, nb_steps=100000, visualize=True, verbose=1)


# In[27]:


# After training is done, we save the final weights.
dqn.save_weights('dqn_{}_weights.h5f'.format("test_car-v0"), overwrite=True)


# In[18]:


# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=5, visualize=True)

