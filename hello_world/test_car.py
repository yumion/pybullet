# coding: utf-8
import pybullet as p
import pybullet_data
from pybullet_envs.bullet import racecar
import numpy as np
import time
import os

# physicsClient = p.connect(p.DIRECT)
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-10)
planeId = p.loadURDF("plane100.urdf")
# planeId = p.loadSDF('stadium.sdf')

p.setAdditionalSearchPath(pybullet_data.getDataPath())
r2d2 = p.loadURDF("r2d2.urdf", [2,0,0.5])

p.setAdditionalSearchPath(pybullet_data.getDataPath())
# racecar = p.loadURDF('racecar/racecar_differential.urdf', [0,0,0.5])
racecar = racecar.Racecar(p, pybullet_data.getDataPath())

p.setAdditionalSearchPath(os.environ['HOME']+"/atsushi/catkin_ws/src/robotHand_v1/urdf/")
startPos = [0,0,1]
startOrientation = p.getQuaternionFromEuler([0,0,0])
car = p.loadURDF("smahoHand.urdf", startPos, startOrientation)

p.setJointMotorControlArray(car, [2,3], p.VELOCITY_CONTROL, targetVelocities=[6,10], forces=[10,10])

# 2台目
cuid = p.loadURDF("test_car.urdf",[0,1,1], startOrientation)
p.createMultiBody(cuid)

#円柱を出現させる
target_col = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.2, height=2, collisionFramePosition=[2,0,0])
target_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.2, length=2, visualFramePosition=[2,0,0], rgbaColor=[0,255,0,1])
mass= 0 #static box
p.createMultiBody(mass, target_col, target_vis)

base_pos, orn = p.getBasePositionAndOrientation(racecar.racecarUniqueId)
print(base_pos)
orn
p.invertTransform(base_pos, orn)
r2d2_pos, r2d2_orn = p.getBasePositionAndOrientation(r2d2)
r2d2_pos
p.multiplyTransforms(base_pos, orn, r2d2_pos, r2d2_orn)
# 物理パラメータ変更
p.getNumJoints(car)
p.getJointInfo(car, 3)
p.getDynamicsInfo(racecar.racecarUniqueId, 0)
# 摩擦係数を変更
for joint in range(p.getNumJoints(car)):
    p.changeDynamics(car, joint, lateralFriction=10)
p.changeDynamics(car, 1, lateralFriction=10)
p.changeDynamics(car, 2, lateralFriction=10)
p.changeDynamics(car, 0, mass=100)

#オブジェクト視点カメラ
cam_eye = np.array(base_pos) + [0.1,0,0.2]
cam_target = [2,0,0.2]
cam_upvec = [1,0,1] #upベクトル(カメラの向きを決める)[1,0,1]で正面？
view_matrix = p.computeViewMatrix(
        cameraEyePosition=cam_eye,
        cameraTargetPosition=cam_target,
        cameraUpVector=cam_upvec)

print(cam_eye)

#固定点カメラ
cam_dist = 4 # min 1
cam_yaw = 0
cam_pitch = -10
view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=base_pos,
        distance=cam_dist,
        yaw=cam_yaw,
        pitch=cam_pitch,
        roll=0,
        upAxisIndex=2)

RENDER_WIDTH = 320
RENDER_HEIGHT = 320
proj_matrix = p.computeProjectionMatrixFOV(
        fov=60, aspect=float(RENDER_WIDTH)/RENDER_HEIGHT,
        nearVal=0.1, farVal=100.0)

#カメラ画像を取得(width, height, rgb, depth, segment_mask)
(_, _, rgb, _, mask) = p.getCameraImage(
        width=RENDER_WIDTH, height=RENDER_HEIGHT, viewMatrix=view_matrix,
        projectionMatrix=proj_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)
#renderer=p.ER_BULLET_HARDWARE_OPENGL or ER_TINY_RENDERER はp.connect(p.DIRECT)のとき

#np.array形式に変換
rgb_array = np.array(rgb)
rgb_array = rgb_array[:,:,:3]


def render(uniqueId, width=320, height=320):
    base_pos, orn = p.getBasePositionAndOrientation(uniqueId)
    yaw = p.getEulerFromQuaternion(orn)[2]
    rot_matrix = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    target_relative_vec2D = np.array([2,0])
    target_abs_vec2D = np.dot(rot_matrix, target_relative_vec2D)

    cam_eye = np.array(base_pos) + np.array([0,0,0.2])
    cam_target = np.array(base_pos) + np.append(target_abs_vec2D, 0.2)
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
    mask_array = np.array(mask)
    return rgb_array, mask_array


#レンダリングされたカメラ画像を確認
import matplotlib.pyplot as plt
img, mask = render(racecar.racecarUniqueId)
# plt.imshow(mask, cmap="gray")
plt.imshow(img)

def calc_center(self, img):
    '''重心座標(x,y)を求める'''
    img = self.green_detect(img)
    mu = cv2.moments(img, False)
    x, y = int(mu["m10"] / (mu["m00"] + 1e-7)), int(mu["m01"] / (mu["m00"] + 1e-7))
    # 重心を丸でくくる
    #cv2.circle(img, (x, y), 4, 100, 2, 4)
    print('x',x,'y',y)
    return x, y


#モーターを動かす
maxForce = 20
velocity = 20
mode = p.VELOCITY_CONTROL
for wheel in [8,15]:
    p.setJointMotorControl2(racecar, wheel, mode, targetVelocity=velocity, force=maxForce)
for steer in [0,2]:
    p.setJointMotorControl2(racecar, steer, p.POSITION_CONTROL, targetPosition=0.3)

racecar.applyAction([1,-0.6])



#シミュレーション開始
for i in range(5000):
    p.stepSimulation()
    # img, mask = render(car) #renderするととても遅くなる
    time.sleep(1./240.)#世界の時間スピード?
base_pos, orn = p.getBasePositionAndOrientation(car)
print(base_pos)
print(p.getEulerFromQuaternion(orn))

p.getClosestPoints(bodyA=car, bodyB=target_col, distance=1000)
p.getJointInfo(racecar.racecarUniqueId,5)

p.setJointMotorControlArray(
    car, np.arange(p.getNumJoints(car))[1:], p.VELOCITY_CONTROL,
    targetVelocities=[12,20,12,20],
    forces=np.ones(4)*20)

np.arange(p.getNumJoints(car))[1:]

p.disconnect()
