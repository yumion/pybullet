# coding: utf-8
import pybullet as p
import time
import pybullet_data

#ウインドウ表示
p.connect(p.GUI) #or p.DIRECT for non-graphical version

#urdfファイルを読み込む時のパスを指定してあげる
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally

# 重力
p.setGravity(0,0,-10)
#フィールドを追加
planeId = p.loadURDF("plane.urdf")

#オブジェクトの初期位置を設定
cubeStartPos = [0,0,1]
cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])

#オブジェクトを読み込み表示
boxId = p.loadURDF("r2d2.urdf",cubeStartPos, cubeStartOrientation)

p.setAdditionalSearchPath("../../catkin_ws/src/simple_car/simple_car_description/urdf/")
car = p.loadURDF("test_car.urdf",[1,0,0], cubeStartOrientation)

#立方体を出現させる
cuid = p.createCollisionShape(p.GEOM_BOX, halfExtents = [1, 1, 1])
mass= 0 #static box
p.createMultiBody(mass,cuid)

#the number of joints using the getNumJoints API
#ジョイント数を検索
p.getNumJoints(boxId)

#jointの情報
"""
jointIndex : the same joint index as the input parameter
jointName : the name of the joint, as specified in the URDF (or SDF etc) file
jointType : type of the joint, this also implies the number of position and velocity variables.
JOINT_REVOLUTE, JOINT_PRISMATIC, JOINT_SPHERICAL, JOINT_PLANAR, JOINT_FIXED. See the section on Base, Joint and Links for more details.
qIndex : the first position index in the positional state variables for this body
uIndex : the first velocity index in the velocity state variables for this body
flags : reserved
jointDamping : the joint damping value, as specified in the URDF file
jointFriction : the joint friction value, as specified in the URDF file
jointLowerLimit : Positional lower limit for slider and revolute (hinge) joints.
jointUpperLimit : Positional upper limit for slider and revolute joints. Values ignored in case upper limit <lower limit.
jointMaxForce : Maximum force specified in URDF (possibly other file formats) Note that this value is not automatically used. You can use maxForce in 'setJointMotorControl2'.
jointMaxVelocity : Maximum velocity specified in URDF. Note that the maximum velocity is not used in actual motor control commands at the moment.
linkName : the name of the link, as specified in the URDF (or SDF etc.) file
jointAxis : joint axis in local frame (ignored for JOINT_FIXED)
parentFramePos : joint position in parent frame
parentFrameOrn : joint orientation in parent frame
parentIndex : parent link index, -1 for base
"""
p.getJointInfo(boxId,2)

"""
mass : mass in kg
lateral_friction : friction coefficient
local inertia diagonal : local inertia diagonal.
local inertial pos : position of inertial frame in local coordinates of the joint frame
local inertial orn :  orientation of inertial frame in local coordinates of joint frame
restitution : coefficient of restitution
rolling friction : rolling friction coefficient orthogonal to contact normal
spinning friction : spinning friction coefficient around contact normal
contact damping : -1 if not available. damping of contact constraints.
contact stiffness : -1 if not available. stiffness of contact constraints.
"""
p.getDynamicsInfo(boxId,2)

#モーターを動かす
maxForce = 10
mode = p.VELOCITY_CONTROL
p.setJointMotorControlArray(boxId, jointIndices=[1,2,3,4], controlMode=mode, targetVelocities=[10,10,10,10], force=maxForce)

#シミュレーション開始
for i in range (10000):
    p.stepSimulation()
    time.sleep(1./240.)#世界の時間スピード

#最終的なオブジェクトの位置と向き
cubePos, cubeOrn = p.getBasePositionAndOrientation(car)
print(cubePos,cubeOrn)

#フィールド、モデルすべてをリセット
p.resetSimulation()

#終了
p.disconnect()


'''
#オブジェクトを複数表示
import pybullet as p
import time

p.connect(p.GUI)
p.createCollisionShape(p.GEOM_PLANE)
p.createMultiBody(0,0)

sphereRadius = 0.05
colSphereId = p.createCollisionShape(p.GEOM_SPHERE,radius=sphereRadius)
colBoxId = p.createCollisionShape(p.GEOM_BOX,halfExtents=[sphereRadius,sphereRadius,sphereRadius])

mass = 1
visualShapeId = -1


link_Masses=[1]
linkCollisionShapeIndices=[colBoxId]
linkVisualShapeIndices=[-1]
linkPositions=[[0,0,0.11]]
linkOrientations=[[0,0,0,1]]
linkInertialFramePositions=[[0,0,0]]
linkInertialFrameOrientations=[[0,0,0,1]]
indices=[0]
jointTypes=[p.JOINT_REVOLUTE]
axis=[[0,0,1]]

for i in range (3):
	for j in range (3):
		for k in range (3):
			basePosition = [1+i*5*sphereRadius,1+j*5*sphereRadius,1+k*5*sphereRadius+1]
			baseOrientation = [0,0,0,1]
			if (k&2):
				sphereUid = p.createMultiBody(mass,colSphereId,visualShapeId,basePosition,baseOrientation)
			else:
				sphereUid = p.createMultiBody(mass,colBoxId,visualShapeId,basePosition,baseOrientation,linkMasses=link_Masses,linkCollisionShapeIndices=linkCollisionShapeIndices,linkVisualShapeIndices=linkVisualShapeIndices,linkPositions=linkPositions,linkOrientations=linkOrientations,linkInertialFramePositions=linkInertialFramePositions, linkInertialFrameOrientations=linkInertialFrameOrientations,linkParentIndices=indices,linkJointTypes=jointTypes,linkJointAxis=axis)

			p.changeDynamics(sphereUid,-1,spinningFriction=0.001, rollingFriction=0.001,linearDamping=0.0)
			for joint in range (p.getNumJoints(sphereUid)):
				p.setJointMotorControl2(sphereUid,joint,p.VELOCITY_CONTROL,targetVelocity=1,force=10)


p.setGravity(0,0,-10)
p.setRealTimeSimulation(1)

p.getNumJoints(sphereUid)
for i in range (p.getNumJoints(sphereUid)):
	p.getJointInfo(sphereUid,i)

while (1):
	keys = p.getKeyboardEvents()
	print(keys)

	time.sleep(0.01)
'''
