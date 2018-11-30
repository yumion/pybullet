
# coding: utf-8

# In[1]:


import pybullet as p
import time
import pybullet_data


# In[2]:


#ウインドウ表示
p.connect(p.GUI) #or p.DIRECT for non-graphical version


# In[3]:


#urdfファイルを読み込む時のパスを指定してあげる
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally


# In[4]:


# 重力
p.setGravity(0,0,-10)


# In[5]:


#フィールドを追加
planeId = p.loadURDF("plane.urdf")


# In[6]:


#オブジェクトの初期位置を設定
cubeStartPos = [0,0,1]
cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])


# In[7]:


#オブジェクトを読み込み表示
boxId = p.loadURDF("r2d2.urdf",cubeStartPos, cubeStartOrientation)


# In[7]:


p.setAdditionalSearchPath("/home/dl-box/ros_ws/src/test_car_description/urdf/")
car = p.loadURDF("test_car.urdf",[1,0,0], cubeStartOrientation)


# In[1]:


#立方体を出現させる
cuid = p.createCollisionShape(p.GEOM_BOX, halfExtents = [1, 1, 1])
mass= 0 #static box
p.createMultiBody(mass,cuid)


# In[8]:


#the number of joints using the getNumJoints API
#ジョイント数を検索
p.getNumJoints(boxId)


# In[10]:


#jointの情報
"""
jointIndex ; int
jointName ; string
jointType ; int
qIndex ; int
uIndex ; int 
flags ; int
jointDamping ; float
jointFriction ; float
jointLowerLimit ; float
jointUpperLimit ; float
jointMaxForce ; float
jointMaxVelocity ; float
linkName ; string
jointAxis ; vec3
parentFramePos ; vec3
parentFrameOrn ; vec3
parentIndex ; int
"""
p.getJointInfo(boxId,2)


# In[8]:


#モーターを動かす
maxForce = 10
mode = p.VELOCITY_CONTROL
p.setJointMotorControlArray(boxId, jointIndices=[1,2,3,4], controlMode=mode, targetVelocities=[10,10,10,10], force=maxForce)


# In[ ]:


#シミュレーション開始
for i in range (10000):
    p.stepSimulation()
    time.sleep(1./240.)#世界の時間スピード


# In[9]:


#最終的なオブジェクトの位置と向き
cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
print(cubePos,cubeOrn)


# In[10]:


#フィールド、モデルすべてをリセット
p.resetSimulation()


# In[33]:


#終了
p.disconnect()


# In[ ]:


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

