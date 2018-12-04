import pybullet as p

p.connect(p.DIRECT)
p.setAdditionalSearchPath("../catkin_ws/src/simple_car/simple_car_description/urdf/")

# p.setAdditionalSearchPath()

num = 0
while True:
    Id = p.loadURDF("plane100.urdf")
    car = p.loadURDF("test_car.urdf")
    print("number: ", num)
    p.stepSimulation()
    p.resetSimulation()
    num += 1
