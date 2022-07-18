from zmqRemoteApi import RemoteAPIClient
import matplotlib.pyplot as plt
import numpy as np
import math
import time

l1 = 0.475
l2 = 0.4
offset = 0.1

tol = 0.1
dt = 0.05


def fkine(t1, t2, t3, d4):
    return np.array([[math.cos(t1 + t2 - t3), -math.sin(t1 + t2 - t3), 0, l1 * math.cos(t1) + l2 * math.cos(t1 + t2)],
                    [math.sin(t1 + t2 - t3), math.cos(t1 + t2 - t3), 0, l1 * math.sin(t1) + l2 * math.sin(t1 + t2)],
                    [0, 0, -1, -d4],
                    [0, 0, 0, 1]])


def jacobian(t1, t2):
    return np.array([[-l2 * math.sin(t1 + t2) - l1 * math.sin(t1), -l2 * math.sin(t1 + t2), 0, 0],
                    [l2 * math.cos(t1 + t2) + l1 * math.cos(t1), l2 * math.cos(t1 + t2), 0, 0],
                    [0, 0, -1, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [1, 1, 0, -1]])


def getJointAngles():
    return np.array([sim.getJointPosition(sim.getObject('/MTB/axis')),
                    sim.getJointPosition(sim.getObject('/MTB/link/axis')),
                    sim.getJointPosition(sim.getObject('/MTB/link/axis/link/axis')),
                    sim.getJointPosition(sim.getObject('/MTB/link/axis/link/axis/axis'))])


def getMatrix(objectHandler):
    matrix = sim.getObjectMatrix(objectHandler, -1)

    return np.array([[matrix[0], matrix[4], matrix[8], matrix[3]],
                    [matrix[1], matrix[5], matrix[9], matrix[7]],
                    [matrix[2], matrix[6], matrix[10], matrix[11]],
                    [0, 0, 0, 1]])


def setJointAngles(alpha, beta, gamma, delta):
    fistJointHandler = sim.getObject('/MTB/axis')
    secondJointHandler = sim.getObject('/MTB/link/axis')
    thirdJointHandler = sim.getObject('/MTB/link/axis/link/axis')
    fourthJointHandler = sim.getObject('/MTB/link/axis/link/axis/axis')

    sim.setJointPosition(fistJointHandler, alpha)
    sim.setJointPosition(secondJointHandler, beta)
    sim.setJointPosition(thirdJointHandler, delta)
    sim.setJointPosition(fourthJointHandler, gamma - offset)


def extractPose(T):
    x = T[0][3]
    y = T[1][3]
    z = T[2][3]

    roll = math.atan2(T[2][1], T[2][2])
    pitch = math.atan2(-T[2][0], math.sqrt(T[2][1]**2 + T[2][2]**2))
    yaw = math.atan2(T[1][0], T[0][0])

    return np.array([x, y, z, roll, pitch, yaw])

def plotData(current, goal):
    axis[0, 0].plot(np.array(current)[:, 0], linestyle='--', color='royalblue')
    axis[0, 0].plot(np.array(goal)[:, 0], linestyle='-', color='slategray')
    axis[0, 0].set_title('X')

    axis[0, 1].plot(np.array(current)[:, 1], linestyle='--', color='royalblue')
    axis[0, 1].plot(np.array(goal)[:, 1], linestyle='-', color='slategray')
    axis[0, 1].set_title('Y')

    axis[0, 2].plot(np.array(current)[:, 2], linestyle='--', color='royalblue')
    axis[0, 2].plot(np.array(goal)[:, 2], linestyle='-', color='slategray')
    axis[0, 2].set_title('Z')

    axis[1, 0].plot(np.array(current)[:, 3], linestyle='--', color='royalblue')
    axis[1, 0].plot(np.array(goal)[:, 3], linestyle='-', color='slategray')
    axis[1, 0].set_title('Roll')

    axis[1, 1].plot(np.array(current)[:, 4], linestyle='--', color='royalblue')
    axis[1, 1].plot(np.array(goal)[:, 4], linestyle='-', color='slategray')
    axis[1, 1].set_title('Pitch')

    axis[1, 2].plot(np.array(current)[:, 5], linestyle='--', color='royalblue')
    axis[1, 2].plot(np.array(goal)[:, 5], linestyle='-', color='slategray')
    axis[1, 2].set_title('Yaw')

    plt.pause(dt)

figure, axis = plt.subplots(2, 3)
goal = []
current = []

client = RemoteAPIClient()
sim = client.getObject('sim')
client.setStepping(True)

sim.startSimulation()

dummyHandler = sim.getObjectHandle('/reference')

pos = getJointAngles()
T = fkine(pos[0], pos[1], pos[2], pos[3])
J = jacobian(pos[0], pos[1])

goal_pose = extractPose(getMatrix(dummyHandler))
robot_pose = extractPose(T)

while np.linalg.norm(goal_pose - robot_pose) >= tol:
    goal_pose = extractPose(getMatrix(dummyHandler))
    goal.append(goal_pose)

    J = jacobian(pos[0], pos[1])
    J_cross = np.linalg.pinv(J)
    q_dot = J_cross @ (goal_pose - robot_pose).T
    pos = pos + (q_dot * dt)

    setJointAngles(pos[0], pos[1], pos[2], pos[3])
    time.sleep(dt)

    pos = getJointAngles()
    T = fkine(pos[0], pos[1], pos[2], pos[3])

    robot_pose = extractPose(T)
    current.append(robot_pose)

    plotData(current, goal)

sim.stopSimulation()
plt.show()