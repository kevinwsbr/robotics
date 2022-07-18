from zmqRemoteApi import RemoteAPIClient
import matplotlib.pyplot as plt
import numpy as np
import math
import time

tol = 0.1
dt = 0.05

d1 = 0.125
a2 = 0.210
a3 = -0.075
d4 = 0.210
d6 = 0.070


def fkine(t1, t2, t3, t4, t5, t6):
    A1 = np.array([[math.cos(t1), 0, math.sin(t1), 0],
                   [math.sin(t1), 0, -math.cos(t1), 0],
                   [0, 1, 0, d1],
                   [0, 0, 0, 1]])

    A2 = np.array([[math.cos(t2 + math.pi / 2), -math.sin(t2 + math.pi / 2), 0, a2 * math.cos(t2 + math.pi / 2)],
                   [math.sin(t2 + math.pi / 2), math.cos(t2 + math.pi / 2), 0, a2 * math.sin(t2 + math.pi / 2)],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])

    A3 = np.array([[math.cos(t3 - math.pi / 2), 0, -math.sin(t3 - math.pi / 2), a3 * math.cos(t3 - math.pi / 2)],
                   [math.sin(t3 - math.pi / 2), 0, math.cos(t3 - math.pi / 2), a3 * math.sin(t3 - math.pi / 2)],
                   [0, -1, 0, 0],
                   [0, 0, 0, 1]])

    A4 = np.array([[math.cos(t4), 0, math.sin(t4), 0],
                   [math.sin(t4), 0, -math.cos(t4), 0],
                   [0, 1, 0, d4],
                   [0, 0, 0, 1]])

    A5 = np.array([[math.cos(t5), 0, -math.sin(t5), 0],
                   [math.sin(t5), 0, math.cos(t5), 0],
                   [0, -1, 0, 0],
                   [0, 0, 0, 1]])
    A6 = np.array([[math.cos(t6), -math.sin(t6), 0, 0],
                   [math.sin(t6), math.cos(t6), 0, 0],
                   [0, 0, 1, d6],
                   [0, 0, 0, 1]])

    return A1 @ A2 @ A3 @ A4 @ A5 @ A6


def jacobian(t1, t2, t3, t4, t5, t6):
    A1 = np.array([[math.cos(t1), 0, math.sin(t1), 0],
                   [math.sin(t1), 0, -math.cos(t1), 0],
                   [0, 1, 0, d1],
                   [0, 0, 0, 1]])

    A2 = np.array([[math.cos(t2 + math.pi / 2), -math.sin(t2 + math.pi / 2), 0, a2 * math.cos(t2 + math.pi / 2)],
                   [math.sin(t2 + math.pi / 2), math.cos(t2 + math.pi / 2), 0, a2 * math.sin(t2 + math.pi / 2)],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])

    A3 = np.array([[math.cos(t3 - math.pi / 2), 0, -math.sin(t3 - math.pi / 2), a3 * math.cos(t3 - math.pi / 2)],
                   [math.sin(t3 - math.pi / 2), 0, math.cos(t3 - math.pi / 2), a3 * math.sin(t3 - math.pi / 2)],
                   [0, -1, 0, 0],
                   [0, 0, 0, 1]])

    A4 = np.array([[math.cos(t4), 0, math.sin(t4), 0],
                   [math.sin(t4), 0, -math.cos(t4), 0],
                   [0, 1, 0, d4],
                   [0, 0, 0, 1]])

    A5 = np.array([[math.cos(t5), 0, -math.sin(t5), 0],
                   [math.sin(t5), 0, math.cos(t5), 0],
                   [0, -1, 0, 0],
                   [0, 0, 0, 1]])
    A6 = np.array([[math.cos(t6), -math.sin(t6), 0, 0],
                   [math.sin(t6), math.cos(t6), 0, 0],
                   [0, 0, 1, d6],
                   [0, 0, 0, 1]])

    T02 = A1 @ A2
    T03 = T02 @ A3
    T04 = T03 @ A4
    T05 = T04 @ A5
    T06 = T05 @ A6

    z0 = np.array([0, 0, 1]).T
    z1 = np.array([A1[0][2], A1[1][2], A1[2][2]]).T
    z2 = np.array([T02[0][2], T02[1][2], T02[2][2]]).T
    z3 = np.array([T03[0][2], T03[1][2], T03[2][2]]).T
    z4 = np.array([T04[0][2], T04[1][2], T04[2][2]]).T
    z5 = np.array([T05[0][2], T05[1][2], T05[2][2]]).T

    p0 = np.array([0, 0, 0]).T
    p1 = np.array([A1[0][3], A1[1][3], A1[2][3]]).T
    p2 = np.array([T02[0][3], T02[1][3], T02[2][3]]).T
    p3 = np.array([T03[0][3], T03[1][3], T03[2][3]]).T
    p4 = np.array([T04[0][3], T04[1][3], T04[2][3]]).T
    p5 = np.array([T05[0][3], T05[1][3], T05[2][3]]).T
    p = np.array([T06[0][3], T06[1][3], T06[2][3]]).T

    J0 = np.cross(z0, p - p0, axis=0)
    J1 = np.cross(z1, p - p1, axis=0)
    J2 = np.cross(z2, p - p2, axis=0)
    J3 = np.cross(z3, p - p3, axis=0)
    J4 = np.cross(z4, p - p4, axis=0)
    J5 = np.cross(z5, p - p5, axis=0)

    J = np.zeros((6, 6))

    J[:, 0] = np.vstack([J0, z0]).flatten()
    J[:, 1] = np.vstack([J1, z1]).flatten()
    J[:, 2] = np.vstack([J2, z2]).flatten()
    J[:, 3] = np.vstack([J3, z3]).flatten()
    J[:, 4] = np.vstack([J4, z4]).flatten()
    J[:, 5] = np.vstack([J5, z5]).flatten()

    return J


def getJointAngles():
    return np.array([sim.getJointPosition(sim.getObject('/joint1')),
                    sim.getJointPosition(sim.getObject('/joint2')),
                    sim.getJointPosition(sim.getObject('/joint3')),
                    sim.getJointPosition(sim.getObject('/joint4')),
                    sim.getJointPosition(sim.getObject('/joint5')),
                    sim.getJointPosition(sim.getObject('/joint6'))])


def setJointAngles(t1, t2, t3, t4, t5, t6):
    sim.setJointPosition(sim.getObject('/joint1'), t1)
    sim.setJointPosition(sim.getObject('/joint2'), t2)
    sim.setJointPosition(sim.getObject('/joint3'), t3)
    sim.setJointPosition(sim.getObject('/joint4'), t4)
    sim.setJointPosition(sim.getObject('/joint5'), t5)
    sim.setJointPosition(sim.getObject('/joint6'), t6)


def extractPose(T):
    x = T[0][3]
    y = T[1][3]
    z = T[2][3]

    roll = math.atan2(T[2][1], T[2][2])
    pitch = math.atan2(-T[2][0], math.sqrt(T[2][1]**2 + T[2][2]**2))
    yaw = math.atan2(T[1][0], T[0][0])

    return np.array([x, y, z, roll, pitch, yaw])


def getMatrix(objectHandler):
    matrix = sim.getObjectMatrix(objectHandler, -1)

    return np.array([[matrix[0], matrix[4], matrix[8], matrix[3]],
                    [matrix[1], matrix[5], matrix[9], matrix[7]],
                    [matrix[2], matrix[6], matrix[10], matrix[11]],
                    [0, 0, 0, 1]])


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

T = fkine(pos[0], pos[1], pos[2], pos[3], pos[4], pos[5])
J = jacobian(pos[0], pos[1], pos[2], pos[3], pos[4], pos[5])

goal_pose = extractPose(getMatrix(dummyHandler))
robot_pose = extractPose(T)

while np.linalg.norm(goal_pose - robot_pose) >= tol:
    goal_pose = extractPose(getMatrix(dummyHandler))
    goal.append(goal_pose)

    J = jacobian(pos[0], pos[1], pos[2], pos[3], pos[4], pos[5])
    J_cross = np.linalg.pinv(J)
    q_dot = J_cross @ (goal_pose - robot_pose).T
    pos = pos + (q_dot * dt)

    setJointAngles(pos[0], pos[1], pos[2], pos[3], pos[4], pos[5])
    time.sleep(dt)

    pos = getJointAngles()

    T = fkine(pos[0], pos[1], pos[2], pos[3], pos[4], pos[5])
    robot_pose = extractPose(T)
    current.append(robot_pose)

    plotData(current, goal)

sim.stopSimulation()
plt.show()
