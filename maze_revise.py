import numpy as np
import tkinter as tk
import random
import time
import matplotlib.pyplot as plt

# -1 is origin, 0 is road, 1 is wall, 2 is goal 
maze = np.array([
    [0, 0, -1, 0, 0],
    [0, 1, 1, 1, 1],
    [0, 0, 0, 0, 0],
    [1, 0, 1, 1, 0],
    [0, 0, 0, 1, 1],
    [1, 0, 1, 0, 1],
    [1, 0, 1, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 1, 0, 1, 2]
])

def init_q_table(maze):
    Q = np.zeros(maze.shape).tolist()
    for i, row in enumerate(Q):
        for j, _ in enumerate(row):
            Q[i][j] = [0, 0, 0, 0] # up, down, left, right
    return np.array(Q, dtype='f')

def get_action(Q_table, state, action_list, e_greedy=0.8):
    if random.random() > e_greedy:
        return random.choice(action_list)
    else:
        Qsa = Q_table[state].tolist()
        return action_list[Qsa.index(max(Qsa))]

def get_next_max_q(Q_table, state):
    return max(np.array(Q_table[state]))

def update_q_table(Q_table, state, action, next_state, reward, lr=0.7, gamma=0.9):
    Qs = Q_table[state]
    Qsa = Qs[action]
    Qs[action] = (1 - lr) * Qsa + lr * (reward + gamma * get_next_max_q(Q_table, next_state))
    return Q_table

def get_next_state(state, action):
    row = state[0]
    column = state[1]
    if action == 'up':
        row -= 1
    elif action == 'down':
        row += 1
    elif action == 'left':
        column -= 1
    elif action == 'right':
        column += 1
    nextState = (row, column)
    try:
        # Beyond the boundary or hit the wall.
        if row < 0 or column < 0 or maze[row, column] == 1:
            return [state, False]
        # Goal
        elif maze[row, column] == 2:
            return [nextState, True]
        # Forward
        else:
            return [nextState, False]
    except IndexError as e:
        # Beyond the boundary.
        return [state, False]

def do_action(state, action):
    nextState, result = get_next_state(state, action)
    # No move
    if nextState == state:
        reward = -10
    # Goal
    elif result:
        reward = 100
    # Forward
    else:
        reward = -1
    return [reward, nextState, result]


initState = (np.where(maze == -1)[0][0], np.where(maze == -1)[1][0])
Q_table = init_q_table(maze)
action_list = ['up', 'down', 'left', 'right']
y=[]
for j in range(0, 30):
    state = initState
    time.sleep(0.1)
    i = 0
    while True:
        i += 1
        # Get the next step from the Agent
        action = get_action(Q_table, state, action_list, 0.9)
        # Give the action to the Environment to execute
        reward, next_state, result = do_action(state, action)
        # Update Q Table based on Environment's response
        Q_table = update_q_table(Q_table, state, action_list.index(action), next_state, reward)
        # Agent's state changes
        state = next_state
        if result:
            #print(f' {j+1:2d} : {i} steps to the goal.')
            print(j)
            y.append(i)
            break
plt.plot(np.arange(30),y)
plt.xlabel('Iteration (cumulative Q table)'); plt.ylabel('Steps to goal')
plt.show()
# print(Q_table)
