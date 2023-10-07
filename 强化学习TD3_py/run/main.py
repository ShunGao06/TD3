from env.static_environment import Env
from agent.td3 import DDPG
import numpy as np
from matplotlib.patches import Ellipse, Circle
import matplotlib.pyplot as plt
import torch
import xlsxwriter as xw

def prepareing(myenv, ddpg):
    done = True
    while done:
        s = myenv.reset()
        for i in range(1000):
            a = ddpg.choose_action(s)
            a = np.clip(np.random.normal(a, 1), -1, 1)
            s_, r, don, state, fai = myenv.step(a)
            myenv.update_state(state, fai)
            don = 1 - don
            ddpg.store_transition(s, a, r, s_, don)
            if don == 0:
                break
            if ddpg.pointer >= ddpg.memory_capacity:
                done = False
            s = s_

def drawing(obstacles, u_x, u_y):
    fig, ax = plt.subplots()
    ax.set_aspect(aspect=1, adjustable=None, anchor=None, share=False)
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 50)
    ax.xaxis.set_minor_locator(plt.MultipleLocator(10))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(10))
    ax.set_xlabel('x(km)')
    ax.set_ylabel('y(km)')
    for obstacle in obstacles:
        cirl = Circle(xy=(obstacle[0], obstacle[1]), radius=4, alpha=1)
        ax.add_patch(cirl)
    ax.plot(u_x, u_y)
    ax.plot(myenv.goal[0], myenv.goal[1], 'r+')
    ax.plot(u_x[0], u_y[0], 'bs')
    ax.plot(u_x[-1], u_y[-1], 'b+')
    cirl = Circle(xy=(myenv.goal[0], myenv.goal[1]), radius=5, alpha=1, fill=False)
    ax.add_patch(cirl)
    plt.pause(1)
    plt.close()

def drawing_1(obstacles, u_x, u_y):
    fig, ax = plt.subplots()
    ax.set_aspect(aspect=1, adjustable=None, anchor=None, share=False)
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 50)
    ax.xaxis.set_minor_locator(plt.MultipleLocator(10))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(10))
    ax.set_xlabel('x(km)')
    ax.set_ylabel('y(km)')
    for obstacle in obstacles:
        cirl = Circle(xy=(obstacle[0], obstacle[1]), radius=4, alpha=1)
        ax.add_patch(cirl)
    ax.plot(u_x, u_y)
    ax.plot(myenv.goal[0], myenv.goal[1], 'r+')
    ax.plot(u_x[0], u_y[0], 'bs')
    ax.plot(u_x[-1], u_y[-1], 'b+')
    cirl = Circle(xy=(myenv.goal[0], myenv.goal[1]), radius=5, alpha=1, fill=False)
    ax.add_patch(cirl)
    plt.show()

if __name__ == '__main__':
    obstacle = [[100, 100]]
    # [[15, 10], [10, 30], [20, 20], [35, 15], [30, 25], [30, 40], [40, 30], [45, 15]]
    #[[20, 20], [20, 40], [37, 25], [30, 7]]
    #obstacle = [[10, 20], [15, 5], [15, 35], [20, 20], [30, 10], [30, 20], [30, 35], [30, 45], [40, 30], [45, 15]]
    #obstacle = [[10, 20], [15, 5], [15, 35], [20, 15], [30, 10], [30, 20], [30, 35], [30, 45], [40, 30], [45, 15]]
    #obstacle = [[20, 10], [10, 30], [20, 20], [35, 15], [30, 25], [30, 35], [30, 45], [40, 30], [45, 15]]
    #obstacle = [[15, 10], [10, 30], [20, 20], [35, 15], [30, 25], [30, 35], [30, 45], [40, 30], [45, 15]]
    #obstacle = [[10, 30], [20, 20], [25, 10], [25, 30], [35, 20], [30, 40], [40, 30], [45, 15]]
    #obstacle = [[10, 30], [20, 20], [32, 10], [27, 30], [25, 45], [40, 30], [45, 15]]

    myenv = Env(obstacle)

    MEMORY_CAPACITY = 65536
    state_dim = myenv.r_number + myenv.s_number
    action_dim = 2
    REPLACEMENT = [
        dict(name='soft', tau=0.005),
        dict(name='hard', rep_iter=600)
    ][0]
    ddpg = DDPG(state_dim=state_dim,
                action_dim=action_dim,
                replacement=REPLACEMENT,
                memory_capacity=MEMORY_CAPACITY)
    rd = []
    var = 3
    MAX_EPISODES = 1000
    MAX_EP_STEPS = 500
    a_loss = []
    c_loss = []
    prepareing(myenv, ddpg)
    u_x0 = []
    u_y0 = []

    for i in range(MAX_EPISODES):
        s = myenv.reset()
        ep_reward = 0
        u_x = []
        u_y = []
        u_x.append(myenv.state[0])
        u_y.append(myenv.state[1])
        for j in range(MAX_EP_STEPS):
            a = ddpg.choose_action(s)
            a = np.clip(np.random.normal(a, var), -1, 1)
            s_, r, done, state, fai = myenv.step(a)
            u_x.append(myenv.state[0])
            u_y.append(myenv.state[1])
            done = 1 - done
            ep_reward += r
            ddpg.store_transition(s, a, r, s_, done)
            ddpg.learn(a_loss, c_loss)
            if myenv.done == 1 and myenv.done_ == 1:
                print('EPISODES', i, 'Episodes_steps', j, 'success', 'reward', ep_reward)
                rd.append(ep_reward)
                u_x0.append(u_x)
                u_y0.append(u_y)
                #drawing(obstacle, u_x, u_y)
                break
            elif myenv.done == 1 and myenv.done_ == 0:
                print('EPISODES', i, 'Episodes_steps', j, 'fail', 'reward', ep_reward)
                rd.append(ep_reward)
                #drawing(obstacle, u_x, u_y)
                break
            elif j == MAX_EP_STEPS - 1:
                print('EPISODES', i, 'Episodes_steps', j, 'fail', 'reward', ep_reward)
                rd.append(ep_reward)
                #drawing(obstacle, u_x, u_y)
                break
            s = s_
            myenv.update_state(state, fai)
        var *= 0.996

    torch.save(ddpg.actor, 'actor.pt')
    torch.save(ddpg.actor_target, 'actor_target.pt')
    torch.save(ddpg.critic_1, 'critic_1.pt')
    torch.save(ddpg.critic_2, 'critic_2.pt')
    torch.save(ddpg.critic_1_target, 'critic_1_target.pt')
    torch.save(ddpg.critic_2_target, 'critic_2_target.pt')

    workbook = xw.Workbook('数据.xlsx')  # 创建工作簿
    worksheet1 = workbook.add_worksheet("sheet1")
    worksheet1.activate()
    title = ['奖励']  # 设置表头
    worksheet1.write_row('A1', title)
    i = 2  # 从第二行开始写入数据
    for j in range(len(rd)):
        insertData = [rd[j]]
        row = 'A' + str(i)
        worksheet1.write_row(row, insertData)
        i += 1

    worksheet2 = workbook.add_worksheet("sheet2")
    worksheet2.activate()
    title = ['x']  # 设置表头
    worksheet2.write_row('A1', title)
    row = 1
    col = 0  # 从第二行开始写入数据

    for j in u_x0:
        for k in j:
            insertData = [k]
            worksheet2.write_row(row, col, insertData)
            row += 1
        col += 1
        row = 1

    worksheet3 = workbook.add_worksheet("sheet3")
    worksheet3.activate()
    title = ['y']
    worksheet3.write_row('A1', title)
    row = 1
    col = 0  # 从第二行开始写入数据

    for j in u_y0:
        for k in j:
            insertData = [k]
            worksheet3.write_row(row, col, insertData)
            row += 1
        col += 1
        row = 1
    workbook.close()

    # file = open('reward.txt', 'a')
    # mid = str(rd).replace('[', '').replace(']', '')
    # mid = mid.replace("'", '').replace(',', '') + '\n'
    # file.write(mid)
    # file.close()
    #
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1000)
    ax.xaxis.set_minor_locator(plt.MultipleLocator(100))
    ax.set_xlabel('Episode')
    ax.set_ylabel('Episode Reward')
    ax.plot(rd)
    plt.show()
    #drawing_1(myenv.obstacles, u_x0, u_y0)


