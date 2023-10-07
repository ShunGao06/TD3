from env.static_environment import Env
import numpy as np
from matplotlib.patches import Ellipse, Circle
import matplotlib.pyplot as plt

def choose_action():
    a_1 = list(np.arange(10) + 1)
    a_1.reverse()
    a_1 = np.array(a_1)
    a_2 = np.arange(11)
    a_1 = a_1 / 10
    a_2 = a_2 * 2 / 10 - 1
    return a_1, a_2

if __name__=='__main__':
    obstacle = [[20, 20], [30, 30]]
    a_1, a_2 = choose_action()
    myenv = Env(obstacle)
    s = myenv.reset()
    min_state = [0, 0]
    min_fai = 0
    x_s = []
    y_s = []
    fai_s = []
    for i in range(500):
        min_score = -10000
        r_ = np.zeros(110)
        l = 0
        for j in a_1:
            for k in a_2:
                a = [j, k]
                _, r, done, state, fai = myenv.step(a)
                r_[l] = r
                l += 1
                if r > min_score:
                    min_score = r
                    min_state = state
                    min_fai = fai
                    min_a = a
        myenv.update_state(min_state, min_fai)
        x_s.append(min_state[0])
        y_s.append(min_state[1])
        fai_s.append(min_fai)
        if myenv.done == 1:
            fai_s = np.array(fai_s)
            x_s = np.array(x_s)
            y_s = np.array(y_s)
            break

    fig, ax = plt.subplots()
    ax.set_aspect(aspect=1, adjustable=None, anchor=None, share=False)
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 50)
    ax.xaxis.set_minor_locator(plt.MultipleLocator(10))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(10))
    ax.set_xlabel('x(km)')
    ax.set_ylabel('y(km)')
    cirl = Circle(xy=(myenv.goal[0], myenv.goal[1]), radius=5, alpha=1, fill=False)
    ax.add_patch(cirl)
    for ob in obstacle:
        cirl = Circle(xy=(ob[0], ob[1]), radius=myenv.r_obstacle, alpha=1)
        ax.add_patch(cirl)
    plt.plot(x_s, y_s, 'r')
    plt.show()

