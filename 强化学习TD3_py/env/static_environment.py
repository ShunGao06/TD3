import math as ma
import sympy
import numpy as np

class Env:
    def __init__(self, obstacle):
        self.size = 50.0
        self.obstacles = obstacle
        self.r_obstacle = 3.0
        self.r_radar = 2.0
        self.o_number = len(obstacle)
        self.success_condition = 5.0
        self.r_number = 5
        self.s_number = 2
        self.seta = 20/180 * ma.pi
        self.r_ = [self.seta * (n - self.r_number//2) for n in range(self.r_number)]
        self.turn = [True]

    def reset(self):
        self.state = [5.0, 5.0]
        self.goal = [40.0, 40.0]
        self.fai = self.Angle(self.state, self.goal)
        d_r = self.radar(self.state, self.fai)
        s = [(self.goal[0] - self.state[0])/self.size, (self.goal[1] - self.state[1])/self.size] + d_r
        return s

    def step(self, a):
        d = self.distance(self.state, self.goal)
        v = abs(a[0] + 1) * 0.15
        fai = self.Fai(self.fai + a[1] * (ma.pi/9))
        x = self.state[0] + v * ma.cos(fai)
        y = self.state[1] + v * ma.sin(fai)
        state = [x, y]
        d_t = self.distance(state, self.goal)
        d_r = self.radar(state, fai)
        s_ = [(self.goal[0] - state[0])/self.size, (self.goal[1] - state[1])/self.size] + d_r
        self.done, self.done_ = self.is_end(state, d_t)
        if self.done == 1 and self.done_ == 1:
            r = 10
        elif self.done == 1 and self.done_ == 0:
            r = -10
        else:
            if min(d_r) < 1:
                r_1 = 0
                r_2 = (d_r[2] - 1) + (d_r[1] - 1) + (d_r[3] - 1) + 0.2 * (d_r[0] - 1) + 0.2 * (d_r[4] - 1)
            else:
                r_1 = d - d_t
                r_2 = 0
            r = r_1 + r_2
        return s_, r, self.done, state, fai

    def distance(self, x, y):
        d = ma.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)
        return d

    def Angle(self, x, y):
        beta = ma.atan((y[1] - x[1])/(y[0] - x[0] + (1e-8)))
        if x[0] <= y[0] and x[1] < y[1]:
            beta = beta
        elif x[0] > y[0] and x[1] <= y[1]:
            beta = ma.pi + beta
        elif x[0] >= y[0] and x[1] > y[1]:
            beta = beta - ma.pi
        else:
            beta = beta
        return beta

    def Fai(self, fai):
        if fai > ma.pi:
            fai = fai - ma.pi * 2
        elif fai < -ma.pi:
            fai = ma.pi * 2 + fai
        else:
            fai = fai
        return fai

    def radar(self, state, fai):
        d = list(map(self.distance, [state] * self.o_number, self.obstacles))
        if min(d) - self.r_obstacle >= self.r_radar:
            d_r = [1] * self.r_number
        else:
            t = [self.obstacles[d.index(n)] for n in d if n - self.r_obstacle < self.r_radar]
            fai_s = self.fais(fai)
            states = list(map(self.compute_radar_state, [state] * self.r_number, fai_s))
            d_r = list(map(self.compute_distance_radar, [state] * self.r_number, states, [fai] * self.r_number, fai_s, [t] * self.r_number))
        return d_r

    def compute_radar_state(self, state, fai):
        x = state[0] + self.r_radar * ma.cos(fai)
        y = state[1] + self.r_radar * ma.sin(fai)
        state_ = [x, y]
        return state_

    def compute_distance_obstacles(self, state, fai, obstacle):
        d = abs(ma.tan(fai) * obstacle[0] + (-1) * obstacle[1] + (state[1] - ma.tan(fai) * state[0]))/ma.sqrt((ma.tan(fai) ** 2 + 1))
        return d

    def fais(self, fai):
        fai_s = [self.Fai(a + b) for a, b in zip([fai] * self.r_number, self.r_)]
        return fai_s

    def compute_distance_radar(self, state, states, fai, fai_s, k):
        beta = list(map(self.Angle, [state] * len(k), k))
        max_fai = self.Fai(fai + ma.pi/2)
        min_fai = self.Fai(fai - ma.pi/2)
        beta_ = [t for t in beta if t <= max_fai and t >= min_fai]
        o = [k[beta.index(t)] for t in beta_]
        if len(o) == 0:
            return 1
        else:
            d_2 = list(map(self.compute_distance_obstacles, [state] * len(o), [fai_s] * len(o), o))
            if min(d_2) > self.r_obstacle:
                return 1
            else:
                t = [i for i in d_2 if i < self.r_obstacle]
                t_1 = [o[d_2.index(t)]]
                d_3 = list(map(self.distance, [state] * len(t_1), t_1))
                t_2 = [t_1[d_3.index(min(d_3))]]
                inter = self.intersections(state, states, t_2[0], fai_s)
                d = np.clip(self.distance(state, inter), 0, 1)
                return d / self.r_radar

    def intersections(self, state, states, j, fai):
        x = sympy.symbols('x')
        t = sympy.solve((x - j[0]) ** 2 + (ma.tan(fai) * (x - states[0]) + states[1] - j[1]) ** 2 - self.r_obstacle ** 2, x)
        d_1 = self.distance(state, [t[0], ma.tan(fai) * (t[0] - states[0]) + states[1]])
        d_2 = self.distance(state, [t[1], ma.tan(fai) * (t[1] - states[0]) + states[1]])
        d = [d_1, d_2]
        m = d.index(min(d))
        x = t[m]
        y = ma.tan(fai) * (x - states[0]) + states[1]
        intersection = [x, y]
        return intersection

    def is_end(self, state, d_t):
        d = list(map(self.distance, [state] * self.o_number, self.obstacles))
        if state[0] <= 0 or state[0] >= self.size or state[1] <= 0 or state[1] >= self.size or min(d) <= self.r_obstacle:
            done = 1
            done_ = 0
        else:
            if d_t <= self.success_condition:
                done = 1
                done_ = 1
            else:
                done = 0
                done_ = 0
        return done, done_

    def update_state(self, state, fai):
        self.state = state
        self.fai = fai