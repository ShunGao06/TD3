import torch
import torch.nn as nn
import math as ma
import numpy as np

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        n_hidden_1 = 64
        n_hidden_2 = 64

        self.layer_1 = nn.Linear(state_dim, n_hidden_1)
        nn.init.normal_(self.layer_1.weight, 0., 0.1)
        nn.init.constant_(self.layer_1.bias, 0.1)

        self.layer_2 = nn.Linear(n_hidden_1, n_hidden_2)
        nn.init.normal_(self.layer_2.weight, 0., 0.1)
        nn.init.constant_(self.layer_2.bias, 0.1)

        self.output = nn.Linear(n_hidden_2, action_dim)
        self.output.weight.data.normal_(0., 0.1)
        self.output.bias.data.fill_(0.1)

    def forward(self, s):
        x = torch.relu(self.layer_1(s))
        x = torch.relu(self.layer_2(x))
        a = torch.tanh(self.output(x))
        # 对action进行放缩，实际上a in [-1,1]
        return a

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        n_hidden_1 = 64
        n_hidden_2 = 64
        out_dim = 1
        self.layer_1 = nn.Linear(state_dim, n_hidden_1)
        nn.init.normal_(self.layer_1.weight, 0., 0.1)
        nn.init.constant_(self.layer_1.bias, 0.1)

        self.layer_2 = nn.Linear(action_dim, n_hidden_1)
        nn.init.normal_(self.layer_2.weight, 0., 0.1)
        nn.init.constant_(self.layer_2.bias, 0.1)

        self.layer_3 = nn.Linear(n_hidden_1, n_hidden_2)
        nn.init.normal_(self.layer_3.weight, 0., 0.1)
        nn.init.constant_(self.layer_3.bias, 0.1)

        self.output = nn.Linear(n_hidden_2, out_dim)

    def forward(self, s, a):
        x = self.layer_1(s)
        y = self.layer_2(a)
        q_val = torch.relu(x+y)
        q_val = torch.relu(self.layer_3(q_val))
        q_val = self.output(q_val)
        return q_val