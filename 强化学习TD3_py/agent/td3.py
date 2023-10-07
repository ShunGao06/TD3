import torch
import torch.nn as nn
import numpy as np
from net.net import Actor
from net.net import Critic

class DDPG(object):
    def __init__(self, state_dim, action_dim, replacement, memory_capacity, gamma=0.99, lr_a=0.0003,
                 lr_c=0.0003, batch_size=64):
        super(DDPG, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory_capacity = memory_capacity
        self.replacement = replacement
        self.gamma = gamma
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.batch_size = batch_size

        self.memory = np.zeros((memory_capacity, state_dim * 2 + action_dim + 2))
        self.pointer = 0

        self.actor = Actor(state_dim, action_dim)
        self.actor_target = Actor(state_dim, action_dim)

        self.critic_1 = Critic(state_dim, action_dim)
        self.critic_1_target = Critic(state_dim, action_dim)

        self.critic_2 = Critic(state_dim, action_dim)
        self.critic_2_target = Critic(state_dim, action_dim)

        self.aopt = torch.optim.Adam(self.actor.parameters(), lr=lr_a)
        self.copt_1 = torch.optim.Adam(self.critic_1.parameters(), lr=lr_c)
        self.copt_2 = torch.optim.Adam(self.critic_2.parameters(), lr=lr_c)

        self.update_cnt = 0
        self.policy_target_update = 2

        self.mse_loss = nn.MSELoss()
        self.tau = 0.005

    def sample(self):
        indices = np.random.choice(self.memory_capacity, size=self.batch_size)
        return self.memory[indices, :]

    def choose_action(self, s):
        s = torch.FloatTensor(s)
        action = self.actor(s)
        return action.detach().numpy()

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1 - self.tau) + param.data * self.tau)

    def learn(self, a_los, c_los):
        self.update_cnt += 1
        bm = self.sample()
        bs = torch.FloatTensor(bm[:, :self.state_dim])
        ba = torch.FloatTensor(bm[:, self.state_dim:self.state_dim + self.action_dim])
        br = torch.FloatTensor(bm[:, self.state_dim + self.action_dim: self.state_dim + self.action_dim + 1])
        bs_ = torch.FloatTensor(bm[:, self.state_dim + self.action_dim + 1:self.state_dim * 2 + self.action_dim + 1])
        b_done = torch.FloatTensor(bm[:, self.state_dim * 2 + self.action_dim + 1:])

        a_ = self.actor_target(bs_)
        #a_ = torch.clip(a_ + torch.clip(torch.randn_like(a_) * 0.2, -0.1, 0.1), -1, 1)
        q_1 = self.critic_1_target(bs_, a_)
        q_2 = self.critic_2_target(bs_, a_)
        q_target = br + self.gamma * torch.min(q_1, q_2) * b_done

        q_eval_1 = self.critic_1(bs, ba)
        critic_1_loss = self.mse_loss(q_eval_1, q_target)
        self.copt_1.zero_grad()
        critic_1_loss.backward(retain_graph=True)
        self.copt_1.step()

        q_eval_2 = self.critic_2(bs, ba)
        critic_2_loss = self.mse_loss(q_eval_2, q_target)
        self.copt_2.zero_grad()
        critic_2_loss.backward(retain_graph=True)
        self.copt_2.step()

        if self.update_cnt % self.policy_target_update == 0:
            a = self.actor(bs)
            q = self.critic_1(bs, a)
            a_loss = -q.mean()
            self.aopt.zero_grad()
            a_loss.backward()
            self.aopt.step()
            a_los.append(a_loss.detach().numpy())
            self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic_1, self.critic_1_target)
        self.soft_update(self.critic_2, self.critic_2_target)
        #self.s_update()
        return a_los, c_los

    def store_transition(self, s, a, r, s_, done):
        transition = np.hstack((s, a, [r], s_, [done]))
        index = self.pointer % self.memory_capacity
        self.memory[index, :] = transition
        self.pointer += 1