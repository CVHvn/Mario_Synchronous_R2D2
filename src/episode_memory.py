import numpy as np

class Episode_memory():
    def __init__(self, num_envs, m, l, n):
        self.num_envs = num_envs
        self.m = m
        self.l = l
        self.n = n
        self.lens = [self.m for _ in range(self.num_envs)]
        self.reset()

    def save(self, states, actions, rewards, dones, values, target_values, h, c, last_actions, last_rewards):
        for i in range(self.num_envs):
            self.lens[i] += 1
            self.states[i].append(states[i])
            self.actions[i].append(actions[i])
            self.rewards[i].append(rewards[i])
            self.dones[i].append(dones[i])
            self.values[i].append(values[i].detach().cpu())
            self.target_values[i].append(target_values[i].detach().cpu())
            self.h[i].append(h[i].detach().cpu())
            self.c[i].append(c[i].detach().cpu())
            self.last_actions[i].append(last_actions[i].detach().cpu())
            self.last_rewards[i].append(last_rewards[i].detach().cpu())

    def reset(self, env_i = None):
        if env_i is None:
            self.lens = [self.m for _ in range(self.num_envs)]
            self.states = [[] for _ in range(self.num_envs)]
            self.actions = [[] for _ in range(self.num_envs)]
            self.rewards = [[] for _ in range(self.num_envs)]
            self.dones = [[] for _ in range(self.num_envs)]
            self.values = [[] for _ in range(self.num_envs)]
            self.target_values = [[] for _ in range(self.num_envs)]
            self.h = [[] for _ in range(self.num_envs)]
            self.c = [[] for _ in range(self.num_envs)]
            self.last_actions = [[] for _ in range(self.num_envs)]
            self.last_rewards = [[] for _ in range(self.num_envs)]
        else:
            self.lens[env_i] = self.m
            self.states[env_i] = []
            self.actions[env_i] = []
            self.rewards[env_i] = []
            self.dones[env_i] = []
            self.values[env_i] = []
            self.target_values[env_i] = []
            self.h[env_i] = []
            self.c[env_i] = []
            self.last_actions[env_i] = []
            self.last_rewards[env_i] = []

    def get_data(self, env_i = None):
        if env_i is None:
            return self.states, self.actions, self.rewards, self.dones, self.values, self.target_values, self.h, self.c, self.last_actions, self.last_rewards
        return (self.states[env_i], self.actions[env_i], self.rewards[env_i],
                self.dones[env_i], self.values[env_i], self.target_values[env_i], self.h[env_i], self.c[env_i],
                self.last_actions[env_i], self.last_rewards[env_i])