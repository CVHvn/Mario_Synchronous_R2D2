from PIL import Image
from collections import deque
from datetime import datetime
from pathlib import Path
import copy
import cv2
import imageio
import numpy as np
import random, os
import torch
from torch import nn
import torch.nn.functional as F
import torch.multiprocessing as mp
#import multiprocessing as mp
from torchvision import transforms as T
import gc

# Gym is an OpenAI toolkit for RL
import gym
from gym.spaces import Box
from gym.wrappers import FrameStack

# NES Emulator for OpenAI Gym
from nes_py.wrappers import JoypadSpace

# Super Mario environment for OpenAI Gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY

from src.environment import *
from src.episode_memory import *
from src.sumtree import *
from src.per import *
from src.model import *

class Agent():
    def __init__(self, world, stage, action_type, envs, num_envs, additional_bonus_state_8_4_option,
                 state_dim, action_dim, save_dir, save_model_step,
                 save_figure_step, learn_step, total_step_or_episode, total_step, total_episode, model, target_model,
                 gamma, learning_rate, max_grad_norm, target_update_freq, replay_buffer_size, per,
                 per_eps, per_alpha, per_beta, eta, m, l, n, batch_size, loss_type, epsilons, start_learning_step,
                 start_learning_sequence, device):
        self.world = world
        self.stage = stage
        self.action_type = action_type

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir
        self.learn_step = learn_step
        self.total_step_or_episode = total_step_or_episode
        self.total_step = total_step
        self.total_episode = total_episode
        self.target_update_freq = target_update_freq

        self.current_step = 0
        self.current_episode = 0

        self.save_model_step = save_model_step
        self.save_figure_step = save_figure_step

        self.device = device
        self.save_dir = save_dir

        self.num_envs = num_envs
        self.envs = envs
        self.additional_bonus_state_8_4_option = additional_bonus_state_8_4_option
        self.model = model.to(self.device)
        self.target_model = target_model.to(self.device)

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, eps = 1e-3)

        #self.model = torch.compile(self.model)

        self.m = m
        self.l = l
        self.n = n
        self.epsilons = epsilons
        self.start_learning_step = start_learning_step
        self.start_learning_sequence = start_learning_sequence

        self.per_eps = per_eps
        self.per_alpha = per_alpha
        self.per_beta = per_beta
        self.eta = eta
        self.replay_buffer_size = replay_buffer_size
        self.per = per

        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.is_completed = False

        self.env = None
        self.max_test_score = -1e9
        self.loss_type = loss_type

        # I just log 1000 lastest update and print it to log.
        self.losses = np.zeros((1000,)).reshape(-1)
        self.loss_index = 0
        self.len_loss = 0

    def save_figure(self, is_training = False):
        # test current model and save model/figure if model yield best total rewards.
        # create env for testing, reset test env
        if self.env is None:
            self.env = create_env(self.world, self.stage, self.action_type, self.additional_bonus_state_8_4_option, True)
        state = self.env.reset()
        done = False

        images = []
        total_reward = 0
        total_step = 0
        num_repeat_action = 0
        old_action = -1

        episode_time = datetime.now()

        # create h, c as zeros
        h = torch.zeros((1, 512), dtype=torch.float, device = self.device)
        c = torch.zeros((1, 512), dtype=torch.float, device = self.device)

        last_actions = torch.zeros((1, self.action_dim), device = self.device)
        last_rewards = torch.zeros((1,), device = self.device).reshape(-1)

        # play 1 episode, just get loop action with max probability from model until the episode end.
        while not done:
            with torch.no_grad():
                Q, h, c = self.model(torch.tensor(state, dtype = torch.float, device = self.device).unsqueeze(0), h, c,
                                     last_actions, last_rewards.unsqueeze(-1))

            action = Q.argmax(-1).item()
            next_state, reward, done, trunc, info = self.env.step(action)
            state = next_state
            img = Image.fromarray(self.env.current_state)
            images.append(img)
            total_reward += reward
            total_step += 1

            last_actions = F.one_hot(torch.as_tensor(np.array([action]), device = self.device), self.action_dim).to(self.device)
            last_rewards[0] = reward

            if action == old_action:
                num_repeat_action += 1
            else:
                num_repeat_action = 0
            old_action = action
            if num_repeat_action == 200:
                break

        #logging, if model yield better result, save figure (test_episode.mp4) and model (best_model.pth)
        if is_training:
            f_out = open(f"logging_test.txt", "a")
            f_out.write(f'episode_reward: {total_reward:.4f} episode_step: {total_step} current_step: {self.current_step} \
loss: {(self.losses.sum()/self.len_loss):.4f} len_per: {self.per.real_size} episode_time: {datetime.now() - episode_time}\n')
            f_out.close()

        if total_reward > self.max_test_score or info['flag_get']:
            imageio.mimsave('test_episode.mp4', images)
            self.max_test_score = total_reward
            if is_training:
                torch.save(self.model.state_dict(), f"best_model.pth")

        # if model can complete this game, stop training by set self.is_completed to True
        if info['flag_get']:
            self.is_completed = True

    def save_model(self):
        torch.save(self.model.state_dict(), f"model_{self.current_step}.pth")

    def load_model(self, model_path = None):
        if model_path is None:
            model_path = f"model_{self.current_step}.pth"
        self.model.load_state_dict(torch.load(model_path))

    def update_loss_statis(self, loss):
        # update loss for logging, just save 1000 latest updates.
        self.losses[self.loss_index] = loss
        self.loss_index = (self.loss_index + 1)%1000
        self.len_loss = min(self.len_loss+1, 1000)

    def select_action(self, states, h, c, target_h, target_c, last_actions, last_rewards):
        states = torch.as_tensor(np.array(states), device = self.device)

        with torch.no_grad():
            Q, h, c = self.model(states, h, c, last_actions, last_rewards.unsqueeze(-1))
            target_q, target_h, target_c = self.target_model(states, target_h, target_c, last_actions, last_rewards.unsqueeze(-1))
            actions = Q.argmax(-1)

            n_actions = Q.shape[1]
            batch_size = states.shape[0]
            random_actions = torch.randint(0, n_actions, (batch_size,), device=self.device)

            use_random = torch.rand(batch_size, device=self.device) < self.epsilons
            actions = torch.where(use_random, random_actions, actions)

        return actions, Q, target_q, h, c, target_h, target_c

    def update_target_model(self):
        with torch.no_grad():
            for target_param, online_param in zip(self.target_model.parameters(), self.model.parameters()):
                target_param.copy_(online_param)
        gc.collect()
        torch.cuda.empty_cache()

    @staticmethod
    def value_rescale(value, eps=1e-3):
        return value.sign()*((value.abs()+1).sqrt()-1) + eps*value

    @staticmethod
    def inverse_value_rescale(value, eps=1e-3):
        temp = ((1 + 4*eps*(value.abs()+1+eps)).sqrt() - 1) / (2*eps)
        return value.sign() * (temp.square() - 1)

    def learn(self):
        # get all data
        # states: (m+l+n) x b x 1 x 84 x 84
        # actions: (m+l+n) x b
        # rewards: (m+l+n) x b
        # dones: (m+l+n) x b
        # init_h: b x 512
        # init_c: b x 512
        # last_actions: b x 512 x a
        # last_rewards: b x 512
        # masks: (m+l+n) x b
        (states, actions, rewards, dones, init_h, init_c, last_actions, last_rewards, masks), weights, sample_idx = self.per.sample(self.batch_size)
        states = torch.as_tensor(states, device=self.device)
        actions = torch.as_tensor(actions, device=self.device, dtype=torch.long)
        rewards = torch.as_tensor(rewards, device=self.device, dtype=torch.float32)
        dones = torch.as_tensor(dones, device=self.device, dtype=torch.float32)
        masks = torch.as_tensor(masks, device=self.device, dtype=torch.float32)
        weights = torch.as_tensor(weights, device=self.device, dtype=torch.float32)
        last_actions = torch.as_tensor(last_actions, device=self.device)
        last_rewards = torch.as_tensor(last_rewards, device=self.device)

        Qs, target_qs = [], []

        # burn in
        init_h, init_c = torch.tensor(init_h).to(self.device), torch.tensor(init_c).to(self.device)
        h, c = init_h.clone(), init_c.clone()
        target_h, target_c = init_h.clone(), init_c.clone()

        for t in range(self.m):
            with torch.no_grad():
                q, h, c = self.model(states[t], h, c, last_actions[t], last_rewards[t].unsqueeze(-1))
                target_q, target_h, target_c = self.target_model(states[t], target_h, target_c, last_actions[t], last_rewards[t].unsqueeze(-1))
                h, c = h * masks[t].unsqueeze(1), c * masks[t].unsqueeze(1)
                target_h, target_c = target_h * masks[t].unsqueeze(1), target_c * masks[t].unsqueeze(1)
                Qs.append(q)
                target_qs.append(target_q)

        # calculate Q and target_Q
        for t in range(self.m, self.m + self.l):
            q, h, c = self.model(states[t], h, c, last_actions[t], last_rewards[t].unsqueeze(-1))
            with torch.no_grad():
                target_q, target_h, target_c = self.target_model(states[t], target_h, target_c, last_actions[t], last_rewards[t].unsqueeze(-1))
            Qs.append(q)
            target_qs.append(target_q)

        # last n_step
        h = h.detach().clone()
        c = c.detach().clone()

        with torch.no_grad():
            for t in range(self.m + self.l, self.m + self.l + self.n):
                q, h, c = self.model(states[t], h, c, last_actions[t], last_rewards[t].unsqueeze(-1))
                target_q, target_h, target_c = self.target_model(states[t], target_h, target_c, last_actions[t], last_rewards[t].unsqueeze(-1))
                Qs.append(q)
                target_qs.append(target_q)

        # calculate target
        idx = torch.arange(0, len(h), device = self.device)
        targets = []
        with torch.no_grad():
            for t in range(self.m + self.l - 1, self.m-1, -1):
                max_action = Qs[t + self.n].argmax(-1)
                target = self.inverse_value_rescale(target_qs[t + self.n][idx, max_action])

                for k in range(t + self.n - 1, t-1, -1):
                    target = target * self.gamma * (1. - dones[k]) + rewards[k]
                targets.append(self.value_rescale(target))

        # calculate loss
        loss = 0.
        targets = targets[::-1]
        max_delta = torch.zeros((len(idx),)).reshape(-1).to(self.device)
        mean_delta = 0.
        counts = 0.
        for t in range(self.m, self.m + self.l):
            if self.loss_type == 'huber':
                delta = (F.huber_loss(Qs[t][idx, actions[t]], targets[t-self.m], reduction = "none") * masks[t])
            else:
                delta = (F.mse_loss(Qs[t][idx, actions[t]], targets[t-self.m], reduction = "none") * masks[t])
            loss += (delta.reshape(-1) * weights.reshape(-1)).sum()
            counts += masks[t]
            with torch.no_grad():
                delta = (Qs[t][idx, actions[t]] - targets[t-self.m]).abs() * masks[t]
                max_delta = torch.maximum(max_delta, delta.reshape(-1))
                mean_delta = mean_delta + delta.reshape(-1)
        mean_delta = mean_delta / counts
        loss /= counts.sum()

        #update model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        #update loss statis
        self.update_loss_statis(loss.item())

        #update priority
        priorities = self.eta * max_delta + (1-self.eta) * mean_delta
        self.per.update_priorities(sample_idx, priorities)

    def calculate_target(self, rewards, dones, values, target_values):
        targets = []
        T = len(rewards)
        for t in range(T):
            if t + self.n < T:
                max_action = values[t + self.n].argmax(-1)
                target = self.inverse_value_rescale(target_values[t + self.n][max_action])
            else:
                target = 0

            for k in range(min(T-1, t + self.n - 1), t-1, -1):
                target = target * self.gamma * (1. - dones[k].float()) + rewards[k]
            targets.append(self.value_rescale(target))

        return torch.tensor(targets).reshape(-1)

    def calculate_priorities(self, values, targets, actions, sequence_idx):
        T = len(values)
        priorities = []
        idxs = torch.arange(0, len(values), device = self.device)
        values = values[idxs, actions]
        deltas = (values.cpu() - targets).abs()
        for i, idx in enumerate(sequence_idx):
            start = idx
            end = min(T, idx + self.l)
            delta = deltas[start:end]
            max_delta = delta.max()
            mean_delta = delta.mean()
            priorities.append(self.eta * max_delta.item() + (1-self.eta) * mean_delta.item())
        return priorities

    def save_episode(self, data):
        # (T x 1 x 84 x 84) (T) (T) (T) (T a) (T a) (T 512) (T 512) (T 512 a) (T 512) (1)
        states, actions, rewards, dones, values, target_values, h, c, last_actions, last_rewards = data

        states = np.array(states)
        actions, rewards, dones = torch.tensor(actions, dtype=torch.long).to(self.device), torch.tensor(rewards), torch.tensor(dones).to(self.device)
        values, target_values = torch.stack(values, 0).to(self.device), torch.stack(target_values, 0).to(self.device)
        h, c = torch.stack(h, 0).to(self.device), torch.stack(c, 0).to(self.device)
        last_actions, last_rewards = np.array(last_actions), np.array(last_rewards)

        T = len(states)
        sequence_idx = torch.arange(0, T, self.l).reshape(-1) # (S)
        burnin_sequence_idx = torch.clamp(sequence_idx - self.m, min=0)
        end_sequence_idx = torch.clamp(sequence_idx + self.l + self.n - 1, max=T - 1)

        targets = self.calculate_target(rewards, dones, values, target_values) # (T 512)
        priorities = self.calculate_priorities(values, targets, actions, sequence_idx) # (S)

        h, c = h[burnin_sequence_idx].detach(), c[burnin_sequence_idx].detach()

        self.per.add((states, actions.cpu(), rewards.cpu(), dones.cpu(), h.cpu(), c.cpu(), last_actions, last_rewards),
                     burnin_sequence_idx, sequence_idx, end_sequence_idx, priorities)

    def train(self):
        episode_reward = [0] * self.num_envs
        episode_step = [0] * self.num_envs
        max_episode_reward = 0
        max_episode_step = 0
        episode_time = [datetime.now() for _ in range(self.num_envs)]
        total_time = datetime.now()

        last_episode_rewards = []

        #reset envs
        states = self.envs.reset() #list n, 1, 84, 84

        # create h, c as zeros
        h = torch.zeros((self.num_envs, 512), dtype=torch.float, device = self.device) # n 512
        c = torch.zeros((self.num_envs, 512), dtype=torch.float, device = self.device) # n 512

        target_h = torch.zeros((self.num_envs, 512), dtype=torch.float, device = self.device) # n 512
        target_c = torch.zeros((self.num_envs, 512), dtype=torch.float, device = self.device) # n 512

        episode_memorys = Episode_memory(self.num_envs, self.m, self.l, self.n)

        self.update_target_model()

        last_actions = torch.zeros((self.num_envs, self.action_dim), device = self.device) # n a
        last_rewards = torch.zeros((self.num_envs,), device = self.device).reshape(-1) # n

        while True:
            # finish training if agent reach total_step or total_episode base on what type of total_step_or_episode is step or episode
            self.current_step += 1

            if self.total_step_or_episode == 'step':
                if self.current_step >= self.total_step:
                    break
            else:
                if self.current_episode >= self.total_episode:
                    break

            # (n) (n x a), (n x a), (n x 512), (n x 512), (n x 512), (n x 512)
            actions, values, target_values, new_h, new_c, new_target_h, new_target_c = self.select_action(states, h, c, target_h, target_c,
                                                                                                          last_actions, last_rewards)

            # (n, 1, 84, 84) (n, 1, 84, 84) (n,) (n,)
            next_states, next_states_, rewards, dones, truncs, infos = self.envs.step(actions.cpu().numpy())

            # save to episode memorys
            episode_memorys.save(states, actions, rewards, dones, values, target_values, h, c, last_actions, last_rewards)

            last_actions = F.one_hot(actions.long().reshape(-1),
                                     self.action_dim).to(self.device).float()
            last_rewards = torch.as_tensor(np.array(rewards), device=self.device).reshape(-1).float()

            episode_reward = [x + reward for x, reward in zip(episode_reward, rewards)]
            episode_step = [x+1 for x in episode_step]

            # reset h and c to zeros for enviroments that just ending episode, just multiply h and c with (1-dones)
            h = new_h * (1 - torch.tensor(dones, device = self.device, dtype = torch.float).reshape(-1, 1))
            c = new_c * (1 - torch.tensor(dones, device = self.device, dtype = torch.float).reshape(-1, 1))

            target_h = new_target_h * (1 - torch.tensor(dones, device = self.device, dtype = torch.float).reshape(-1, 1))
            target_c = new_target_c * (1 - torch.tensor(dones, device = self.device, dtype = torch.float).reshape(-1, 1))

             # logging after each step, if 1 episode is ending, I will log this to logging.txt
            for i, done in enumerate(dones):
                if done:
                    self.current_episode += 1
                    max_episode_reward = max(max_episode_reward, episode_reward[i])
                    max_episode_step = max(max_episode_step, episode_step[i])
                    last_episode_rewards.append(episode_reward[i])
                    f_out = open(f"logging.txt", "a")
                    f_out.write(f'episode: {self.current_episode} agent: {i} epsilon: {self.epsilons[i]:.4f} rewards: {episode_reward[i]:.4f} steps: {episode_step[i]} \
complete: {infos[i]["flag_get"]==True} mean_rewards: {np.array(last_episode_rewards[-min(len(last_episode_rewards), 100):]).mean():.4f} \
max_rewards: {max_episode_reward:.4f} max_steps: {max_episode_step} current_step: {self.current_step} loss: {(self.losses.sum()/self.len_loss):.4f} \
len_per: {self.per.real_size} episode_time: {datetime.now() - episode_time[i]} total_time: {datetime.now() - total_time}\n')
                    f_out.close()
                    episode_reward[i] = 0
                    episode_step[i] = 0
                    episode_time[i] = datetime.now()

                    self.save_episode(episode_memorys.get_data(i))
                    episode_memorys.reset(i)

                    last_actions[i] = last_actions[i] * 0
                    last_rewards[i] = 0

            # training agent every learn_step
            if (self.current_step % self.learn_step == 0 and
                self.current_step * self.num_envs >= self.start_learning_step and
                self.per.real_size >= self.start_learning_sequence and
                self.per.real_size > self.batch_size):
                self.learn()

            # update target model every target_update_freq
            if self.current_step % (self.target_update_freq * self.learn_step) == 0:
                self.update_target_model()

            # eval agent every save_figure_step
            if self.current_step % self.save_figure_step == 0:
                self.save_figure(is_training=True)
                if self.is_completed:
                    return

            if self.current_step % self.save_model_step == 0:
                self.save_model()

            states = list(next_states_)

        f_out = open(f"logging.txt", "a")
        f_out.write(f' mean_rewards: {np.array(last_episode_rewards[-min(len(last_episode_rewards), 100):]).mean()} max_rewards: {max_episode_reward} \
max_steps: {max_episode_step} current_step: {self.current_step} total_time: {datetime.now() - total_time}\n')
        f_out.close()