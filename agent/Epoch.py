import math
import torch
import torch.utils.data as data
from utils import Dataset, Advantage_func
import numpy as np


class Epoch():


    def __init__(self, i_episode,tol_eps, state, gamma, save_name):
        self.i_episode = i_episode
        self.tol_eps = tol_eps
        self.n_step = 96  #Timesteps per day(episode/epoch)
        self.n_step_sec = 900 #900 seconds per step
        self.log_probs = []
        self.rewards = []
        self.real_rewards = []
        self.old_log_probs = []
        self.states = [state]
        self.disturbance = []
        self.actions = []  # Save for Parameter Updates
        self.CC = []
        self.cc = []
        self.sigma = 1 - 0.9 * i_episode / tol_eps
        self.gamma = gamma
        self.save_name = save_name

    def epoch_step(
        self,
        agent,
        current, cur_time):

        step = self.get_step(cur_time)
        sigma = 1 - 0.9 * self.i_episode / self.tol_eps

        return agent.do_action(current, self.n_step, cur_time,sigma)

    def get_step(self,cur_time):
        # 96 timesteps per day; 900 seconds per step; which timestep are we?
        seconds_since_midnight = (cur_time - cur_time.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
        step = math.floor(seconds_since_midnight / self.n_step_sec)
        return step

    def epoch_done(self, agent, update_episode, cur_time):

        perf = []

        advantages = Advantage_func(self.rewards, self.gamma)
        old_log_probs = torch.stack(
            self.old_log_probs).squeeze().detach().clone()
        next_states = torch.stack(self.states[1:]).squeeze(1)
        states = torch.stack(self.states[:-1]).squeeze(1)
        actions = torch.stack(self.actions).squeeze(1).detach().clone()
        CC = torch.stack(self.CC).squeeze()  # n_batch x T x (m+n) x (m+n)
        cc = torch.stack(self.cc).squeeze()  # n_batch x T x (m+n)
        disturbance = torch.stack(self.disturbance)  # n_batch x T x n_dist
        agent.memory.append(states, actions, next_states, advantages, old_log_probs, disturbance, CC, cc)

        # if -1, do not update parameters
        if update_episode == -1:
            print("Pass")
            #pass
        elif (agent.memory.len >= update_episode) & (self.i_episode %
                                                     update_episode == 0):
            batch_states, batch_actions, b_next_states, batch_dist, batch_rewards, batch_old_logprobs, batch_CC, batch_cc = agent.memory.sample_batch(update_episode)
            batch_set = Dataset(batch_states, batch_actions, b_next_states, batch_dist, batch_rewards, batch_old_logprobs, batch_CC, batch_cc)
            batch_loader = data.DataLoader(batch_set, batch_size=48, shuffle=True, num_workers=2)
            agent.update_parameters(batch_loader, self.sigma)

        perf.append([np.mean(self.real_rewards), np.std(self.real_rewards)])
        print("{}, reward: {}".format(cur_time, np.mean(self.real_rewards)))
