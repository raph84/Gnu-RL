# Helper Functions
import numpy as np
import torch
import torch.utils.data as data

def make_dict(obs_name, obs):
    zipbObj = zip(obs_name, obs)
    return dict(zipbObj)

def R_func(obs_dict, action, eta):
    reward = - 0.5 * eta[int(obs_dict["Occupancy Flag"])] * (obs_dict["Indoor Temp."] - obs_dict["Indoor Temp. Setpoint"])**2 - action
    return reward#.item()
    
# Calculate the advantage estimate
def Advantage_func(rewards, gamma):
    R = torch.zeros(1, 1).double()
    T = len(rewards)
    advantage = torch.zeros((T,1)).double()
    
    for i in reversed(range(len(rewards))):
        R = gamma * R + rewards[i]
        advantage[i] = R
    return advantage

class Dataset(data.Dataset):
    def __init__(self, states, actions, next_states, disturbance, rewards, old_logprobs, CC, cc):
        self.states = states
        self.actions = actions
        self.next_states = next_states
        self.disturbance = disturbance
        self.rewards = rewards
        self.old_logprobs = old_logprobs
        self.CC = CC
        self.cc = cc

    def __len__(self):
        return len(self.states)

    def __getitem__(self, index):
        return self.states[index], self.actions[index], self.next_states[index], self.disturbance[index], self.rewards[index], self.old_logprobs[index], self.CC[index], self.cc[index]
    
class Replay_Memory():
    def __init__(self, memory_size=10):
        self.memory_size = memory_size
        self.len = 0
        self.rewards = []
        self.states = []
        self.n_states = []
        self.log_probs = []
        self.actions = []
        self.disturbance = []
        self.CC = []
        self.cc = []

    def sample_batch(self, batch_size):
        rand_idx = np.arange(-batch_size, 0, 1)
        batch_rewards = torch.stack([self.rewards[i] for i in rand_idx]).reshape(-1)
        batch_states = torch.stack([self.states[i] for i in rand_idx])
        batch_nStates = torch.stack([self.n_states[i] for i in rand_idx])
        batch_actions = torch.stack([self.actions[i] for i in rand_idx])
        batch_logprobs = torch.stack([self.log_probs[i] for i in rand_idx]).reshape(-1)
        batch_disturbance = torch.stack([self.disturbance[i] for i in rand_idx])
        batch_CC = torch.stack([self.CC[i] for i in rand_idx])
        batch_cc = torch.stack([self.cc[i] for i in rand_idx])
        # Flatten
        _, _, n_state =  batch_states.shape
        batch_states = batch_states.reshape(-1, n_state)
        batch_nStates = batch_nStates.reshape(-1, n_state)
        _, _, n_action =  batch_actions.shape
        batch_actions = batch_actions.reshape(-1, n_action)
        _, _, T, n_dist =  batch_disturbance.shape
        batch_disturbance = batch_disturbance.reshape(-1, T, n_dist)
        _, _, T, n_tau, n_tau =  batch_CC.shape
        batch_CC = batch_CC.reshape(-1, T, n_tau, n_tau)
        batch_cc = batch_cc.reshape(-1, T, n_tau)
        return batch_states, batch_actions, batch_nStates, batch_disturbance, batch_rewards, batch_logprobs, batch_CC, batch_cc

    def append(self, states, actions, next_states, rewards, log_probs, dist, CC, cc):
        self.rewards.append(rewards)
        self.states.append(states)
        self.n_states.append(next_states)
        self.log_probs.append(log_probs)
        self.actions.append(actions)
        self.disturbance.append(dist)
        self.CC.append(CC)
        self.cc.append(cc)
        self.len += 1
        
        if self.len > self.memory_size:
            self.len = self.memory_size
            self.rewards = self.rewards[-self.memory_size:]
            self.states = self.states[-self.memory_size:]
            self.log_probs = self.log_probs[-self.memory_size:]
            self.actions = self.actions[-self.memory_size:]
            self.nStates = self.n_states[-self.memory_size:]
            self.disturbance = self.disturbance[-self.memory_size:]
            self.CC = self.CC[-self.memory_size:]
            self.cc = self.cc[-self.memory_size:]

