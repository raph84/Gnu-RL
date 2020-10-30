import numpy as np
import pandas as pd
import copy
import pickle
from datetime import datetime
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from torch.distributions import MultivariateNormal, Normal

from utils import make_dict, R_func, Advantage_func, Replay_Memory, Dataset, next_path, str_to_bool, drop_keys

from flask import Flask, url_for, request
import json

app = Flask(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE


parser = argparse.ArgumentParser(description='GruRL Demo: Online Learning')
parser.add_argument('--gamma', type=float, default=0.98, metavar='G',
                    help='discount factor (default: 0.98)')
parser.add_argument('--seed', type=int, default=42, metavar='N',
                    help='random seed (default: 42)')
parser.add_argument('--lr', type=float, default=5e-4, metavar='G',
                    help='Learning Rate')
parser.add_argument('--update_episode', type=int, default=1, metavar='N',
                    help='PPO update episode (default: 1); If -1, do not update weights')
parser.add_argument('--T', type=int, default=12, metavar='N',
                    help='Planning Horizon (default: 12)')
parser.add_argument('--step', type=int, default=300*3, metavar='N',
                    help='Time Step in Simulation, Unit in Seconds (default: 900)') # 15 Minutes Now!
parser.add_argument('--save_name', type=str, default='rl',
                    help='save name')
parser.add_argument('--eta', type=int, default=4,
                    help='Hyper Parameter for Balancing Comfort and Energy')


class PPO(nn.Module):
    def __init__(self,
                 memory,
                 T,
                 n_ctrl,
                 n_state,
                 target,
                 disturbance,
                 eta,
                 u_upper,
                 u_lower,
                 clip_param=0.1,
                 F_hat=None,
                 Bd_hat=None):

        super(PPO, self).__init__()

        self.memory = memory
        self.clip_param = clip_param

        self.T = T
        self.step = args.step
        self.n_ctrl = n_ctrl
        self.n_state = n_state
        self.eta = eta

        self.target = target
        self.dist = disturbance
        self.n_dist = self.dist.shape[1]

        if F_hat is not None:  # Load pre-trained F if provided
            print("Load pretrained F")
            self.F_hat = torch.tensor(F_hat).double().requires_grad_()
            print(self.F_hat)
        else:
            self.F_hat = torch.ones((self.n_state, self.n_state + self.n_ctrl))
            self.F_hat = self.F_hat.double().requires_grad_()

        if Bd_hat is not None:  # Load pre-trained Bd if provided
            print("Load pretrained Bd")
            self.Bd_hat = Bd_hat
        else:
            self.Bd_hat = 0.1 * np.random.rand(self.n_state, self.n_dist)
        self.Bd_hat = torch.tensor(self.Bd_hat).requires_grad_()
        print(self.Bd_hat)

        self.Bd_hat_old = self.Bd_hat.detach().clone()
        self.F_hat_old = self.F_hat.detach().clone()

        self.optimizer = optim.RMSprop([self.F_hat, self.Bd_hat], lr=args.lr)

        self.u_lower = u_lower * torch.ones(n_ctrl).double()
        self.u_upper = u_upper * torch.ones(n_ctrl).double()

    # Use the "current" flag to indicate which set of parameters to use
    def forward(self, x_init, ft, C, c, current=True, n_iters=20):
        T, n_batch, n_dist = ft.shape
        if current == True:
            F_hat = self.F_hat
            Bd_hat = self.Bd_hat
        else:
            F_hat = self.F_hat_old
            Bd_hat = self.Bd_hat_old

        x_lqr, u_lqr, objs_lqr = mpc.MPC(
            n_state=self.n_state,
            n_ctrl=self.n_ctrl,
            T=self.T,
            u_lower=self.u_lower.repeat(self.T, n_batch, 1),
            u_upper=self.u_upper.repeat(self.T, n_batch, 1),
            lqr_iter=n_iters,
            backprop=True,
            verbose=0,
            exit_unconverged=False,
        )(x_init.double(), QuadCost(C.double(), c.double()),
          LinDx(F_hat.repeat(self.T - 1, n_batch, 1, 1), ft.double()))
        return x_lqr, u_lqr

    def select_action(self, mu, sigma):
        if self.n_ctrl > 1:
            sigma_sq = torch.ones(mu.size()).double() * sigma**2
            dist = MultivariateNormal(
                mu,
                torch.diag(sigma_sq.squeeze()).unsqueeze(0))
        else:
            dist = Normal(mu, torch.ones_like(mu) * sigma)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

    def evaluate_action(self, mu, actions, sigma):
        n_batch = len(mu)
        if self.n_ctrl > 1:
            cov = torch.eye(self.n_ctrl).double() * sigma**2
            cov = cov.repeat(n_batch, 1, 1)
            dist = MultivariateNormal(mu, cov)
        else:
            dist = Normal(mu, torch.ones_like(mu) * sigma)
        log_prob = dist.log_prob(actions.double())
        entropy = dist.entropy()
        return log_prob, entropy

    def update_parameters(self, loader, sigma):
        for i in range(1):
            for states, actions, next_states, dist, advantage, old_log_probs, C, c in loader:
                n_batch = states.shape[0]
                advantage = advantage.double()
                f = self.Dist_func(dist,
                                   current=True)  # T-1 x n_batch x n_state
                opt_states, opt_actions = self.forward(
                    states,
                    f,
                    C.transpose(0, 1),
                    c.transpose(0, 1),
                    current=True)  # x, u: T x N x Dim.
                log_probs, entropies = self.evaluate_action(
                    opt_actions[0], actions, sigma)

                tau = torch.cat([states, actions],
                                1)  # n_batch x (n_state + n_ctrl)
                nState_est = torch.bmm(
                    self.F_hat.repeat(n_batch, 1, 1),
                    tau.unsqueeze(-1)).squeeze(-1) + f[0]  # n_batch x n_state
                mse_loss = torch.mean((nState_est - next_states)**2)

                ratio = torch.exp(log_probs.squeeze() - old_log_probs)
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_param,
                                    1 + self.clip_param) * advantage
                loss = -torch.min(surr1, surr2).mean()
                self.optimizer.zero_grad()
                loss.backward()
                #nn.utils.clip_grad_norm_([self.F_hat, self.Bd_hat], 100)
                self.optimizer.step()

            self.F_hat_old = self.F_hat.detach().clone()
            self.Bd_hat_old = self.Bd_hat.detach().clone()
            print(self.F_hat)
            print(self.Bd_hat)

    def Dist_func(self, d, current=False):
        if current:  # d in n_batch x n_dist x T-1
            n_batch = d.shape[0]
            ft = torch.bmm(self.Bd_hat.repeat(n_batch, 1, 1),
                           d)  # n_batch x n_state x T-1
            ft = ft.transpose(1, 2)  # n_batch x T-1 x n_state
            ft = ft.transpose(0, 1)  # T-1 x n_batch x n_state
        else:  # d in n_dist x T-1
            ft = torch.mm(self.Bd_hat_old, d).transpose(0, 1)  # T-1 x n_state
            ft = ft.unsqueeze(1)  # T-1 x 1 x n_state
        return ft

    def Cost_function(self, cur_time):
        diag = torch.zeros(self.T, self.n_state + self.n_ctrl)
        occupied = self.dist["Occupancy Flag"][
            cur_time:cur_time +
            pd.Timedelta(seconds=(self.T - 1) * self.step)]  # T
        eta_w_flag = torch.tensor([self.eta[int(flag)] for flag in occupied
                                   ]).unsqueeze(1).double()  # Tx1
        diag[:, :self.n_state] = eta_w_flag
        diag[:, self.n_state:] = 1e-6

        C = []
        for i in range(self.T):
            C.append(torch.diag(diag[i]))
        C = torch.stack(C).unsqueeze(1)  # T x 1 x (m+n) x (m+n)

        x_target = self.target[cur_time:cur_time + pd.Timedelta(
            seconds=(self.T - 1) * self.step)]  # in pd.Series
        x_target = torch.tensor(np.array(x_target))

        c = torch.zeros(self.T, self.n_state + self.n_ctrl)  # T x (m+n)
        c[:, :self.n_state] = -eta_w_flag * x_target
        c[:, self.n_state:] = 1  # L1-norm now!

        c = c.unsqueeze(1)  # T x 1 x (m+n)
        return C, c

def main():

    # Modify here: Outputs from EnergyPlus; Match the variables.cfg file.
    obs_name = ["Outdoor Temp.", "Outdoor RH", "Wind Speed", "Wind Direction", "Diff. Solar Rad.", "Direct Solar Rad.", "Htg SP", "Clg SP", "Indoor Temp.", "Indoor Temp. Setpoint", "PPD", "Occupancy Flag", "Coil Power", "HVAC Power", "Sys In Temp.", "Sys In Mdot", "OA Temp.", "OA Mdot", "MA Temp.", "MA Mdot", "Sys Out Temp.", "Sys Out Mdot"]

    # Modify here: Change based on the specific control problem
    state_name = ["Indoor Temp."]
    dist_name = ["Outdoor Temp.", "Outdoor RH", "Wind Speed", "Wind Direction", "Direct Solar Rad.", "Occupancy Flag"]
    # Caveat: The RL agent controls the difference between Supply Air Temp. and Mixed Air Temp., i.e. the amount of heating from the heating coil. But, the E+ expects Supply Air Temp. Setpoint.
    ctrl_name = ["SA Temp Setpoint"]
    target_name = ["Indoor Temp. Setpoint"]

    n_state = len(state_name)
    n_ctrl = len(ctrl_name)

    eta = [0.1, args.eta] # eta: Weight for comfort during unoccupied and occupied mode
    step = args.step # step: Timestep; Unit in seconds
    T = args.T # T: Number of timesteps in the planning horizon
    tol_eps =  args.eps # tol_eps: Total number of episodes; Each episode is a natural day

    u_upper = 5
    u_lower = 0


    # TODO : TARGET
    target = None
    # TODO : DISTURBANCE ; In relation to obs below?
    disturbance = None

    dir = 'results'
    if not os.path.exists(dir):
        os.mkdir(dir)

    perf = []
    multiplier = 10  # Normalize the reward for better training performance
    n_step = 96  #timesteps per day


    agent = torch.load('torch_model.pth')
    agent.eval()

    

    # TODO : Date from API request
    start_time = pd.datetime(year=2020,
                             month=10,
                             day=29)
    cur_time = start_time

    # TODO : ==== Values from API request ====
    obs_dict = None
    observations.append([list(obs_dict.values())])

    # TODO : if there is a previous observation, calculate the reward
    reward = R_func(obs_dict, action, eta)
    real_rewards.append(reward)
    rewards.append(reward.double() / multiplier)
    
    
    


    # TODO : move this into agent to persist for the day between runs
    observations = []
    actions_taken = []
    actions = []
    states = []

    # TODO : new day? Update model and reset history arrays


    state = torch.tensor([obs_dict[name] for name in state_name])
                                                .unsqueeze(0).double()  # 1 x n_state
    states.append(state)

    timeStamp = [start_time]
    

    # One action at a time for the current day
    i_episode = 0 
    tol_eps = 1
    sigma = 1 - 0.9*i_episode/tol_eps

    # 96 timesteps per day; as which timestep are we?
    step = args.step  # 900 seconds per step
    now = datetime.now()
    seconds_since_midnight = (now - now.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
    t = math.floor(seconds_since_midnight / step)
    print("Step {} of {}".format(t,n_step))

    # TODO : review dt content
    dt = np.array(agent.dist[cur_time : cur_time + pd.Timedelta(seconds = (agent.T-2) * agent.step)]) # T-1 x n_dist
    dt = torch.tensor(dt).transpose(0, 1) # n_dist x T-1
    ft = agent.Dist_func(dt) # T-1 x 1 x n_state
    C, c = agent.Cost_function(cur_time)
    opt_states, opt_actions = agent.forward(state, ft, C, c, current = False) # x, u: T x 1 x Dim.
    action, old_log_prob = agent.select_action(opt_actions[0], sigma)
    old_log_probs.append(old_log_prob)
    disturbance.append(dt)
    CC.append(C.squeeze())
    cc.append(c.squeeze())

    # ==== Current action result ====
    SAT_stpt = max(0, action.item())
    if action.item()<0:
        action = torch.zeros_like(action)
    actions.append(action)
    actions_taken.append([action.item(), SAT_stpt])
    
    print("{}, Action: {}, SAT Setpoint: {}, Actual SAT:{}, State: {}, Target: {}, Occupied: {}, Reward: {}".format(cur_time,
                action.item(), SAT_stpt, obs_dict["Sys Out Temp."], obs_dict["Indoor Temp."], obs_dict["Indoor Temp. Setpoint"], obs_dict["Occupancy Flag"], reward))
    # if

