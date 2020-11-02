import debugpy

import os
import sys

import gym
import eplus_env

# Assign mpc_path to be the file path where mpc.torch is located.
mpc_path = os.path.abspath(os.path.join(__file__,'..', '..'))
sys.path.insert(0, mpc_path)

from diff_mpc import mpc
from diff_mpc.mpc import QuadCost, LinDx

import argparse
import numpy as np
import pandas as pd
import copy
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from torch.distributions import MultivariateNormal, Normal

from utils import make_dict, R_func, Advantage_func, Replay_Memory, Dataset

from flask import Flask, url_for, request
import json


app = Flask(__name__)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


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
parser.add_argument('--debug_ppo', type=str_to_bool, nargs='?', const=True, default=False,
                    help='Activate debugpy')
parser.add_argument('--api_mode', type=str_to_bool, nargs='?', const=True, default=False,
                    help='Flask API')
parser.add_argument('--eps', type=int, default=90,
                    help='Total number of episode. Each episode is a natural day')
parser.add_argument('--no-reload', type=str_to_bool, nargs='?', const=True, default=False,
                    help='reload')
parser.add_argument('--weights_imit', type=str_to_bool, nargs='?', const=True, default=True,
                    help='Use weights saved from imitation learning')
args = parser.parse_args()

    

if args.debug_ppo :
    # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
    debugpy.listen(("127.0.0.1",5679))
    print("Waiting for debugger attach")
    if not args.api_mode:
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
        debugpy.breakpoint()
        print('break on this line')

class PPO(nn.Module):

    class P():

        def __init__(self):
            #if self.start_time == None:
            #    self.start_time = datetime.now()
            
            self.observations = [] ##
            self.actions_taken = [] ##

            self.actions = [] #
            self.states = [] #

            self.start_time = None
            self.timestamp = [] ##

            self.perf = [] ##

            self.rewards = [] #
            self.real_rewards = [] #

            self.old_log_probs = [] #

            self.disturbances = [] #

            self.CC = [] #
            self.cc = [] #

    def __init__(self, memory, T, n_ctrl, n_state, target, disturbance, eta, u_upper, u_lower, clip_param = 0.1, F_hat = None, Bd_hat = None):

        super(PPO, self).__init__()

        self.p = self.P()

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

        if F_hat is not None: # Load pre-trained F if provided
            print("Load pretrained F")
            self.F_hat = torch.tensor(F_hat).double().requires_grad_()
            print(self.F_hat)
        else:
            self.F_hat = torch.ones((self.n_state, self.n_state+self.n_ctrl))
            self.F_hat = self.F_hat.double().requires_grad_()

        if Bd_hat is not None:  # Load pre-trained Bd if provided
            print("Load pretrained Bd")
            self.Bd_hat = Bd_hat
        else:
            self.Bd_hat =  0.1 * np.random.rand(self.n_state, self.n_dist)
        self.Bd_hat = torch.tensor(self.Bd_hat).requires_grad_()
        print(self.Bd_hat)

        self.Bd_hat_old = self.Bd_hat.detach().clone()
        self.F_hat_old = self.F_hat.detach().clone()

        self.optimizer = optim.RMSprop([self.F_hat, self.Bd_hat], lr=args.lr)

        self.u_lower = u_lower * torch.ones(n_ctrl).double()
        self.u_upper = u_upper * torch.ones(n_ctrl).double()

    

        

    # Use the "current" flag to indicate which set of parameters to use
    def forward(self, x_init, ft, C, c, current = True, n_iters=20):
        T, n_batch, n_dist = ft.shape
        if current == True:
            F_hat = self.F_hat
            Bd_hat = self.Bd_hat
        else:
            F_hat = self.F_hat_old
            Bd_hat = self.Bd_hat_old

        x_lqr, u_lqr, objs_lqr = mpc.MPC(n_state=self.n_state,
                                         n_ctrl=self.n_ctrl,
                                         T=self.T,
                                         u_lower= self.u_lower.repeat(self.T, n_batch, 1),
                                         u_upper= self.u_upper.repeat(self.T, n_batch, 1),
                                         lqr_iter=n_iters,
                                         backprop = True,
                                         verbose=0,
                                         exit_unconverged=False,
                                         )(x_init.double(), QuadCost(C.double(), c.double()),
                                           LinDx(F_hat.repeat(self.T-1, n_batch, 1, 1), ft.double()))
        return x_lqr, u_lqr

    def select_action(self, mu, sigma):
        if self.n_ctrl > 1:
            sigma_sq = torch.ones(mu.size()).double() * sigma**2
            dist = MultivariateNormal(mu, torch.diag(sigma_sq.squeeze()).unsqueeze(0))
        else:
            dist = Normal(mu, torch.ones_like(mu)*sigma)
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
            dist = Normal(mu, torch.ones_like(mu)*sigma)
        log_prob = dist.log_prob(actions.double())
        entropy = dist.entropy()
        return log_prob, entropy

    def update_parameters(self, loader, sigma):
        for i in range(1):
            for states, actions, next_states, dist, advantage, old_log_probs, C, c in loader:
                n_batch = states.shape[0]
                advantage = advantage.double()
                f = self.Dist_func(dist, current = True) # T-1 x n_batch x n_state
                opt_states, opt_actions = self.forward(states, f, C.transpose(0, 1), c.transpose(0, 1), current = True) # x, u: T x N x Dim.
                log_probs, entropies = self.evaluate_action(opt_actions[0], actions, sigma)

                tau = torch.cat([states, actions], 1) # n_batch x (n_state + n_ctrl)
                nState_est = torch.bmm(self.F_hat.repeat(n_batch, 1, 1), tau.unsqueeze(-1)).squeeze(-1) + f[0] # n_batch x n_state
                mse_loss = torch.mean((nState_est - next_states)**2)

                ratio = torch.exp(log_probs.squeeze()-old_log_probs)
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1-self.clip_param, 1+self.clip_param) * advantage
                loss  = -torch.min(surr1, surr2).mean()
                self.optimizer.zero_grad()
                loss.backward()
                #nn.utils.clip_grad_norm_([self.F_hat, self.Bd_hat], 100)
                self.optimizer.step()

            self.F_hat_old = self.F_hat.detach().clone()
            self.Bd_hat_old = self.Bd_hat.detach().clone()
            print(self.F_hat)
            print(self.Bd_hat)

    def Dist_func(self, d, current = False):
        if current: # d in n_batch x n_dist x T-1
            n_batch = d.shape[0]
            ft = torch.bmm(self.Bd_hat.repeat(n_batch, 1, 1), d) # n_batch x n_state x T-1
            ft = ft.transpose(1,2) # n_batch x T-1 x n_state
            ft = ft.transpose(0,1) # T-1 x n_batch x n_state
        else: # d in n_dist x T-1
            ft = torch.mm(self.Bd_hat_old, d).transpose(0, 1) # T-1 x n_state
            ft = ft.unsqueeze(1) # T-1 x 1 x n_state
        return ft

    def Cost_function(self, cur_time):
        diag = torch.zeros(self.T, self.n_state+self.n_ctrl)
        occupied = self.dist["Occupancy Flag"][cur_time : cur_time + pd.Timedelta(seconds = (self.T-1) * self.step)]  # T
        eta_w_flag = torch.tensor([self.eta[int(flag)] for flag in occupied]).unsqueeze(1).double() # Tx1
        diag[:, :self.n_state] = eta_w_flag
        diag[:, self.n_state:] = 1e-6

        C = []
        for i in range(self.T):
            C.append(torch.diag(diag[i]))
        C = torch.stack(C).unsqueeze(1) # T x 1 x (m+n) x (m+n)

        x_target = self.target[cur_time : cur_time + pd.Timedelta(seconds = (self.T-1) * self.step)] # in pd.Series
        x_target = torch.tensor(np.array(x_target))

        c = torch.zeros(self.T, self.n_state+self.n_ctrl) # T x (m+n)
        c[:, :self.n_state] = -eta_w_flag*x_target
        c[:, self.n_state:] = 1 # L1-norm now!

        c = c.unsqueeze(1) # T x 1 x (m+n)
        return C, c

class Env:

    timeStep = None
    obs = None
    isTerminal = None

    start_year = None
    start_mon = None
    start_day = None

    def __init__(self, start_year, start_mon, start_day):
        self.start_year = start_year
        self.start_mon = start_mon
        self.start_day = start_day

    def reset(self):
        return self.timeStep, self.obs, self.isTerminal

# Modify here: Outputs from EnergyPlus; Match the variables.cfg file.
obs_name = ["Outdoor Temp.", "Outdoor RH", "Wind Speed", "Wind Direction", "Diff. Solar Rad.", "Direct Solar Rad.", "Htg SP", "Clg SP", "Indoor Temp.", "Indoor Temp. Setpoint", "PPD", "Occupancy Flag", "Coil Power", "HVAC Power", "Sys In Temp.", "Sys In Mdot", "OA Temp.", "OA Mdot", "MA Temp.", "MA Mdot", "Sys Out Temp.", "Sys Out Mdot"]

# Modify here: Change based on the specific control problem
state_name = ["Indoor Temp."]
dist_name = ["Outdoor Temp.", "Outdoor RH", "Wind Speed", "Wind Direction", "Diff. Solar Rad.", "Direct Solar Rad.", "Occupancy Flag"]
# Caveat: The RL agent controls the difference between Supply Air Temp. and Mixed Air Temp., i.e. the amount of heating from the heating coil. But, the E+ expects Supply Air Temp. Setpoint.
ctrl_name = ["SA Temp Setpoint"]
target_name = ["Indoor Temp. Setpoint"]

def next_path(path_pattern, n=0):
    """
    Finds the next free path in an sequentially named list of files

    e.g. path_pattern = 'file-%s.txt':

    file-1.txt
    file-2.txt
    file-3.txt

    Runs in log(n) time where n is the number of existing files in sequence
    """
    i = 1

    # First do an exponential search
    while os.path.exists(path_pattern % i):
        i = i * 2

    # Result lies somewhere in the interval (i/2..i]
    # We call this interval (a..b] and narrow it down until a + 1 = b
    a, b = (i // 2, i)
    while a + 1 < b:
        c = (a + b) // 2  # interval midpoint
        a, b = (c, b) if os.path.exists(path_pattern % c) else (a, c)

    return path_pattern % (b-n)


@app.route('/mpc/', methods=['POST'])
def mpc_api():

    request_json = request.get_json()
    year = request_json['year']
    month = request_json['month']
    day = request_json['day']
    env_api = Env(start_year=year, start_mon=month, start_day=day)

    weather = request_json['weather']
    print(json.dumps(request_json))
    timeStamp = []
    for x in weather:
        timeStamp.append(x['dt'])
        del x['dt']

    obs_name = [
        "Outdoor Temp.", "Outdoor RH", "Wind Speed", "Wind Direction",
        "Diff. Solar Rad.", "Direct Solar Rad.", "Htg SP", "Clg SP",
        "Indoor Temp.", "Indoor Temp. Setpoint", "PPD", "Occupancy Flag",
        "Coil Power", "HVAC Power", "Sys In Temp.", "Sys In Mdot", "OA Temp.",
        "OA Mdot", "MA Temp.", "MA Mdot", "Sys Out Temp.", "Sys Out Mdot"
    ]

    env_api.obs = weather[0]
    main(env=env_api)

    return ('', 204)


def main(env=None):

    # Modify here: Outputs from EnergyPlus; Match the variables.cfg file.
    obs_name = ["Outdoor Temp.", "Outdoor RH", "Wind Speed", "Wind Direction", "Diff. Solar Rad.", "Direct Solar Rad.", "Htg SP", "Clg SP", "Indoor Temp.", "Indoor Temp. Setpoint", "PPD", "Occupancy Flag", "Coil Power", "HVAC Power", "Sys In Temp.", "Sys In Mdot", "OA Temp.", "OA Mdot", "MA Temp.", "MA Mdot", "Sys Out Temp.", "Sys Out Mdot"]

    # Modify here: Change based on the specific control problem
    state_name = ["Indoor Temp."]
    dist_name = ["Outdoor Temp.", "Outdoor RH", "Wind Speed", "Wind Direction", "Direct Solar Rad.", "Occupancy Flag"]
    # Caveat: The RL agent controls the difference between Supply Air Temp. and Mixed Air Temp., i.e. the amount of heating from the heating coil. But, the E+ expects Supply Air Temp. Setpoint.
    ctrl_name = ["SA Temp Setpoint"]
    target_name = ["Indoor Temp. Setpoint"]

    if env is None:
        # Create Simulation Environment
        env = gym.make('7Zone-control_TMYx-v0')

    n_state = len(state_name)
    n_ctrl = len(ctrl_name)

    eta = [0.1, args.eta] # eta: Weight for comfort during unoccupied and occupied mode
    step = args.step # step: Timestep; Unit in seconds
    T = args.T # T: Number of timesteps in the planning horizon
    tol_eps =  args.eps # tol_eps: Total number of episodes; Each episode is a natural day

    u_upper = 5
    u_lower = 0

    # Read Information on Weather, Occupancy, and Target Setpoint
    obs = pd.read_pickle("results/Dist-TMY2.pkl")
    target = obs[target_name]
    disturbance = obs[dist_name]

    # Min-Max Normalization
    disturbance = (disturbance - disturbance.min())/(disturbance.max() - disturbance.min())

    torch.manual_seed(args.seed)
    memory = Replay_Memory()


    # From Imitation Learning
    if args.weights_imit:
        print("Loading weights from Imitation learning.")
        epoch = 19
        imit_F_path = next_path("results/weights/F-%s.npy",1)
        imit_Bd_path = next_path("results/weights/Bd-%s.npy",1)
        F_hat = np.load(imit_F_path)
        Bd_hat = np.load(imit_Bd_path)
    else:
        print("Loading weights from last PPO execution.")
        imit_F_path = next_path("results/weights/ppo_F-%s.npy", 1)
        imit_Bd_path = next_path("results/weights/ppo_Bd-%s.npy", 1)
        print("F path : {}".format(imit_F_path))
        print("Bd path : {}".format(imit_Bd_path))
        F_hat = np.load(imit_F_path)
        Bd_hat = np.load(imit_Bd_path)

    ## After first round of training
    #F_hat = np.array([[0.9248, 0.1440]])
    #Bd_hat = np.array([[0.7404, 0.1490, 0.3049, 0.5458, 0.2676, 0.3085, 0.6900]])
    agent = PPO(memory, T, n_ctrl, n_state, target, disturbance, eta, u_upper, u_lower, F_hat = F_hat, Bd_hat = Bd_hat)

    dir = 'results'
    if not os.path.exists(dir):
        os.mkdir(dir)

    perf = []
    multiplier = 10 # Normalize the reward for better training performance
    n_step = 96 #timesteps per day

    timeStep, obs, isTerminal = env.reset()
    start_time = pd.datetime(year = env.start_year, month = env.start_mon, day = env.start_day)
    cur_time = start_time
    print(cur_time)
    obs_dict = make_dict(obs_name, obs)

    drop_keys = [
        "Diff. Solar Rad.", "Clg SP", "Sys In Temp.", "Sys In Mdot", "OA Temp.",
        "HVAC Power", "MA Mdot", "OA Mdot", "Sys Out Mdot"
    ]
    for k in drop_keys:
        del obs_dict[k]
    obs_name_filter = list(obs_dict.keys())

    state = torch.tensor([obs_dict[name] for name in state_name]).unsqueeze(0).double() # 1 x n_state

    # Save for record
    timeStamp = [start_time]
    observations = [list(obs_dict.values())]
    actions_taken = []

    for i_episode in range(tol_eps):
        print("Episode {} of {}".format(i_episode,tol_eps))
        log_probs = []
        rewards = []
        real_rewards = []
        old_log_probs = []
        states = [state]
        disturbance = []
        actions = [] # Save for Parameter Updates
        CC = []
        cc = []
        sigma = 1 - 0.9*i_episode/tol_eps

        for t in range(n_step):
            print("Step {} of {}".format(t,n_step))
            dt = np.array(agent.dist[cur_time : cur_time + pd.Timedelta(seconds = (agent.T-2) * agent.step)]) # T-1 x n_dist
            dt = torch.tensor(dt).transpose(0, 1) # n_dist x T-1
            ft = agent.Dist_func(dt) # T-1 x 1 x n_state
            C, c = agent.Cost_function(cur_time)
            opt_states, opt_actions = agent.forward(state, ft, C, c, current = False) # x, u: T x 1 x Dim.
            action, old_log_prob = agent.select_action(opt_actions[0], sigma)

            # Modify here based on the specific control problem.
            # Caveat: I send the Supply Air Temp. Setpoint to the Gym-Eplus interface. But, the RL agent controls the difference between Supply Air Temp. and Mixed Air Temp., i.e. the amount of heating from the heating coil.
            SAT_stpt = obs_dict["MA Temp."] + max(0, action.item())
            if action.item()<0:
                action = torch.zeros_like(action)
            # If the room gets too warm during occupied period, uses outdoor air for free cooling.
            if (obs_dict["Indoor Temp."]>obs_dict["Indoor Temp. Setpoint"]) & (obs_dict["Occupancy Flag"]==1):
                SAT_stpt = obs_dict["Outdoor Temp."]
            timeStep, obs, isTerminal = env.step([SAT_stpt])

            obs_dict = make_dict(obs_name, obs)

            drop_keys = [
                "Diff. Solar Rad.", "Clg SP", "Sys In Temp.", "Sys In Mdot", "OA Temp.",
                "HVAC Power", "MA Mdot","OA Mdot", "Sys Out Mdot"
            ]
            for k in drop_keys:
                del obs_dict[k]

            cur_time = start_time + pd.Timedelta(seconds = timeStep)
            reward = R_func(obs_dict, action, eta)

            # Per episode
            real_rewards.append(reward)
            rewards.append(reward.double() / multiplier)
            state = torch.tensor([obs_dict[name] for name in state_name]).unsqueeze(0).double()
            actions.append(action)
            old_log_probs.append(old_log_prob)
            states.append(state)
            disturbance.append(dt)
            CC.append(C.squeeze())
            cc.append(c.squeeze())

            # Save for record
            timeStamp.append(cur_time)
            observations.append(list(obs_dict.values()))
            actions_taken.append([action.item(), SAT_stpt])
            print("{}, Action: {}, SAT Setpoint: {}, Actual SAT:{}, State: {}, Target: {}, Occupied: {}, Reward: {}".format(cur_time,
                action.item(), SAT_stpt, obs_dict["Sys Out Temp."], obs_dict["Indoor Temp."], obs_dict["Indoor Temp. Setpoint"], obs_dict["Occupancy Flag"], reward))

        advantages = Advantage_func(rewards, args.gamma)
        old_log_probs = torch.stack(old_log_probs).squeeze().detach().clone()
        next_states = torch.stack(states[1:]).squeeze(1)
        states = torch.stack(states[:-1]).squeeze(1)
        actions = torch.stack(actions).squeeze(1).detach().clone()
        CC = torch.stack(CC).squeeze() # n_batch x T x (m+n) x (m+n)
        cc = torch.stack(cc).squeeze() # n_batch x T x (m+n)
        disturbance = torch.stack(disturbance) # n_batch x T x n_dist
        agent.memory.append(states, actions, next_states, advantages, old_log_probs, disturbance, CC, cc)

        # if -1, do not update parameters
        if args.update_episode == -1:
            print("Pass")
            pass
        elif (agent.memory.len>= args.update_episode)&(i_episode % args.update_episode ==0):
            batch_states, batch_actions, b_next_states, batch_dist, batch_rewards, batch_old_logprobs, batch_CC, batch_cc = agent.memory.sample_batch(args.update_episode)
            batch_set = Dataset(batch_states, batch_actions, b_next_states, batch_dist, batch_rewards, batch_old_logprobs, batch_CC, batch_cc)
            batch_loader = data.DataLoader(batch_set, batch_size=48, shuffle=True, num_workers=2)
            agent.update_parameters(batch_loader, sigma)

        perf.append([np.mean(real_rewards), np.std(real_rewards)])
        print("{}, reward: {}".format(cur_time, np.mean(real_rewards)))

        save_name = args.save_name
        obs_df = pd.DataFrame(np.array(observations), index = np.array(timeStamp), columns = obs_name_filter)
        action_df = pd.DataFrame(np.array(actions_taken), index = np.array(timeStamp[:-1]), columns = ["Delta T", "Supply Air Temp. Setpoint"])
        obs_df.to_pickle("results/perf_"+save_name+"_obs.pkl")
        action_df.to_pickle("results/perf_"+save_name+"_actions.pkl")
        pickle.dump(np.array(perf), open("results/perf_"+save_name+".npy", "wb"))

        # Save weights
        F_path = next_path("results/weights/ppo_F-%s.npy")
        Bd_path = next_path("results/weights/ppo_Bd-%s.npy")
        F_hat = agent.F_hat.detach().numpy()
        Bd_hat = agent.Bd_hat.detach().numpy()
        np.save(F_path, F_hat)
        np.save(Bd_path, Bd_hat)

    torch.save(agent, 'torch_model.pth')

if __name__ == '__main__':
    if args.api_mode:
        if __name__ == "__main__":
            app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
    else:
        main()
