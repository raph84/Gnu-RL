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

from PPO_AGENT import PPO

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
