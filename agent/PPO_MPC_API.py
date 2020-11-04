import numpy as np
import pandas as pd
import copy
import pickle
from datetime import datetime
import math
import warnings
import argparse

import os
import sys

import torch

from flask import Flask, url_for, request
import json

from utils import make_dict, R_func, Advantage_func, Replay_Memory, Dataset, next_path, str_to_bool, drop_keys

from PPO_AGENT import PPO

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
parser.add_argument('--eps', type=int, default=90,
                    help='Total number of episode. Each episode is a natural day')
parser.add_argument('--save_agent', type=bool, default=False, const=True, nargs="?",
                    help='Save updates of the agent.')
args, unknown = parser.parse_known_args()


# Modify here: Outputs from EnergyPlus; Match the variables.cfg file.
obs_name = ["Outdoor Temp.", "Outdoor RH", "Wind Speed", "Wind Direction", "Direct Solar Rad.", "Htg SP", "Indoor Temp.", "Indoor Temp. Setpoint", "PPD", "Occupancy Flag", "Coil Power", "MA Temp.", "Sys Out Temp."]

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

agent = None

def initialize():

    global agent

    print("Initializing agent...")
    agent = torch.load('torch_model.pth')
    agent.eval()
    if agent.p.start_time == None:
        agent.p.start_time = datetime.now()

@app.route('/mpc/', methods=['POST'])
def mpc_api():

    req = request.get_json()
    pickle.dump(req, open(next_path("results/req-%s.p"), "wb"))
    date_request = datetime.strptime(req['date'], '%Y-%m-%d %H:%M:%S')
    #target = [req['disturbances'][0][k] for k in target_name]

    # TODO : DISTURBANCE ; In relation to obs below?
    d_ = copy.deepcopy(req['disturbances'])
    dist_time = []
    for d in d_:
        dist_time.append(datetime.strptime(d['dt'], '%Y-%m-%d %H:%M:%S'))
        for k in list(d.keys()):
            if k not in dist_name:
                del d[k]

    disturbance = pd.DataFrame(d_, index=dist_time)
    target = pd.DataFrame(req['disturbances'], index=dist_time)[target_name]
    agent.dist = disturbance
    agent.n_dist = agent.dist.shape[1]
    agent.target = target

    dir = 'results'
    if not os.path.exists(dir):
        os.mkdir(dir)

    multiplier = 10  # Normalize the reward for better training performance
    n_step = 96  #timesteps per day






    # TODO : ==== Values from API request ====
    obs_dict = {k: req['disturbances'][0][k] for k in obs_name}

    # TODO : Date from API request
    cur_time = pd.datetime(year = date_request.year, month = date_request.month, day = date_request.day)


    # 96 timesteps per day; which timestep are we?
    step = args.step  # 900 seconds per step
    seconds_since_midnight = (date_request - date_request.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
    t = math.floor(seconds_since_midnight / step)
    print("Step {} of {}".format(t,n_step))


    # One action at a time for the current day
    i_episode = 0
    tol_eps = 1
    sigma = 1 - 0.9*i_episode/tol_eps


    # New day? Update model and reset history arrays
    # TODO : don't rely on NOW. Use date from API request
    if agent.p.start_time.day != date_request.day:

        print("==== Begining new day - Update model - Reset agent ====")

        # Torch variables to append to agent memory
        advantages = Advantage_func(agent.p.rewards, args.gamma)
        old_log_probs = torch.stack(agent.p.old_log_probs).squeeze().detach().clone()
        next_states = torch.stack(agent.p.states[1:]).squeeze(1)
        states = torch.stack(agent.p.states[:-1]).squeeze(1)
        actions = torch.stack(agent.p.actions).squeeze(1).detach().clone()
        CC = torch.stack(agent.p.CC).squeeze() # n_batch x T x (m+n) x (m+n)
        cc = torch.stack(agent.p.cc).squeeze() # n_batch x T x (m+n)
        disturbance = torch.stack(agent.p.disturbances) # n_batch x T x n_dist
        agent.memory.append(states, actions, next_states, advantages, old_log_probs, disturbance, CC, cc)

        if (agent.memory.len>= 1):
            batch_states, batch_actions, b_next_states, batch_dist, batch_rewards, batch_old_logprobs, batch_CC, batch_cc = agent.memory.sample_batch(args.update_episode)
            batch_set = Dataset(batch_states, batch_actions, b_next_states, batch_dist, batch_rewards, batch_old_logprobs, batch_CC, batch_cc)
            batch_loader = data.DataLoader(batch_set, batch_size=48, shuffle=True, num_workers=2)
            agent.update_parameters(batch_loader, sigma)
        else:
            warnings.warn("agent.memory.len should be greater than 0.")

        agent.p.perf.append([np.mean(agent.p.real_rewards), np.std(agent.p.real_rewards)])
        print("{}, reward: {}".format(cur_time, np.mean(agent.p.real_rewards)))

        save_name = agent.p.timestamp.strftime("%Y%m%d_") + args.save_name
        obs_df = pd.DataFrame(np.array(agent.p.observations), index = np.array(timeStamp), columns = obs_name_filter)
        action_df = pd.DataFrame(np.array(agent.p.actions_taken), index = np.array(timeStamp[:-1]), columns = ["Delta T", "Supply Air Temp. Setpoint"])
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

        agent.p = PPO.P()



    agent.p.observations.append([list(obs_dict.values())])
    agent.p.timestamp.append(cur_time)

    state = torch.tensor([obs_dict[name] for name in state_name]).unsqueeze(0).double()
    agent.p.states.append(state)

    # TODO : review dt content
    dt = np.array(agent.dist[cur_time : cur_time + pd.Timedelta(seconds = (agent.T-2) * agent.step)]) # T-1 x n_dist
    dt = torch.tensor(dt).transpose(0, 1) # n_dist x T-1
    ft = agent.Dist_func(dt) # T-1 x 1 x n_state
    C, c = agent.Cost_function(cur_time)
    opt_states, opt_actions = agent.forward(state, ft, C, c, current = False) # x, u: T x 1 x Dim.
    action, old_log_prob = agent.select_action(opt_actions[0], sigma)

    agent.p.old_log_probs.append(old_log_prob)
    agent.p.disturbances.append(dt)
    agent.p.CC.append(C.squeeze())
    agent.p.cc.append(c.squeeze())


    # ==== Current action result ====
    SAT_stpt = max(0, action.item())
    if action.item()<0:
        action = torch.zeros_like(action)
    agent.p.actions.append(action)
    agent.p.actions_taken.append([action.item(), SAT_stpt])
    print("New indoor set point : {}".format(SAT_stpt))

    # If we have 2 observations or more, calculate the reward.
    if len(agent.p.observations) > 1:
        reward = R_func(obs_dict, action, eta)
        agent.p.real_rewards.append(reward)
        agent.p.rewards.append(reward.double() / multiplier)
    else:
        reward = None

    print("{}, Action: {}, SAT Setpoint: {}, Actual SAT:{}, State: {}, Target: {}, Occupied: {}, Reward: {}".format(cur_time,
                action.item(), SAT_stpt, obs_dict["Sys Out Temp."], obs_dict["Indoor Temp."], obs_dict["Indoor Temp. Setpoint"], obs_dict["Occupancy Flag"], reward))
    # if

    if args.save_agent:
        print("Saving agent...")
        torch.save(agent, 'torch_model.pth')


initialize()

if __name__ == "__main__":
    print("Loading Flask API...")
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))