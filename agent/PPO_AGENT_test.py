import os
from shutil import copyfile
import copy
from PPO_AGENT import PPO
from Epoch import Epoch
from EnvApi import EnvApi
from utils import Replay_Memory
import pandas as pd
import torch
from my_quick_dt_util import parse_date
import numpy as np
from tst_payload import payload 


repo_model = 'torch_model.pth'
bucket_model = 'torch_model_x.pth'
state_name = ["Indoor Temp."]

def test_agent():
    dir_path = find_model_file()
    copyfile(os.path.join(dir_path, repo_model),
                 os.path.join(dir_path, bucket_model))

    epoch = Epoch(1, 1, None, 0.98, None)
    envApi = EnvApi(payload()['current'], payload()['disturbances'], parse_date(payload()['date']))

    model_file = os.path.join(dir_path,bucket_model)
    model = torch.load(model_file)

    memory = Replay_Memory()
    agent = PPO(memory, 12, 1, 1, envApi.target, envApi.disturbances, [0.1, 4], 5, 0, 0.1, model.F_hat, model.Bd_hat)
    agent.load_state_dict(model.state_dict())
    cur_time = envApi.date

    state = torch.tensor([envApi.current[name]
                              for name in state_name]).unsqueeze(0).double()

    dt = np.array(agent.dist[cur_time : cur_time + pd.Timedelta(seconds = (agent.T-2) * agent.step)]) # T-1 x n_dist
    dt = torch.tensor(dt).transpose(0, 1) # n_dist x T-1
    ft = agent.Dist_func(dt) # T-1 x 1 x n_state
    C, c = agent.Cost_function(cur_time)
    opt_states, opt_actions = agent.forward(state, ft, C, c, current = False) # x, u: T x 1 x Dim.
    action, old_log_prob = agent.select_action(opt_actions[0], epoch.sigma)

    # Modify here based on the specific control problem.
    # Caveat: I send the Supply Air Temp. Setpoint to the Gym-Eplus interface. But, the RL agent controls the difference between Supply Air Temp. and Mixed Air Temp., i.e. the amount of heating from the heating coil.
    #SAT_stpt = obs_dict["MA Temp."] + max(0, action.item())
    if action.item()<0:
        action = torch.zeros_like(action)

    assert action.item() != 0, action


def find_model_file():
    def check_model(dir_path):
        found = False
        if os.path.exists(os.path.join(dir_path, bucket_model)):
            found = True
        return found

    parent = '..'
    found = False
    dir_path = os.path.dirname(os.path.realpath(__file__))
    found = check_model(dir_path)

    while not found:
        dir_path = os.path.join(dir_path, '..')
        found = check_model(dir_path)

    return dir_path
