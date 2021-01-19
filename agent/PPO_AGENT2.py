import os
import sys
from shutil import copyfile
import logging

from google.cloud import storage

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from torch.distributions import MultivariateNormal, Normal

# Assign mpc_path to be the file path where mpc.torch is located.
mpc_path = os.path.abspath(os.path.join(__file__, '..', '..'))
sys.path.insert(0, mpc_path)

from diff_mpc import mpc
from diff_mpc.mpc import QuadCost, LinDx

import numpy as np
import pandas as pd

from utils import make_dict, R_func, Advantage_func, Replay_Memory, Dataset, next_path, str_to_bool, drop_keys
from variables import state_name
from my_quick_dt_util import utcnow, floor_date, get_tz
from PPO_AGENT import PPO as PPO_OLD


repo_model = 'torch_model.pth'
bucket_model = 'torch_model_x.pth'


class PPO(nn.Module):

    class P():
        def __init__(self):
            #if self.start_time == None:
            #    self.start_time = datetime.now()

            self.observations = []  ##
            self.actions_taken = []  ##

            self.actions = []  #
            self.states = []  #

            self.start_time = None
            self.timestamp = []  ##

            self.perf = []  ##

            self.rewards = []  #
            self.real_rewards = []  #

            self.old_log_probs = []  #

            self.disturbances = []  #

            self.CC = []  #
            self.cc = []  #

    def __init__(self,
                 target,
                 disturbance,
                 T=12,
                 n_ctrl=1,
                 n_state=1,
                 eta=[0.1, 4],
                 u_upper=5,
                 u_lower=0,
                 seed_arg=42,
                 clip_param=0.1,
                 F_hat=None,
                 Bd_hat=None,
                 step=900,
                 lr=5e-4):

        super(PPO, self).__init__()

        self.p = self.P()

        torch.manual_seed(seed_arg)
        self.memory=Replay_Memory()

        self.clip_param = clip_param

        self.T = T
        self.step = step
        self.n_ctrl = n_ctrl
        self.n_state = n_state
        self.eta = eta

        self.target = target
        self.dist = disturbance
        self.n_dist = self.dist.shape[1]


        if F_hat is not None:  # Load pre-trained F if provided
            print("Load pretrained F")
            self.F_hat = F_hat.clone().detach().double().requires_grad_()
            print(self.F_hat)
        else:
            self.F_hat = torch.ones((self.n_state, self.n_state + self.n_ctrl))
            self.F_hat = self.F_hat.double().requires_grad_()

        if Bd_hat is not None:  # Load pre-trained Bd if provided
            print("Load pretrained Bd")
            self.Bd_hat = Bd_hat
        else:
            self.Bd_hat = 0.1 * np.random.rand(self.n_state, self.n_dist)
        self.Bd_hat = self.Bd_hat.clone().detach().requires_grad_()
        print(self.Bd_hat)

        self.Bd_hat_old = self.Bd_hat.detach().clone()
        self.F_hat_old = self.F_hat.detach().clone()

        self.optimizer = optim.RMSprop([self.F_hat, self.Bd_hat], lr=lr)

        self.u_lower = u_lower * torch.ones(n_ctrl).double()
        self.u_upper = u_upper * torch.ones(n_ctrl).double()


    def do_action(self, current, n_step, cur_time, sigma):
        #print("Step {} of {}".format(step, n_step))


        state = torch.tensor([current[name]
                              for name in state_name]).unsqueeze(0).double()

        dt = np.array(self.dist[cur_time:cur_time + pd.Timedelta(
            seconds=(self.T - 2) * self.step)])  # T-1 x n_dist
        dt = torch.tensor(dt).transpose(0, 1)  # n_dist x T-1
        ft = self.Dist_func(dt)  # T-1 x 1 x n_state
        C, c = self.Cost_function(cur_time)
        opt_states, opt_actions = self.forward(
            state, ft, C, c, current=False)  # x, u: T x 1 x Dim.
        action, old_log_prob = self.select_action(opt_actions[0], sigma)

        # Modify here based on the specific control problem.
        # Caveat: I send the Supply Air Temp. Setpoint to the Gym-Eplus interface. But, the RL agent controls the difference between Supply Air Temp. and Mixed Air Temp., i.e. the amount of heating from the heating coil.
        #SAT_stpt = obs_dict["MA Temp."] + max(0, action.item())
        if action.item() < 0:
            action = torch.zeros_like(action)

        return action, old_log_prob, state, C, c


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



def find_model_file():
    def check_model(dir_path):
        found = False
        if os.path.exists(os.path.join(dir_path, bucket_model)):
            found = True
        return found

    found = False
    dir_path = os.path.dirname(os.path.realpath(__file__))
    found = check_model(dir_path)

    while not found:
        dir_path = os.path.join(dir_path, '..')
        found = check_model(dir_path)

    return dir_path


def initialize(target, disturbances):

    standalone = os.environ.get('STANDALONE', "")
    if standalone == "True":
        standalone = True
    else:
        standalone = False

    dir_path = find_model_file()
    if standalone:
        copyfile(os.path.join(dir_path, repo_model),
                 os.path.join(dir_path, bucket_model))
    else:
        # Instantiates a client
        bucket_name = os.environ['AGENT_BUCKET']
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blobs = list(
            storage_client.list_blobs(bucket_name, prefix='torch_model'))

        if len(blobs) > 0:
            logging.info('Downloading agent from bucket...')
            blobs[-1].download_to_filename('torch_model_x.pth')
        else:
            logging.info(
                'Agent model not available in bucket, uploading repo version...'
            )
            blob = bucket.blob(bucket_model)
            blob.upload_from_filename(repo_model,
                                      content_type='application/octet-stream')
            os.rename(repo_model, bucket_model)

    logging.info("Initializing agent...")
    model_file = os.path.join(dir_path, bucket_model)
    agent_old = torch.load(model_file)

    if isinstance(agent_old, PPO_OLD):
        agent = PPO(target,
                    disturbances,
                    F_hat=agent_old.F_hat,
                    Bd_hat=agent_old.Bd_hat)
        agent.load_state_dict(agent_old.state_dict())
    else:
        agent = agent_old

        
    agent.eval()

    logging.info('Initial start_time : {}'.format(agent.p.start_time))
    if agent.p.start_time is None:
        agent.p.start_time = floor_date(utcnow(), minutes=15)
        logging.info('Fresh agent, initializing start_time to {}'.format(
            agent.p.start_time))
    else:
        agent.p.start_time = agent.p.start_time.replace(tzinfo=get_tz())
        elapse = utcnow() - agent.p.start_time
        if elapse.days > 2:
            agent_now = floor_date(utcnow(), minutes=15)
            logging.info('Agent too old, initializing start_time to {}'.format(
                agent_now))
            agent.p.start_time = agent_now
    logging.info('start_time : {}'.format(agent.p.start_time))

    return agent