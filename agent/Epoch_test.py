import os
from Epoch import Epoch
from my_quick_dt_util import parse_date
from PPO_AGENT2 import initialize
from EnvApi import EnvApi
from tst_payload import payload

def test_get_step():
    epoch = Epoch(1, 1, None, 0.98, None)
    date = parse_date('2021-01-18T00:00:00-05:00')
    step0 = epoch.get_step(date)
    date = parse_date('2021-01-18T00:15:00-05:00')
    step1 = epoch.get_step(date)
    date = parse_date('2021-01-18T23:45:00-05:00')
    step95 = epoch.get_step(date)
    date = parse_date('2021-01-18T23:35:00-05:00')
    step94 = epoch.get_step(date)

    assert step0 == 0, step0
    assert step1 == 1, step1
    assert step95 == 95, step95
    assert step94 == 94, step94


def test_epoch_step():
    os.environ['STANDALONE'] = "True"
    os.environ['AGENT_BUCKET'] = "gnu-rl-agent"

    epoch = Epoch(1, 1, None, 0.98, None)
    envApi = EnvApi(payload()['current'], payload()['disturbances'], parse_date(payload()['date']))
    agent = initialize(envApi.target, envApi.disturbances)
    action, old_log_prob, state, C, c = epoch.epoch_step(agent, envApi.current,
                                                      envApi.date)
    print(action.item())
    assert action.item() != 0, action.item()
