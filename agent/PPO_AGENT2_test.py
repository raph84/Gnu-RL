import os
from datetime import timedelta
from my_quick_dt_util import utcnow, parse_date
from PPO_AGENT2 import initialize, find_model_file
from tst_payload import payload
from Epoch import Epoch
from EnvApi import EnvApi


def test_initialize_standalone():
    os.environ['STANDALONE'] = "True"
    os.environ['AGENT_BUCKET'] = "gnu-rl-agent"


    epoch = Epoch(1, 1, None, 0.98, None)
    envApi = EnvApi(payload()['current'], payload()['disturbances'],
                    parse_date(payload()['date']))
    agent = initialize(envApi.target, envApi.disturbances)

    check_now = utcnow()
    assert check_now - agent.p.start_time < timedelta(minutes=35)
    assert agent.training is False, agent.training


def test_find_model_file():

    dir_path = find_model_file()
    assert dir_path.endswith('agent'), dir_path