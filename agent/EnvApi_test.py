from EnvApi import EnvApi
import pandas as pd
from my_quick_dt_util import parse_date
from tst_payload import payload

def test_init():
    env = EnvApi(payload()['current'],payload()['disturbances'], parse_date(payload()['date']))

    assert isinstance(env.disturbances, pd.DataFrame)
    assert isinstance(env.target, pd.DataFrame)
    assert isinstance(env.obs, pd.DataFrame)

    assert env.date == parse_date(payload()['date']), payload()['date']




