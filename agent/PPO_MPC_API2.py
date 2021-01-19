from flask import Flask, url_for, request
import copy
import os

import torch
import logging
import pandas as pd
from PPO_MPC_EP import dist_name
from my_quick_date_util.my_quick_dt_util import parse_date, utcnow, floor_date, ceil_date, get_tz
from EnvApi import EnvApi



save_agent = os.environ.get('SAVE_AGENT', "")
if save_agent == "True":
    save_agent = True
else:
    save_agent = False


app = Flask(__name__)



agent = initialize()




@app.route('/mpc/', methods=['POST'])
def mpc_api():
    req = request.get_json()

    d_ = copy.deepcopy(req['disturbances'])

    env = EnvApi(req['disturbances'], parse_date(req['date']))

    obs, target, disturbances = env.get_obs()



    



    return "ok"


if __name__ == "__main__":
    print("Main")

    logging.info("Loading Flask API...")
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))