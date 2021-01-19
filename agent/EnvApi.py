import copy
from Env import Env
from my_quick_dt_util import parse_date
import pandas as pd

from variables import target_name, obs_name, dist_name


class EnvApi(Env):

    def __init__(self, current, disturbances, date):

        idx = []
        target = []
        obs = []

        for d in disturbances:
            _d = copy.deepcopy(d)
            idx.append(parse_date(_d['dt']))
            target.append(_d[target_name[0]])

            o = copy.deepcopy(d)
            obs.append(o)
            for k in list(o.keys()):
                if k not in obs_name:
                    del o[k]

            for k in list(d.keys()):
                if k not in dist_name:
                    del d[k]

        self.disturbances = pd.DataFrame(disturbances, index=idx)
        self.obs = pd.DataFrame(obs, index=idx)
        self.target = pd.DataFrame(target, index=idx)
        self.date = date
        self.current = current


    def get_obs(self):

        return self.obs, self.target, self.disturbances

    def get_start_date(self):
        #TODO : env get start date

        return NotImplementedError

    def reset(self):

        return 900, self.current, None

    def step(self):
        #TODO : step

        #TODO timeStep / avancer au prochain 15 minutes / nbr secondes depuis depuis de la journée
        timeStep = None

        #return timeStep
        return NotADirectoryError