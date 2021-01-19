from Env import Env
import pandas as pd
import eplus_env


class EnvEplus(Env):


    def __init__(self):
        self.env = eplus_env()

        # Read Information on Weather, Occupancy, and Target Setpoint
        self.obs = pd.read_pickle("results/Dist-TMY2.pkl")
        self.target = self.obs[target_name]
        disturbance = self.obs[dist_name]

        # Min-Max Normalization
        self.disturbance = (disturbance - disturbance.min()) / (
            disturbance.max() - disturbance.min())

        def get_start_date(self):
            start_time = pd.datetime(year=self.env.start_year,
                                 month=self.env.start_mon,
                                 day=self.env.start_day)
            return start_time

        return get_start_date(self)

    def read_info(self):
        return self.obs, self.target, self.disturbance

    def reset(self):
        return self.env.reset()

    def step(self):
        return self.env.step()
