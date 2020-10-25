import gym
import eplus_env

import pandas as pd
import pickle
import numpy as np

from utils import make_dict

# Create Environment. Follow the documentation of 'Gym-Eplus' to set up additional EnergyPlus simulation environment.
env = gym.make('7Zone-sim_TMYx-v0');
#env = gym.make('5Zone-sim_TMY3-v0');

# Modify here: Outputs from EnergyPlus; Match the variables.cfg file.
obs_name = ["Outdoor Temp.", "Outdoor RH", "Wind Speed", "Wind Direction", "Diff. Solar Rad.", "Direct Solar Rad.", "Htg SP", "Clg SP", "Indoor Temp.", "Indoor Temp. Setpoint", "PPD", "Occupancy Flag", "Coil Power", "HVAC Power", "Sys In Temp.", "Sys In Mdot", "OA Temp.", "OA Mdot", "MA Temp.", "MA Mdot", "Sys Out Temp.", "Sys Out Mdot"]
dist_name = ["Outdoor Temp.", "Outdoor RH", "Wind Speed", "Wind Direction", "Direct Solar Rad.", "Indoor Temp. Setpoint", "Occupancy Flag"]

# Reset the env (creat the EnergyPlus subprocess)
timeStep, obs, isTerminal = env.reset();
obs_dict = make_dict(obs_name, obs)
drop_keys = [
    "Diff. Solar Rad.", "Clg SP", "Sys In Temp.", "Sys In Mdot", "OA Temp.",
    "HVAC Power", "MA Mdot","OA Mdot"
]
for k in drop_keys:
    del obs_dict[k]
obs_name_filter = list(obs_dict.keys())

start_time = pd.datetime(year = env.start_year, month = env.start_mon, day = env.start_day)
print(start_time)

timeStamp = [start_time]
observations = [list(obs_dict.values())]
actions = []

for i in range(91*96):
    # Using EnergyPlus default control strategy;
    action = ()
    timeStep, obs, isTerminal = env.step(action)
    obs_dict = make_dict(obs_name, obs)
    for k in drop_keys:
        del obs_dict[k]
    cur_time = start_time + pd.Timedelta(seconds = timeStep)

    print("{}: Sys Out: {:.2f}({:.2f})-> Zone Temp: {:.2f} | MA Temp: {:.2f}".
          format(cur_time, obs_dict["Sys Out Temp."], obs_dict["Sys Out Mdot"],
                 obs_dict["Indoor Temp."], obs_dict["MA Temp."]))

    timeStamp.append(cur_time)
    observations.append(list(obs_dict.values()))
    #actions.append(action)

# Save Observations
obs_df = pd.DataFrame(np.array(observations), index = np.array(timeStamp), columns = obs_name_filter)
dist_df = obs_df[dist_name]
obs_df.to_pickle("results/Sim-TMY2.pkl")
#obs_df.to_pickle("results/Sim-TMY3.pkl")
dist_df.to_pickle("results/Dist-TMY2.pkl")
#dist_df.to_pickle("results/Dist-TMY3.pkl")
print("Saved!")

env.end_env() # Safe termination of the environment after use.
