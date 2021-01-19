# Modify here: Outputs from EnergyPlus; Match the variables.cfg file.
obs_name = [
    "Outdoor Temp.", "Outdoor RH", "Wind Speed", "Wind Direction",
    "Diff. Solar Rad.", "Direct Solar Rad.", "Htg SP", "Clg SP",
    "Indoor Temp.", "Indoor Temp. Setpoint", "PPD", "Occupancy Flag",
    "Coil Power", "HVAC Power", "Sys In Temp.", "Sys In Mdot", "OA Temp.",
    "OA Mdot", "MA Temp.", "MA Mdot", "Sys Out Temp.", "Sys Out Mdot"
]

# Modify here: Change based on the specific control problem
state_name = ["Indoor Temp."]
dist_name = [
    "Outdoor Temp.", "Outdoor RH", "Wind Speed", "Wind Direction",
    "Diff. Solar Rad.", "Direct Solar Rad.", "Occupancy Flag"
]
# Caveat: The RL agent controls the difference between Supply Air Temp. and Mixed Air Temp., i.e. the amount of heating from the heating coil. But, the E+ expects Supply Air Temp. Setpoint.
ctrl_name = ["SA Temp Setpoint"]
target_name = ["Indoor Temp. Setpoint"]

# Modify here: Change based on the specific control problem
dist_name = [
    "Outdoor Temp.", "Outdoor RH", "Wind Speed", "Wind Direction",
    "Direct Solar Rad.", "Occupancy Flag"
]
# Caveat: The RL agent controls the difference between Supply Air Temp. and Mixed Air Temp., i.e. the amount of heating from the heating coil. But, the E+ expects Supply Air Temp. Setpoint.
ctrl_name = ["SA Temp Setpoint"]
target_name = ["Indoor Temp. Setpoint"]

drop_keys = [
    "Diff. Solar Rad.", "Clg SP", "Sys In Temp.", "Sys In Mdot", "OA Temp.",
    "HVAC Power", "MA Mdot", "OA Mdot", "Sys Out Mdot"
]