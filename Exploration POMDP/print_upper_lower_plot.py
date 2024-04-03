import os
import utility

exp_names = ["Stochastic01Matrix4RoomsCorridorG01T55GogglesMarkovObs", "Stochastic01Matrix4RoomsCorridorG1T55MarkovObs",
             "Stochastic01Matrix4RoomsCorridorG4T55MarkovObs"]

utility.plot_lower_upper_bound_observations(exp_names)