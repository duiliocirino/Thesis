import os
import pickle
import utility

# Name of directories to print the trajectory for
exp_names = ["DeterministicSingleRoom5x5G01Short",
            "DeterministicMultiRoom6x6G01Short", "Deterministic4ObsMultiRoom6x6Short", 
            "Deterministic2ObsMultiRoom6x6Short", "StochasticSingleRoom5x5G01Short",
            "DeterministicSingleRoom5x5U075Short", "DeterministicMultiRoom6x6U075Short",
            "Deterministic4ObsMultiRoom6x6T50", 'Stochastic2ObsMultiRoom6x6Short']
# Name of directories to print the trajectory for
exp_names = ["DeterministicMultiRoom6x6G01Short"]
# Name of the objectives of which we want to print the trajectory for
sub_exp_names = ['Belief', 'BeliefReg', 'TrueFeedback', 'Observation']

utility.print_visuals(exp_names, sub_exp_names)