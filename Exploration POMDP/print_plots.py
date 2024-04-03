import os
import pickle
import utility

# Name of directories to print the trajectory for 
exp_names = ["Thesis-4ObsT40"]

# Name of the objectives of which we want to print the trajectory for, when sub_exp_set=0
sub_exp_names = ['TrueFeedbackMarkovState', 'Belief', 'BeliefReg', 'ObservationMDP', 'ObservationMDPReg']

# Number of episodes to cut a longer run
num_episodes = 15000

for exp_name in exp_names:
    experiment_folder = os.path.join("Results", exp_name)
    sub_exp_folder = os.path.join(experiment_folder, "SubExperiments")
    plot_folder = os.path.join(experiment_folder, "PlotsData")
    entropies, bounds, performances, _ = utility.retrieve_results(experiment_folder, sub_exp_names)
    utility.plot_results(16, 0.95, entropies, bounds, performances, True, plot_folder, num_episodes=15000)
    utility.print_observation_matrix_from_experiment(exp_name)