import os
import pickle
import utility

n_runs = 16
confidence = 0.95

# Name of directories to print the trajectory for 
exp_names = ["NormalSingleRoom3x3G5", "DynamicPolicySingleRoom3x3G5T9"]
# Name of the objectives of which we want to print the trajectory for
sub_exp_names = ['TrueFeedback', 'Belief']
# Name of the folder to put the comparisons in
comparison_name = 'comparison_dynamic3x3_icml_fixed'

utility.same_objective_different_folder_comparison(sub_exp_names, exp_names, n_runs, confidence, comparison_name)