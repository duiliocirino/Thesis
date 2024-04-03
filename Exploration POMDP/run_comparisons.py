import os
import csv
import shutil
import pickle
import argparse
from distutils.util import strtobool
import time


import gym
from gym import spaces
import numpy as np
from scipy.stats import t
from multiprocessing import Pool

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm

import agents
import environments
from train_agent import train_all_belief
from utility import plot_graph, plot_performances, create_sub_experiment_folder, retrieve_results, plot_results, generate_log_content, print_visuals, print_trajectory_from_agent

## Default Values
default_sub_exp_set = 0                                                
default_n_run = 16
default_environment_type = 'GridworldPOMDPEnvGoalless-v0'
default_world_matrix = None
default_confidence = 0.95

default_agent_type = 1
default_policy_type = 1
default_weight_policy = 0 # History
default_history_type = 1 # History
default_importance_belief = 0 # History
default_learnable_weight = 1 # History
default_belief = True
default_sample_belief = False
default_states_policy = False
default_alpha = 0.3
default_true_feedback = False
default_random_init_policy = False
default_enable_belief_policy = False
default_objective = 0
default_baseline_type = 1
default_entropy_coefficient = 1
default_reg_coeff = 0
default_noise_oracle = 0

default_n_traj = 10
default_n_episodes = 12000
default_grid_size = 5
default_length_corridor = 0
default_time_horizon = default_grid_size**2 + default_length_corridor
default_goggles = False
default_goggles_pos = None
default_goggles_corridor_shift = False
default_multi_room = False
default_steepness = 1
default_uniform = 0
default_prob = 0
default_randomize = 0
default_manhattan_obs = True

default_store_trajs = False

# Sets to save the different subexpression depending on the type of experiment we want to run, add any if you want to
sub_exp_sets = [
    ['TrueFeedbackMarkovState', 'ObservationMDP', 'ObservationMDPReg', 'Observation', 'Belief', 'BeliefReg'],       # THESIS
    ['TrueFeedback', 'Observation', 'Belief', 'BeliefReg'],         # ICML
    ['TrueFeedbackMarkovState', 'ObservationMDP', 'ObservationMDPReg'],         # RLC
    ['TrueFeedbackMarkovState', 'ObservationMDP', 'ObservationMDPReg', 'Observation', 'Belief', 'BeliefReg', 'BeliefSampled', 'BeliefRegSampled'],       # ALL
    ['BeliefReg', 'ObservationMDPReg'],
    ['Belief', 'BeliefReg', 'Observation'],
    ['Belief'],
    ['BeliefReg'],
    ['TrueFeedback'],
    ['Observation'],
    ['Belief', 'BeliefReg'],
    ['TrueFeedback', 'Belief'],
    ['TrueFeedback', 'Observation'],
    ['ObservationMDPReg'],
    ['BeliefSampled', 'BeliefRegSampled'],
    ['BeliefRegSampled'],
    ['TrueFeedbackMarkovState'],
    ['ObservationMDP']
]

# Set of preset Matrix Gridworlds
worlds = [
    [ # N.0: gridworld with 4 rooms separated by a corridor room
        [1,1,1,0,1,1,1],
        [1,1,1,1,1,1,1],
        [1,1,1,0,1,1,1],
        [0,1,0,0,0,1,0],
        [1,1,1,0,1,1,1],
        [1,1,1,1,1,1,1],
        [1,1,1,0,1,1,1]
    ],
    [ # N.1: variation of gridworld with 4 rooms separated by a corridor room with an additional magic corridor where to place glasses
        [1,1,1,0,1,1,1],
        [1,1,1,1,1,1,1],
        [1,1,1,0,1,1,1],
        [0,1,0,0,0,1,0],
        [1,1,1,0,1,1,1],
        [1,1,1,1,1,1,1],
        [1,1,1,0,1,1,1],
        [1,0,1,0,0,0,0],
        [1,1,1,0,0,0,0]
    ],
    [ # N.2: variation of N.1 gridworld with 4 rooms separated by a corridor room with a room with better observations and corridor too 
        [1,1,1,0,100,100,100],
        [1,1,1,1,100,100,100],
        [1,1,1,0,100,100,100],
        [0,1,0,0,0,100,0],
        [1,1,1,0,10,10,10],
        [1,1,1,10,10,10,10],
        [1,1,1,0,10,10,10],
        [20,0,20,0,0,0,0],
        [20,20,20,0,0,0,0]
    ],
    [ # N.3: two rooms environment with weights
        [1,1,1,1,0,100,100,100,100],
        [1,1,1,1,1,100,100,100,100],
        [1,1,1,1,0,100,100,100,100]
    ],
    [ # N.4: flipped two rooms environment with weights
        [100,100,100,100,0,1,1,1,1],
        [100,100,100,100,1,1,1,1,1],
        [100,100,100,100,0,1,1,1,1]
    ],
    [ # N.5: choice environment
        [1,1,1,1,1,100,100,100],
        [1,1,1,0,0,100,100,100],
        [1,1,1,0,0,100,100,100],
        [1,0,0,0,0,0,0,1],
        [1,1,1,1,1,0,0,1],
        [1,1,1,1,1,0,0,1],
        [1,1,1,1,1,1,1,1]
    ],
    [ # N.6: choice environment v2
        [1,1,1,1,1,100,100,100,100],
        [1,1,1,0,0,100,100,100,100],
        [1,1,1,0,0,100,100,100,0],
        [1,0,0,0,0,0,0,1,0],
        [1,1,1,1,1,0,0,1,0],
        [1,1,1,1,1,0,0,1,0],
        [1,1,1,1,1,1,1,1,0],
        [0,0,0,1,0,0,0,0,0]
    ],
    [ # N.7: choice environment v3
        [1,1,1,1,1,100,100,100,100],
        [1,1,1,0,0,100,100,100,100],
        [1,1,1,0,0,100,100,100,0],
        [1,0,0,0,0,0,0,1,0],
        [1,1,1,1,1,0,0,1,0],
        [1,1,1,1,1,0,0,1,0],
        [1,1,1,1,1,1,1,1,0],
        [0,0,0,1,0,0,0,0,0],
        [0,0,0,1,1,0,0,0,0]
    ]
]

# Entropy MDP
gym.envs.register(
    id='GridworldEnvGoalless-v0',
    entry_point=environments.GridworldEnvGoalless
)
# Entropy POMDP
gym.envs.register(
    id='GridworldPOMDPEnvGoalless-v0',
    entry_point=environments.GridworldPOMDPEnvGoalless
)
# Entropy POMDP BiModal
gym.envs.register(
    id='GridworldPOMDPEnvBiModal-v0',
    entry_point=environments.GridworldPOMDPEnvBiModal
)
# Entropy POMDP Asymmetric
gym.envs.register(
    id='GridworldPOMDPEnvAsymm-v0',
    entry_point=environments.GridworldPOMDPEnvAsymm
)
# Entropy POMDP Multiroom 4 Obs
gym.envs.register(
    id='GridworldPOMDPEnvGoalless4Obs-v0',
    entry_point=environments.GridworldPOMDPEnvGoalless4Obs
)
# Entropy POMDP Multiroom 2 Obs
gym.envs.register(
    id='GridworldPOMDPEnvGoalless2Obs-v0',
    entry_point=environments.GridworldPOMDPEnvGoalless2Obs
)
# Entropy POMDP
gym.envs.register(
    id='MatrixGridworldPOMDPEnvGoalless-v0',
    entry_point=environments.MatrixGridworldPOMDPEnvGoalless
)
# ToyEnv
gym.envs.register(
    id='ToyPOMDPEnv-v0',
    entry_point=environments.ToyPOMDPEnv
)

def train(exp_name, sub_exp_set, n_run, environment_type, world_matrix, confidence,
          agent_type, policy_type, weight_policy, history_type, importance_belief,
          learnable_weight, belief, sample_belief, states_policy, alpha, true_feedback, enable_belief_policy,
          objective, baseline_type, entropy_coefficient, belief_reg_coeff, moe_reg_coeff,
          noise_oracle, n_traj, n_episodes, grid_size, length_corridor, time_horizon,
          goggles, goggles_pos, goggles_corridor_shift, multi_room, steepness,
          uniform, prob, randomize, manhattan_obs, random_init_policy, store_trajs):
    ## Init

    # Instantiate experiment folder
    experiment_folder = os.path.join("Results", exp_name)

    # Create the experiment folder
    os.makedirs(experiment_folder, exist_ok=True)

    # Create plot savings folder
    plot_folder = os.path.join(experiment_folder, "PlotsData")
    os.makedirs(plot_folder, exist_ok=True)

    # Create a subfolder for logs
    sub_exp_folder = os.path.join(experiment_folder, "SubExperiments")
    os.makedirs(sub_exp_folder, exist_ok=True)

    if store_trajs == True:
        # Create store trajs folder
        store_trajs_folder = os.path.join(experiment_folder, "Trajs")
        os.makedirs(store_trajs_folder, exist_ok=True)

    # Set world if Matrix
    if world_matrix is not None:
        world_matrix = worlds[world_matrix]

    # Record the start time
    start_time_exp = time.time()
    
    ## TODO: extend for history
    for sub_exp_name in sub_exp_sets[sub_exp_set]:
        # Record the start time
        start_time_subexp = time.time()
        
        # Handle Regularization
        if sub_exp_name == 'BeliefReg' or sub_exp_name == 'BeliefRegSampled':
            temp_reg_coeff = belief_reg_coeff
        elif sub_exp_name == "ObservationMDPReg":
            temp_reg_coeff = moe_reg_coeff
        else:
            temp_reg_coeff = default_reg_coeff
        # Handle TrueFeedback
        if sub_exp_name == 'TrueFeedback' or sub_exp_name == 'TrueFeedbackMarkovState':
            true_feedback = True
        else:
            true_feedback = False
        # Handle Agent Type
        if sub_exp_name == 'Observation' or sub_exp_name == 'ObservationMDPReg' or sub_exp_name == 'ObservationMDP' or sub_exp_name == 'TrueFeedbackMarkovState':
            temp_agent_type = 0
        else:
            temp_agent_type = 1
        # Handle Belief policy
        if sub_exp_name == 'BeliefReg' or sub_exp_name == 'Observation' or sub_exp_name == 'TrueFeedback' or sub_exp_name == 'Belief' or sub_exp_name == 'BeliefSampled' or sub_exp_name == 'BeliefRegSampled':
            belief = True
        else:
            belief = False
        # Handle Sampled Belief
        if sub_exp_name == 'BeliefSampled' or sub_exp_name == 'BeliefRegSampled':
            sample_belief = True
        else:
            sample_belief = False
        # Handle TrueFeedbackMarkovState agent_type
        if sub_exp_name == 'TrueFeedbackMarkovState':
            states_policy = True
        else:
            states_policy = False

        # Define the variables dictionary
        variables = {
            "sub_exp_name": sub_exp_name,
            "n_run": n_run,
            "environment_type": environment_type,
            "world_matrix": world_matrix,
            "confidence": confidence,
            "agent_type": temp_agent_type,
            "belief": belief,
            "sample_belief": sample_belief,
            "states_policy": states_policy,
            "alpha": alpha,
            "true_feedback": true_feedback,
            "random_init_policy" : random_init_policy,
            "enable_belief_policy": enable_belief_policy,
            "objective": objective,
            "baseline_type": baseline_type,
            "entropy_coefficient": entropy_coefficient,
            "reg_coeff": temp_reg_coeff,
            "noise_oracle": noise_oracle,
            "n_traj": n_traj,
            "n_episodes": n_episodes,
            "grid_size": grid_size,
            "length_corridor": length_corridor,
            "time_horizon": time_horizon,
            "goggles": goggles,
            "goggles_pos": goggles_pos,
            "goggles_corridor_shift": goggles_corridor_shift,
            "multi_room": multi_room,
            "steepness": steepness,
            "uniform": uniform,
            "prob": prob,
            "randomize": randomize,
            "manhattan_obs": manhattan_obs,
            "store_trajs": store_trajs
        }

        # Generate log content
        log_content = generate_log_content(variables)

        # Create sub_exp folder
        sub_exp_folder_name = create_sub_experiment_folder(sub_exp_folder, sub_exp_name, log_content)

        print(f"Agent: {temp_agent_type}, Belief: {belief}, TrueFeedback: {true_feedback}, States Policy: {states_policy}, Regularization Coefficient: {temp_reg_coeff}, Sample Belief: {sample_belief}")

        # Run the training
        train_all_belief(
            sub_exp_folder=sub_exp_folder_name,
            environment_type=environment_type,
            world_matrix=world_matrix,
            n_traj=n_traj,
            n_episodes=n_episodes,
            time_horizon=time_horizon,
            grid_size=grid_size,
            goggles=goggles,
            goggles_pos=goggles_pos,
            goggles_corridor_shift=goggles_corridor_shift,
            multi_room=multi_room,
            steepness=steepness,
            uniform=uniform,
            prob=prob,
            randomize=randomize,
            manhattan_obs=manhattan_obs,
            agent_type=temp_agent_type,
            belief=belief,
            sample_belief=sample_belief,
            states_policy=states_policy,
            alpha=alpha,
            true_feedback=true_feedback,
            random_init_policy=random_init_policy,
            enable_belief_policy=enable_belief_policy,
            objective=objective,
            baseline_type=baseline_type,
            entropy_coefficient=entropy_coefficient,
            reg_coeff=temp_reg_coeff,
            noise_oracle=noise_oracle,
            plot_folder=plot_folder,
            sub_exp_name=sub_exp_name,
            n_run=n_run,
            confidence=confidence,
            store_trajs=store_trajs
        )
        # Record the end time
        end_time_subexp = time.time()
        # Calculate the elapsed time
        elapsed_time = end_time_subexp - start_time_subexp
        if store_trajs == False:
            print_trajectory_from_agent([exp_name], [sub_exp_name])
            print_visuals([exp_name], [sub_exp_name])

        print(f"Elapsed time for {sub_exp_name}: {elapsed_time} seconds")
    # Record the end time
    end_time_exp = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time_exp - start_time_exp

    print(f"Elapsed time for {exp_name}: {elapsed_time} seconds")
    # Save results
    if store_trajs == False:
        entropies, bounds, performances, believed_entropies = retrieve_results(experiment_folder, sub_exp_sets[sub_exp_set])
        plot_results(n_run, confidence, entropies, bounds, performances, True, plot_folder, num_episodes=n_episodes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This script can launch the experiments given the correct arguments")

    # Define command line arguments with default values
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--sub_exp_set', type=int, default=default_sub_exp_set)
    parser.add_argument('--n_run', type=int, default=default_n_run)
    parser.add_argument('--environment_type', type=str, default=default_environment_type)
    parser.add_argument('--world_matrix', type=int, default=default_world_matrix)
    parser.add_argument('--confidence', type=float, default=default_confidence)
    
    parser.add_argument('--agent_type', type=int, default=default_agent_type)
    parser.add_argument('--policy_type', type=int, default=default_policy_type) # History
    parser.add_argument('--weight_policy', type=float, default=default_weight_policy) # History
    parser.add_argument('--history_type', type=int, default=default_history_type) # History
    parser.add_argument('--importance_belief', type=int, default=default_importance_belief) # History
    parser.add_argument('--learnable_weight', type=int, default=default_learnable_weight) # History
    parser.add_argument('--belief', type=lambda x: bool(strtobool(x)), default=default_belief)
    parser.add_argument('--sample_belief', type=lambda x: bool(strtobool(x)), default=default_sample_belief)
    parser.add_argument('--states_policy', type=lambda x: bool(strtobool(x)), default=default_states_policy)
    parser.add_argument('--alpha', type=float, default=default_alpha)
    parser.add_argument('--true_feedback', type=lambda x: bool(strtobool(x)), default=default_true_feedback)
    parser.add_argument('--random_init_policy', type=lambda x: bool(strtobool(x)), default=default_random_init_policy)
    parser.add_argument('--enable_belief_policy', type=lambda x: bool(strtobool(x)), default=default_enable_belief_policy)
    parser.add_argument('--objective', type=int, default=default_objective)
    parser.add_argument('--baseline_type', type=int, default=default_baseline_type)
    parser.add_argument('--entropy_coefficient', type=float, default=default_entropy_coefficient)
    parser.add_argument('--belief_reg_coeff', type=float, default=default_reg_coeff)
    parser.add_argument('--moe_reg_coeff', type=float, default=default_reg_coeff)
    parser.add_argument('--noise_oracle', type=float, default=default_noise_oracle)
    
    parser.add_argument('--n_traj', type=int, default=default_n_traj)
    parser.add_argument('--n_episodes', type=int, default=default_n_episodes)
    parser.add_argument('--grid_size', type=int, default=default_grid_size)
    parser.add_argument('--length_corridor', type=int, default=default_length_corridor)
    parser.add_argument('--time_horizon', type=int, default=default_time_horizon)
    parser.add_argument('--goggles', type=lambda x: bool(strtobool(x)), default=default_goggles)
    parser.add_argument('--goggles_pos', type=int, default=default_goggles_pos)
    parser.add_argument('--goggles_corridor_shift', type=lambda x: bool(strtobool(x)), default=default_goggles_corridor_shift)
    parser.add_argument('--multi_room', type=lambda x: bool(strtobool(x)), default=default_multi_room)
    parser.add_argument('--steepness', type=float, default=default_steepness)
    parser.add_argument('--uniform', type=float, default=default_uniform)
    parser.add_argument('--prob', type=float, default=default_prob)
    parser.add_argument('--randomize', type=int, default=default_randomize)
    parser.add_argument('--manhattan_obs', type=lambda x: bool(strtobool(x)), default=default_manhattan_obs)

    parser.add_argument('--store_trajs', type=lambda x: bool(strtobool(x)), default=default_store_trajs)

    args = parser.parse_args()

    print(args.belief)

    # Call the main function with the parsed arguments
    train(args.exp_name, args.sub_exp_set, args.n_run, args.environment_type, args.world_matrix, args.confidence,
          args.agent_type, args.policy_type, args.weight_policy, args.history_type, args.importance_belief,
          args.learnable_weight, args.belief, args.sample_belief, args.states_policy, args.alpha, args.true_feedback, args.enable_belief_policy,
          args.objective, args.baseline_type, args.entropy_coefficient, args.belief_reg_coeff, args.moe_reg_coeff,
          args.noise_oracle, args.n_traj, args.n_episodes, args.grid_size, args.length_corridor, args.time_horizon,
          args.goggles, args.goggles_pos, args.goggles_corridor_shift, args.multi_room, args.steepness,
          args.uniform, args.prob, args.randomize, args.manhattan_obs, args.random_init_policy, args.store_trajs)