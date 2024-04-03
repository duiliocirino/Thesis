import pickle
import os
import numpy as np
from multiprocessing import Pool
import gym

from tqdm import tqdm
import pandas as pd

import agents
from utility import plot_graph, compute_intervals

default_run_idx = 1                                                                 # Default values for training variables
default_n_run = 1
default_environment_type = 'GridworldPOMDPEnvGoalless-v0'
default_world_matrix = None
default_confidence = 0.95

default_agent_type = 1
default_policy_type = 1
default_weight_policy = 0
default_history_type = 1
default_importance_belief = 0
default_learnable_weight = 1
default_belief = True
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
default_sample_belief = False

default_n_traj = 16
default_n_episodes = 10000
default_grid_size = 5
default_length_corridor = 0
default_time_horizon = default_grid_size**2 + default_length_corridor
default_goggles = False
default_goggles_pos = (3, 3)
default_goggles_corridor_shift = False
default_multi_room = False
default_steepness = 10
default_uniform = 0
default_prob = 0
default_randomize = 0
default_manhattan_obs = True

default_store_trajs = False

# Define a function to perform the training for a single run
def train_agent(args_list):
    kwargs = args_list[1]
    # Unpack kwargs
    sub_exp_folder = kwargs.get('sub_exp_folder', default_run_idx)                                      # Directories

    run_idx = args_list[0]                                                                              # Training parameters
    environment_type = kwargs.get('environment_type', default_environment_type)
    n_traj = kwargs.get('n_traj', default_n_traj)
    n_episodes = kwargs.get('n_episodes', default_n_episodes)

    time_horizon = kwargs.get('time_horizon', default_time_horizon)                                     # Environment parameters
    grid_size = kwargs.get('grid_size', default_grid_size)
    length_corridor = kwargs.get('length_corridor', default_length_corridor)
    world_matrix = kwargs.get('world_matrix', default_world_matrix)
    goggles = kwargs.get('goggles', default_goggles)
    goggles_pos = kwargs.get('goggles_pos', default_goggles_pos)
    goggles_corridor_shift = kwargs.get('goggles_corridor_shift', default_goggles_corridor_shift)
    multi_room = kwargs.get('multi_room', default_multi_room)
    steepness = kwargs.get('steepness', default_steepness)
    uniform = kwargs.get('uniform', default_uniform)
    prob = kwargs.get('prob', default_prob)
    randomize = kwargs.get('randomize', default_randomize)
    manhattan_obs = kwargs.get('manhattan_obs', default_manhattan_obs)
    
    agent_type = kwargs.get('agent_type', default_agent_type)                                           # Agent parameters
    policy_type = kwargs.get('policy_type', default_policy_type)
    weight_policy = kwargs.get('weight_policy', default_weight_policy)
    history_type = kwargs.get('history_type', default_history_type)
    importance_belief = kwargs.get('importance_belief', default_importance_belief)
    learnable_weight = kwargs.get('learnable_weight', default_learnable_weight)
    belief = kwargs.get('belief', default_belief)
    states_policy = kwargs.get('states_policy', default_states_policy)
    alpha = kwargs.get('alpha', default_alpha)
    true_feedback = kwargs.get('true_feedback', default_true_feedback)
    random_init_policy = kwargs.get('random_init_policy', default_random_init_policy)
    enable_belief_policy = kwargs.get('enable_belief_policy', default_enable_belief_policy)
    objective = kwargs.get('objective', default_objective)
    baseline_type = kwargs.get('baseline_type', default_baseline_type)
    entropy_coefficient = kwargs.get('entropy_coefficient', default_entropy_coefficient)
    reg_coeff = kwargs.get('reg_coeff', default_reg_coeff)
    noise_oracle = kwargs.get('noise_oracle', default_noise_oracle)
    sample_belief = kwargs.get('sample_belief', default_sample_belief)

    store_trajs = kwargs.get('store_trajs', default_store_trajs)

    # Set the numpy seed for the entire run with the run_index
    np.random.seed(run_idx)
    print(world_matrix)
    # Create a new environment for each run
    if world_matrix is None:
        env = gym.make(environment_type, time_horizon=time_horizon, grid_size=grid_size, length_corridor=length_corridor, goggles=goggles, goggles_pos=goggles_pos, 
                       goggles_corridor_shift=goggles_corridor_shift, multi_room=multi_room, steepness=steepness, prob=prob, randomize=randomize, manhattan_obs=manhattan_obs, uniform=uniform)
    else:
        env = gym.make(environment_type, time_horizon=time_horizon, grid_size=grid_size, goggles=goggles, goggles_pos=goggles_pos, world_matrix=world_matrix, 
                       steepness=steepness, prob=prob, randomize=randomize, manhattan_obs=manhattan_obs, uniform=uniform)  # Create a new environment for each run
    # Instantiate the agent requested
    if agent_type == 0:
        agent = agents.REINFORCEAgentE(env, alpha=alpha, belief=belief, states_policy=states_policy, true_feedback=true_feedback, random_init_policy=random_init_policy, sample_belief=sample_belief) 
    elif agent_type == 1:
        agent = agents.REINFORCEAgentEPOMDP(env, alpha=alpha, true_feedback=true_feedback, random_init_policy=random_init_policy, enable_belief_policy=enable_belief_policy,
                                            noise_oracle=noise_oracle, sample_belief=sample_belief)
    elif agent_type == 2:
        agent = agents.REINFORCEDoubleAgentE(env, alpha=alpha, true_feedback=true_feedback, random_init_policy=random_init_policy, enable_belief_policy=enable_belief_policy,
                                             noise_oracle=noise_oracle, policy_type=policy_type, weight_policy=weight_policy, history_type=history_type, 
                                             importance_belief=importance_belief, learnable_weight=learnable_weight, sample_belief=sample_belief)
    #print(f"Agent: {agent_type}, Belief: {belief}")
    # Create the arrays to store the results
    avg_entropies = []
    avg_true_entropies = []
    bounds=[]

    if run_idx == 0:
        with tqdm(total=n_episodes, position=run_idx, ncols=80, desc='Run {}'.format(run_idx)) as pbar:
            for ep in range(n_episodes):
                if agent_type == 2 and learnable_weight > 0:
                    episodes, true_entropies = agent.play(env=env, n_traj=n_traj, seed=run_idx)
                    agent.update_weights(episodes, objective=objective, baseline_type=baseline_type, entropy_coefficient=entropy_coefficient, belief_reg=reg_coeff)
                    ## TODO: if learnable_weight > 1
                episodes, true_entropies = agent.play(env=env, n_traj=n_traj, seed=run_idx)
                # Update policy
                if store_trajs == False:
                    entropies, bound = agent.update_multiple_sampling(episodes, objective=objective, baseline_type=baseline_type, entropy_coefficient=entropy_coefficient, belief_reg=reg_coeff)
                    avg_entropies.append(np.mean(entropies))
                    avg_true_entropies.append(np.mean(true_entropies))
                    bounds.append(bound)
                # Or save trajectories in a file
                else:
                    # Split the directory path into its components
                    parts = sub_exp_folder.split(os.sep)
                    # Remove the last part
                    store_trajs_folder = os.path.join(os.sep.join(parts[:-2]), "Trajs")
                    traj_filename = os.path.join(store_trajs_folder, "trajs_{}.pkl".format(run_idx))
                    with open(traj_filename, 'wb') as f:
                        pickle.dump(episodes, f)
                pbar.update(1)
    else:
        for ep in range(n_episodes):
                if agent_type == 2 and learnable_weight > 0:
                    episodes, true_entropies = agent.play(env=env, n_traj=n_traj, seed=run_idx)
                    agent.update_weights(episodes, objective=objective, baseline_type=baseline_type, entropy_coefficient=entropy_coefficient, belief_reg=reg_coeff)
                    ## TODO: if learnable_weight > 1
                episodes, true_entropies = agent.play(env=env, n_traj=n_traj, seed=run_idx)
                # Update policy
                if store_trajs == False:
                    entropies, bound = agent.update_multiple_sampling(episodes, objective=objective, baseline_type=baseline_type, entropy_coefficient=entropy_coefficient, belief_reg=reg_coeff)
                    avg_entropies.append(np.mean(entropies))
                    avg_true_entropies.append(np.mean(true_entropies))
                    bounds.append(bound)
                # Or save trajectories in a file
                else:
                    # Split the directory path into its components
                    parts = sub_exp_folder.split(os.sep)
                    # Remove the last part
                    store_trajs_folder = os.path.join(os.sep.join(parts[:-2]), "Trajs")
                    traj_filename = os.path.join(store_trajs_folder, "trajs_{}.pkl".format(run_idx))
                    with open(traj_filename, 'wb') as f:
                        pickle.dump(episodes, f)
    stats = agent.print_visuals(env, n_traj=n_traj, seed=run_idx)
    bounds = np.array(bounds)
    if store_trajs == False:
        # Save agent
        agent_filename = os.path.join(sub_exp_folder, "agent_{}.pkl".format(run_idx))
        with open(agent_filename, 'wb') as file:
            pickle.dump(agent, file)

        # Save bounds
        bounds_filename = os.path.join(sub_exp_folder, "bounds_{}.pkl".format(run_idx))
        with open(bounds_filename, 'wb') as file:
            pickle.dump(bounds, file)

        # Save entropies
        entropies_filename = os.path.join(sub_exp_folder, "avg_entropies_{}.pkl".format(run_idx))
        with open(entropies_filename, 'wb') as file:
            pickle.dump(avg_entropies, file)
    return avg_entropies, avg_true_entropies, bounds, stats

# Define a function for the entire training with belief
def train_all_belief(**kwargs):
    plot_folder = kwargs.get('plot_folder')
    sub_exp_name = kwargs.get('sub_exp_name')

    n_run = kwargs.get('n_run', default_n_run)
    n_episodes = kwargs.get('n_episodes', default_n_episodes)
    confidence = kwargs.get('confidence', default_confidence)
    store_trajs = kwargs.get('store_trajs', default_store_trajs)
    
    # Create a list of arguments to pass to train_agent
    args_list = [(r, kwargs) for r in range(n_run)]

    # Use multiprocessing.Pool to parallelize the runs
    with Pool() as pool:
        results = pool.map(train_agent, args_list)

    # Collect the results
    list_entropies = []
    list_true_entropies = []
    list_bounds = []
    for result in results:
        entropies, true_entropies, bounds, _ = result
        list_entropies.append(entropies)
        list_true_entropies.append(true_entropies)
        list_bounds.append(bounds)

    # Transpose the lists to get the desired format
    list_entropies = np.transpose(np.array(list_entropies), (1, 0))
    list_true_entropies = np.transpose(np.array(list_true_entropies), (1, 0))
    list_bounds = np.transpose(np.array(list_bounds), (1, 0))

    if store_trajs == False:
        # Prepare values to print
        plot_args = {}
        entropies_means = np.mean(list_entropies, axis=1)
        entropies_stds = np.std(list_entropies, axis=1)
        plot_args['Learned Entropy'] = [entropies_means, entropies_stds]
        true_entropies_means = np.mean(list_true_entropies, axis=1)
        true_entropies_stds = np.std(list_true_entropies, axis=1)
        plot_args['Learned True Entropy'] = [true_entropies_means, true_entropies_stds]
        # Compute the bound
        plot_graph(n_run, n_episodes, plot_args, confidence, 'Entropy', "Deterministic Entropy POMDP")
        plot_args = {}
        bound_vals = np.mean(list_bounds, axis=1)
        bound_stds = np.std(list_bounds, axis=1)
        plot_args['Lower Bound'] = [bound_vals, bound_vals]
        plot_graph(n_run, n_episodes, plot_args, confidence, 'Entropy', "Bound Plot")
        performance = "Believed Entropy mean: " + str(np.mean(entropies_means[n_episodes-100:])) + "\nTrue Entropy mean: " + str(np.mean(true_entropies_means[n_episodes-100:]))
        print(performance)
    
        lower, upper = compute_intervals(n_episodes, n_run, confidence, [true_entropies_means, true_entropies_stds])
        data = {'Episode': range(0, n_episodes), 'Mean':true_entropies_means, 'Std':true_entropies_stds, 'Upper_Confidence':upper, 'Lower_Confidence':lower}
        df = pd.DataFrame(data)
        df.to_csv(plot_folder + '/' + sub_exp_name + '_true_entropies_data.csv', index=False)

        lower, upper = compute_intervals(n_episodes, n_run, confidence, [entropies_means, entropies_stds])
        data = {'Episode': range(0, n_episodes), 'Mean':entropies_means, 'Std':entropies_stds, 'Upper_Confidence':upper, 'Lower_Confidence':lower}
        df = pd.DataFrame(data)
        df.to_csv(plot_folder + '/' + sub_exp_name + '_believed_entropies_data.csv', index=False)

        lower, upper = compute_intervals(n_episodes, n_run, confidence, [bound_vals, bound_stds])
        data = {'Episode': range(0, n_episodes), 'Mean':bound_vals, 'Std':bound_stds, 'Upper_Confidence':upper, 'Lower_Confidence':lower}
        df = pd.DataFrame(data)
        df.to_csv(plot_folder + '/' + sub_exp_name + '_bound_data.csv', index=False)
