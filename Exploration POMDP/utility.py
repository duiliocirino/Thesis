import os
import pickle
import shutil
import re
from tqdm import tqdm
from queue import PriorityQueue

from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import LogNorm
import matplotlib.patches as patches

import pandas as pd
from tabulate import tabulate
import seaborn as sns

import numpy as np
from scipy.stats import t

## GLOBAL VARIABLES ##
conf_int_keys = ['Total Reward', 'Believed Entropy', 'Learned Entropy']             # Labels of the curves to plot with the intervals of confidence
actions_str = ['Up', 'Down', 'Left', 'Right']                                       # Enum for putting names to the actions, visualization purposes
x_off = [0, 0, -0.4, 0.4]                                                           # Offsets for the arrows given the action
y_off = [-0.4, 0.4, 0, 0]
arrow_type = '->'                                                                   # Set the arrow type for the state action visualization
dot_radius = 0.1                                                                    # Radius for the circle in the grid


# Plotting variables for Curves

default_min_y_plots = 1                                                             # Minimum value to show on the entropy plots

line_styles = ['-', '--', '-.', ':']
colors = ['green', 'orange', 'red', 'darkgrey',
          'blue', 'cyan', 'magenta', 'yellow',
          'sienna', 'tan']

folder_translations = {                                                             # Translation from folder name to Latex objectives
    'TrueFeedback': "PG for MSE",
    'Belief': "PG for MHE",
    'BeliefReg': "PG for Reg-MHE",
    'BeliefSampled': "PG for MP MHE",
    'BeliefRegSampled': "PG for MP Reg-MHE",
    'Observation': "PG for B-MOE",
    'ObservationMDP': "PG for MOE",
    'ObservationMDPReg': "PG for Reg-MOE",
    'TrueFeedbackMarkovState': "PG for MSE"
}
plot_size = (8, 5)                                                                  # figsize for the plot_graph function
plot_colormap = 'seismic'                                                           # Colormap for the curves TODO: see if remains
max_episodes = 15000                                                                # Max number of episodes for the x axis

## PLOTTING ##

def print_gridworld_with_policy(policy_params, env, figsize=(6, 6), title="notitle", ax=None, save_dir=None):
    '''if ax is None:
        '''
    fig, ax = plt.subplots(figsize=figsize)

    if env.goggles:
        n_states = int(env.observation_space.n/2)
    else:
        n_states = env.observation_space.n

    for state in range(n_states):
        col, row = env.index_to_state(state)
        
        ax.add_patch(
            plt.Rectangle((col, env.grid_size - row - 1), 1, 1, facecolor='white', edgecolor='black'))

        for a in range(env.action_space.n):
            prob = np.exp(policy_params[state, a]) / np.sum(np.exp(policy_params[state]))
            # Calculate the center coordinates of the state cell
            center_x = col + 0.5
            center_y = env.grid_size - row - 0.5

            # Calculate the arrow starting coordinates relative to the center
            arrow_start_x = center_x
            arrow_start_y = center_y

            # Calculate the arrow ending coordinates relative to the center based on the action
            if a == 0:  # Up arrow
                arrow_end_x = center_x
                arrow_end_y = center_y + prob * 0.3  # Adjust the scaling factor to control the arrow length
            elif a == 1:  # Down arrow
                arrow_end_x = center_x
                arrow_end_y = center_y - prob * 0.3
            elif a == 2:  # Left arrow
                arrow_end_x = center_x - prob * 0.3
                arrow_end_y = center_y
            elif a == 3:  # Right arrow
                arrow_end_x = center_x + prob * 0.3
                arrow_end_y = center_y

            # Calculate the arrowhead size based on the value
            head_width = prob * 0.2  # Adjust the scaling factor to control the arrowhead width
            head_length = prob * 0.2  # Adjust the scaling factor to control the arrowhead length

            # Draw the arrow
            arrow_thickness = 1  # Adjust the scaling factor to control the arrow thickness
            ax.arrow(arrow_start_x, arrow_start_y, arrow_end_x - arrow_start_x, arrow_end_y - arrow_start_y,
                      head_width=head_width, head_length=head_length, fc='black', ec='black', linewidth=arrow_thickness)

    ax.axis('scaled')
    ax.axis('off')
    ax.set_title(title)

    # Save the plot in PNG format
    if save_dir is not None:
        # Convert title to lowercase and replace spaces with underscores
        save_title = title.lower().replace(' ', '_')
        # Save just the portion _inside_ the axis's boundaries
        fig = plt.gcf()
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(f"{save_dir}/{save_title}_plot.png", bbox_inches=extent, pad_inches=0.1)

def print_heatmap(agent, d_t, title, ax=None, annot=True, cbar=True, cmap='Blues', save_dir=None):
    '''
    This method serves the purpose of creating a heatmap of the state distribution given a trajectory of states drawn by
    an agent acting inside an environment.
    - agent: the agent that acted in the environment;
    - d_t: the state visitation distribution;
    - title: the title for the heatmap;
    - ax: the Axes object to use for plotting (optional);
    - annot: boolean to turn on or off the visualization of numbers on the plot;
    - cbar: boolean to turn on or off the visualization of the colorbar on the right of the plot;
    - cmap: sets the colormap to use for the heatmap.
    '''
    if ax is None:
        ax = plt.gca()
    # Check corridor
    if hasattr(agent.env, 'world_matrix'):
        heatmap_data = np.full(agent.env.world_matrix.shape, np.nan)
    elif hasattr(agent.env, 'length_corridor'):
        heatmap_data = np.full((agent.env.grid_size + agent.env.length_corridor, agent.env.grid_size), np.nan)
    else:
        heatmap_data = np.full((agent.env.grid_size, agent.env.grid_size), np.nan)
    # Prepare data for plotting
    for index in range(d_t.size):
        col, row = agent.env.index_to_state(index)
        # Handle goggles with sum if value is not nan
        if np.isnan(heatmap_data[row][col]):
            heatmap_data[row][col] = d_t[index]
            #print(f"placed {d_t[index]} to state ({row}, {col})")
        else:
            heatmap_data[row][col] += d_t[index]
            #print(f"added {d_t[index]} to state ({row}, {col})")
    # Plot the heatmap
    ax = sns.heatmap(data=heatmap_data, cmap=cmap, vmin=0, vmax=np.max(d_t), annot=annot, cbar=cbar, ax=ax)

    # Set minor locator to place grid lines between ticks
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(MultipleLocator(1))

    # Enable grid lines
    ax.grid(True, which='minor', linestyle=':', linewidth=0.5, color='black')

    # Add walls for multi_room (Matrix gridworld is not affected)
    if hasattr(agent.env, 'multi_room'):
        if agent.env.multi_room == True:
            ax.axhline(agent.env.grid_size/2, xmin=1 / agent.env.grid_size, xmax=(agent.env.grid_size - 1) / agent.env.grid_size, color='black', linestyle='-', linewidth=3)
            ax.axvline(agent.env.grid_size/2, ymin=1 / agent.env.grid_size, ymax=(agent.env.grid_size - 1) / agent.env.grid_size, color='black', linestyle='-', linewidth=3)

    # Save the plot in PNG format
    if save_dir is not None:
        # Create a new figure just for the heatmap
        fig, ax = plt.subplots(figsize=(6,6))
         # Plot the heatmap
        sns.set(font_scale=2)
        sns.heatmap(data=heatmap_data, cmap=cmap, vmin=0, vmax=np.max(d_t), annot=False, cbar=True, ax=ax)
        
        # Remove x-axis and y-axis tick labels
        ax.set_xticks([])
        ax.set_yticks([])

        # Set major locator to place ticks
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(1))

        # Set minor locator to place grid lines between ticks
        ax.xaxis.set_minor_locator(MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(MultipleLocator(0.5))

        # Enable grid lines
        ax.grid(True, which='major', linestyle=':', linewidth=0.5, color='black')

        # Add walls
        if hasattr(agent.env, 'multi_room'):
            if agent.env.multi_room == True:
                ax.axhline(agent.env.grid_size/2, xmin=1 / agent.env.grid_size, xmax=(agent.env.grid_size - 1) / agent.env.grid_size, color='black', linestyle='-', linewidth=3)
                ax.axvline(agent.env.grid_size/2, ymin=1 / agent.env.grid_size, ymax=(agent.env.grid_size - 1) / agent.env.grid_size, color='black', linestyle='-', linewidth=3)
        # Remove white space
        fig.tight_layout()
        # Convert title to lowercase and replace spaces with underscores
        save_title = title.lower().replace(' ', '_')
        fig.savefig(f"{save_dir}/{save_title}_plot.png")


def print_heatmap_with_state_action(agent, d_t, title, true_state_index, action, ax=None, save_dir=None):
    '''
    This method serves the purpose of creating a heatmap of the state distribution given a trajectory of states drawn by
    an agent acting inside an environment.
     - agent: the agent that acted in the environment;
     - d_t: the state visitation distribution.
     - title: the title for the heatmap.
     - ax: the Axes object to use for plotting (optional).
    '''
    # Print the heatmap
    print_heatmap(agent, d_t, title, ax, annot=False)
    
    # Add Arrow
    true_state = agent.env.index_to_state(true_state_index)
    start = (true_state[0] + 0.5,true_state[1] + 0.5)
    end = (start[0] + x_off[action], start[1] + y_off[action])
    arrow = patches.FancyArrowPatch(start, end, arrowstyle=arrow_type, color='red', mutation_scale=15)
    ax.add_patch(arrow)

    # Add a dot (circle) at the beginning of the arrow
    dot = patches.Circle(start, dot_radius, color='red')
    ax.add_patch(dot)

    ax.set_title(title)
    
    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])

    
    # Save the plot in PNG format
    if save_dir is not None:
        # Convert title to lowercase and replace spaces with underscores
        save_title = title.lower().replace(' ', '_')
        # Save just the portion _inside_ the axis's boundaries
        fig = plt.gcf()
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(f"{save_dir}/{save_title}_plot.png", bbox_inches=extent, pad_inches=0.1)


## TODO: fix
def print_environment_from_experiment(exp_name, env_name):
    # Directory path where pickles are stored
    results_directory = os.path.join("Results", exp_name)
    agent_dir = results_directory + '/SubExperiments/Belief/agent_0.pkl'

    # Load agent
    with open(agent_dir, 'rb') as file:
        agent = pickle.load(file)
        if hasattr(agent.env,'room_space'):
            if agent.env.room_space.n == 4:
                d_t = np.zeros(agent.env.n_states)
                for s in range(agent.env.n_states):
                    if s < agent.env.grid_size ** 2 / 2:
                        if s % agent.env.grid_size < agent.env.grid_size / 2:
                            d_t[s] = 0
                        else: 
                            d_t[s] = 0.25
                    else:
                        if s % agent.env.grid_size < agent.env.grid_size / 2:
                            d_t[s] = 0.5
                        else: 
                            d_t[s] = 0.75
            elif agent.env.room_space.n == 2:
                d_t = np.zeros(agent.env.n_states)
                for s in range(agent.env.n_states):
                    if s % agent.env.grid_size < agent.env.grid_size / 2:
                        d_t[s] = 0
                    else: 
                        d_t[s] = 0.25
        else:
            d_t = np.zeros(agent.env.observation_space.n)
        print_heatmap(agent, d_t, env_name, annot=False, cbar=True)


def plot_graph(n_run, n_episodes, plot_args, confidence, objective, title, print_intervals=False, save_dir=None, downsample_factor=100):
    # Initialize the plotting vectors
    fig, ax = plt.subplots(figsize=plot_size)
    # Add a title to the plot
    plt.title(title)
    # Clear the canvas
    ax.clear()
    # Turn on the grid with a faint linestyle
    ax.grid(True, linestyle=':', alpha=0.5)
    line_width = 2.5  # You can adjust the line width
    min_y_value = float('inf')  # Initialize with positive infinity to find the minimum value
    # Legend names for exporting the legend
    legend_names = []
    # Calculate downsample factor
    downsample_factor = int(n_episodes / 100)

    for i, (key, value) in enumerate(plot_args.items()):
        style = line_styles[i % len(line_styles)]  # Cycle through line styles
        color = colors[i % len(colors)]  # Cyvle through colors
        # Get label for the legend
        if key in folder_translations:
            label = folder_translations[key]
        else:
            label = key
        legend_names.append(label)
        # Downsample the data
        x_downsampled = np.arange(0, n_episodes, downsample_factor)
        y_downsampled = value[0][:n_episodes][::downsample_factor]
        ax.plot(x_downsampled, y_downsampled, label=label, linestyle=style, linewidth=line_width, color=color)
        # Plot the confidence intervals of the learned objectives
        if print_intervals:
            under_line, over_line = compute_intervals(n_episodes, n_run, confidence, value)
            under_line_downsampled = under_line[::downsample_factor]
            over_line_downsampled = over_line[::downsample_factor]
            
            # Plot the fill between the downsampled intervals
            ax.fill_between(x_downsampled, under_line_downsampled, over_line_downsampled, alpha=0.2)

        # Update the minimum y-value
        min_y_value = min(min_y_value, np.min(value[0][:n_episodes]))

    # ax.plot(entropies_means, label='Believed Entropy')
    # ax.plot(true_entropies_means, label='True Entropy')
    ax.set_xlabel('Episode', fontsize=22)
    ax.set_ylabel(objective, fontsize=22)

    # Increase the size of ticks on both x-axis and y-axis
    ax.tick_params(axis='both', which='major', labelsize=18)

    # Set a minimum value for the y-axis
    ax.set_ylim(bottom=max(default_min_y_plots, min_y_value))

    # Move legend to the bottom right corner and make it bigger
    legend = ax.legend(loc='lower right', fontsize=18, ncol=2)

    # Set the correct linestyle for each legend item
    for i, line in enumerate(legend.get_lines()):
        line.set_linestyle(line_styles[i % len(line_styles)])  # Set linestyle for legend lines (you can adjust this)

    fig.canvas.draw()

    if save_dir != None:
        # Save in directory
        eps_filename = f"{save_dir}/{title.replace(' ', '_').lower()}.eps"
        plt.savefig(eps_filename, format='eps', bbox_inches='tight', pad_inches=0.1)
        print(f"Graph saved as {eps_filename}")

        png_filename = f"{save_dir}/{title.replace(' ', '_').lower()}.png"
        plt.savefig(png_filename, format='png', bbox_inches='tight', pad_inches=0.1)
        print(f"Graph saved as {png_filename}")

        # Save without legend
        ax.legend_.remove()  # Remove the legend
        fig.canvas.draw()

        eps_filename_without_legend = f"{save_dir}/{title.replace(' ', '_').lower()}_without_legend.eps"
        plt.savefig(eps_filename_without_legend, format='eps', bbox_inches='tight', pad_inches=0.1)
        print(f"Graph saved as {eps_filename_without_legend}")

        png_filename_without_legend = f"{save_dir}/{title.replace(' ', '_').lower()}_without_legend.png"
        plt.savefig(png_filename_without_legend, format='png', bbox_inches='tight', pad_inches=0.1)
        print(f"Graph saved as {png_filename_without_legend}")
    
    # Export legend
    export_horizontal_legend(legend_names, save_dir)

    plt.close()

def plot_performances(performances):
    # Split the values in the dictionary and create a list of lists for tabulate
    table_data = []
    for key, value in performances.items():
        lines = value.split('\n')
        table_data.append([key, *lines])

    # Format the data as a table
    table = tabulate(table_data, headers=['Key', 'Value'], tablefmt='fancy_grid')

    print(table)

def simplify_experiment_label(exp_name, sub_exp_name):
    # Check if "Oracle" followed by a number is present in exp_name
    oracle_match = re.search(r'Oracle(\d+)', exp_name)
    reg_match = re.search(r'Rho(\d+)', exp_name)
    alpha_match = re.search(r'Alpha(\d+)', exp_name)
    trajs_match = re.search(r'Trajs(\d+)', exp_name)
    b_match = re.search(r'Dynamic', exp_name)
    ba_match = re.search(r'Normal', exp_name)
    if oracle_match:
        num = oracle_match.group(1)  # Extract the number found next to Rho
        if num.startswith('0'):
            num = '0.' + num[1:]
        first_word = r"$\bar s^2$=" + num
    elif reg_match:
        num = reg_match.group(1)  # Extract the number found next to Rho
        if num.startswith('0'):
            num = '0.' + num[1:]
        first_word = r"$\beta$=" + num
    elif alpha_match:
        num = alpha_match.group(1)  # Extract the number found next to Rho
        if num.startswith('0'):
            num = '0.' + num[1:]
        first_word = r"$\alpha$=" + num
    elif trajs_match:
        num = trajs_match.group(1)  # Extract the number found next to Rho
        first_word = r"$N$=" + num
    elif ba_match:
        first_word = r"BA"
    elif b_match:
        first_word = r"B"
    else:
        # Extract the first uppercase letter from exp_name
        first_word = exp_name[0]

        # Find the remaining characters after the first uppercase letter
        remaining_chars = exp_name[1:]
        for char in remaining_chars:
            if char.isupper():
                break
            first_word += char

    # Extract uppercase letters from sub_exp_name
    #uppercase_letters = ''.join(char for char in sub_exp_name if char.isupper()) 
    uppercase_letters = folder_translations[sub_exp_name]

    # Create the simplified label
    simplified_label = f"{first_word} {uppercase_letters}" #uncomment to show sub_exp

    return simplified_label

def same_objective_different_folder_comparison(sub_exp_names, exp_names, n_runs, confidence, comparison_name):
    # exp_names = ['Deterministic4ObsMultiRoom6x6Short', 'Deterministic4ObsMultiRoom6x6ShortOracle01', 'Deterministic4ObsMultiRoom6x6ShortOracle05', 'Deterministic4ObsMultiRoom6x6ShortOracle07']
    # sub_exp_names = ['BeliefReg']
    # n_runs = 16
    # confidence = 0.95

    true_entropies = {}
    bounds = {}
    performances = {}
    for exp_name in exp_names:
        results_directory = os.path.join("Results", exp_name)
        for sub_exp_name in sub_exp_names:
            label_acronym = simplify_experiment_label(exp_name, sub_exp_name)
            print(label_acronym)
            
            true_entropy, bound, performance, believed_entropy = retrieve_results(results_directory, [sub_exp_name])
            
            true_entropies[label_acronym] = true_entropy.get(sub_exp_name)
            bounds[label_acronym] = bound.get(sub_exp_name)
            performances[label_acronym] = performance.get(sub_exp_name)
    comparisons_directory = os.path.join("Results/Comparisons", comparison_name)
    os.makedirs(comparisons_directory)
    plot_results(n_runs, confidence, true_entropies, bounds, performances, save_dir=comparisons_directory)

def plot_lower_upper_bound_observations(exp_names):
    '''
    This method is used to print the curves of the true state entropy and observation entropy together with the upper and lower bound
    for the observation objectives.
    Be sure to pass the right experiments, otherwise you would get non sensed results or errors for directory not found.
    '''
    sub_exp_names = ["ObservationMDP", "ObservationMDPReg"]
    for exp_name in exp_names:
        results_directory = os.path.join("Results", exp_name)
        plots_folder = os.path.join(results_directory, "PlotsData")
        for sub_exp_name in sub_exp_names:
            # Create the sub_exp_bound folder 
            sub_exp_bound_folder = os.path.join(plots_folder, sub_exp_name + "_bound_plot")
            os.makedirs(sub_exp_bound_folder, exist_ok=True)
            # Get sub_experiment folder
            sub_exp_folder_name = os.path.join(os.path.join(results_directory, "SubExperiments"), sub_exp_name)
            # Get agents and put the number of runs
            agent_pkls = get_agents_files(sub_exp_folder_name)
            n_run = len(agent_pkls)
            # Get environment of the experiment
            env = retrieve_environment_from_experiment(exp_name, sub_exp_name)
            # Set confidence for intervals
            confidence = 0.95
            # Retrieve results of the experiment and prepare them for plotting
            true_entropy, bound, _, believed_entropy = retrieve_results(results_directory, [sub_exp_name])
            plot_args = {}
            plot_args["Observations"] = believed_entropy[sub_exp_name]
            plot_args["True State"] = true_entropy[sub_exp_name]
            plot_args["Lower Bound"] = compute_lower_bound_observations(believed_entropy[sub_exp_name], bound[sub_exp_name], env)
            #plot_args["Spectral Lower Bound"] = compute_spectral_lower_bound_observations(believed_entropy[sub_exp_name], env)
            #plot_args["Upper Bound"] = compute_upper_bound_observations(believed_entropy[sub_exp_name], env)
            # Plot the results
            plot_results(n_run, confidence, plot_args, {}, {}, save_dir=sub_exp_bound_folder, plot_file_name="entropies_with_lower_bound", entropies_title="Entropy")


def plot_results(n_run, confidence, entropy, bound, performance, print_intervals=True, save_dir=None, num_episodes=max_episodes, plot_file_name="Plot Entropies", entropies_title="MSE"):
    '''
    This method is used to save the plots of entropies and regularization terms in the PlotsData folder of the specified experiment.
    '''
    # Getting the n_episodes with the minimum length
    min_key = min(entropy, key=lambda k: len(entropy[k][0]))
    if num_episodes != max_episodes:
        n_episodes = num_episodes
    else:
        n_episodes = min(len(entropy[min_key][0]), max_episodes)
    
    plot_graph(n_run, n_episodes, entropy, confidence, entropies_title, plot_file_name, print_intervals, save_dir)
    if bound:
        plot_graph(n_run, n_episodes, bound, confidence, "Regularization Term", "Plot Regularization", print_intervals, save_dir)
    
    # plot_performances(performance, )

def retrieve_results(folder_name, sub_exp_names):
    '''
    This method retrieves the data from the csv files in the PlotsData folder of an experiment and puts them in a related dictionary
    for ease of use in the code.
    '''
    true_entropy = {}
    believed_entropy = {}
    bound = {}
    performance = {}

    for sub_exp_name in sub_exp_names:
        # Load true entropies data
        true_entropies_df = pd.read_csv(folder_name + '/PlotsData/' + sub_exp_name + '_true_entropies_data.csv')
        true_entropy[sub_exp_name] = [true_entropies_df['Mean'].tolist(), true_entropies_df['Std'].tolist()]

        # Load believed entropies data
        believed_entropies_df = pd.read_csv(folder_name + '/PlotsData/' + sub_exp_name + '_believed_entropies_data.csv')
        believed_entropy[sub_exp_name] = [believed_entropies_df['Mean'].tolist(), believed_entropies_df['Std'].tolist()]

        # Load bound data
        bound_df = pd.read_csv(folder_name + '/PlotsData/' + sub_exp_name + '_bound_data.csv')
        bound[sub_exp_name] = [bound_df['Mean'].tolist(), bound_df['Std'].tolist()]

        # Calculate mean performance for last 100 episodes
        last_100_true_entropy = np.mean(true_entropy[sub_exp_name][0][-100:])
        last_100_believed_entropy = np.mean(believed_entropy[sub_exp_name][0][-100:])
        performance[sub_exp_name] = f"Believed Entropy mean: {last_100_believed_entropy}\nTrue Entropy mean: {last_100_true_entropy}"

    return true_entropy, bound, performance, believed_entropy

def retrieve_trajectories_from_experiment(exp_name):
    '''
    This method can get the trajectories previously sampled by the apposite function, that are found in the Trajs folder in the form of pickle files.
    '''
    sequences = []

    experiment_folder = os.path.join("Results", exp_name)
    traj_folder = os.path.join(experiment_folder, "Trajs")

    # List all files in the directory
    pickle_files = [f for f in os.listdir(traj_folder) if f.startswith('trajs_') and f.endswith('.pkl')]
    # For each trajectory in each trajs file unwrap into a list of trajectories factored in the correct way
    for pickle_file in tqdm(pickle_files, desc='Loading trajectories'):
        pickle_path = os.path.join(traj_folder, pickle_file)
        with open(pickle_path, 'rb') as file:
            episodes = pickle.load(file)
            for episode in episodes:
                trajectory = episode[0]
                for i in range(len(trajectory)):
                    formatted_trajectory_step = {
                        'belief': None,
                        'action': None,
                        'observation': None,
                        'next_observation': None
                    }
                    belief_state, action, _, _, _, obs, next_obs = trajectory[i]
                    formatted_trajectory_step['belief'] = belief_state
                    formatted_trajectory_step['action'] = action
                    formatted_trajectory_step['observation'] = obs
                    formatted_trajectory_step['next_observation'] = next_obs
                    sequences.append(formatted_trajectory_step)
    print("Finished loading trajectories")
    return sequences

def retrieve_environment_from_experiment(exp_name, sub_exp_name):
    """
    Given an experiment name and a sub_experiment, this method gets and returns the environment of the experiment.
    """
    env = None
    experiment_folder = os.path.join("Results", exp_name)
    sub_exp_folder = experiment_folder + '/SubExperiments/' + sub_exp_name
    # Combine the directory path with the file name
    file_path = os.path.join(sub_exp_folder, "agent_0.pkl")
    if os.path.exists(file_path):
    # Load the pickle file
        with open(file_path, 'rb') as f:
            agent = pickle.load(f)
            env = agent.env
    else:
        raise(f"The file {file_path} does not exist.")
    return env

def retrieve_agent_from_experiment(exp_name, sub_exp_name):
    env = None
    experiment_folder = os.path.join("Results", exp_name)
    sub_exp_folder = experiment_folder + '/SubExperiments/' + sub_exp_name
    # Combine the directory path with the file name
    file_path = os.path.join(sub_exp_folder, "agent_0.pkl")
    if os.path.exists(file_path):
    # Load the pickle file
        with open(file_path, 'rb') as f:
            agent = pickle.load(f)
            return agent
        
def print_avg_entropy_observation_matrix_from_experiment(exp_name):
    # Get the environment
    experiment_folder = os.path.join("Results", exp_name)
    sub_exps_folder = os.path.join(experiment_folder, "SubExperiments")
    sub_exp_name = get_first_folder_name(sub_exps_folder)
    env = retrieve_environment_from_experiment(exp_name, sub_exp_name)
    
    # Init Array
    entropies = []
    # Compute and print the average entropy
    for row in env.observation_matrix:
        entropies.append(compute_entropy(row))
    print(f"Average entropy: {np.mean(entropies)}")

def print_observation_matrix_from_experiment(exp_name):
    '''
    This method saves the observation matrix for the given experiment.
    '''
    experiment_folder = os.path.join("Results", exp_name)
    plot_folder = os.path.join(experiment_folder, "PlotsData")
    sub_exps_folder = os.path.join(experiment_folder, "SubExperiments")
    sub_exp_name = get_first_folder_name(sub_exps_folder)
    env = retrieve_environment_from_experiment(exp_name, sub_exp_name)
    plt.figure(figsize=(20, 20))
    cmap = sns.color_palette("rocket", as_cmap=True)
    cmap.set_bad((0,0,0))
    # h_map = sns.heatmap(env.observation_matrix, norm=LogNorm(), cmap=cmap) # enable logarithmic scaling
    h_map = sns.heatmap(env.observation_matrix, cmap=cmap)
    # Save the heatmap to a file in the plot folder
    plot_file_path = os.path.join(plot_folder, "observation_matrix_heatmap.png")
    plt.savefig(plot_file_path)

    # Get agent
    agent = retrieve_agent_from_experiment(exp_name, sub_exp_name)

    #Print sigle states (hard coded currently)
    plot_file_path = os.path.join(plot_folder, "observation_matrix_heatmap.png")
    print_heatmap(agent, env.observation_matrix[0], f"State {env.index_to_state(0)} Emission Function", save_dir=plot_folder)

    print_heatmap(agent, env.observation_matrix[15], f"State {env.index_to_state(15)} Emission Function", save_dir=plot_folder)

    # Close the plot to free up memory
    plt.close()


def get_agents_files(sub_exp_folder):
    """
    This method gets and returns all the agents from the given sub_experiment folder.
    """
    # List all files in the directory
    pickle_files = [f for f in os.listdir(sub_exp_folder) if f.startswith('agent_') and f.endswith('.pkl')]

    # Sort the files to ensure they are loaded in order
    pickle_files.sort()
    return pickle_files

def print_trajectory_from_agent(exp_names, sub_exp_names, agent_num=0, n_traj=1):
    for exp_name in exp_names:
        # Directory of the experiment of which we want to print and save policies and heatmaps
        results_directory = os.path.join("Results", exp_name)
        for sub_exp_name in sub_exp_names:
            print("Saving Trajectory for " + exp_name + '_' + sub_exp_name)
            sub_exp_folder = results_directory + '/SubExperiments/' + sub_exp_name
            # Instantiate directory to save the visuals
            trajectory_folder = os.path.join(sub_exp_folder, "Trajectory")
            os.makedirs(trajectory_folder, exist_ok=True)
            
            pickle_files = get_agents_files(sub_exp_folder)

            # Iterate through pickle files and load objects            
            pickle_path = os.path.join(sub_exp_folder, pickle_files[agent_num])
            with open(pickle_path, 'rb') as file:
                agent = pickle.load(file)
                print_traj(agent, env=agent.env, save_dir=trajectory_folder)

def print_traj(agent, env, seed=None, save_dir=None):
    # Sample trajectory
    episodes, _ = agent.play(env, n_traj=1)
    # Get trajectory: episode = (ohe_obs/belief, action, probs, reward, true_state, obs, next_obs)
    episode = episodes[0][0]

    for i in range(env.time_horizon):
        # Create subplots with shared x-axis
        fig, axs = plt.subplots(2, 1, figsize=(5, 6), gridspec_kw={'width_ratios': [1], 'height_ratios': [1, 0.1]})
        plt.subplots_adjust(wspace=0.5, hspace=0.5)

        # Get the values to plot of the current timestep
        belief = episode[i][0]
        action = episode[i][1]
        probs = episode[i][2]
        true_state = episode[i][4]['true_state']

        # Fix goggles view
        if env.goggles and true_state > env.n_states:
            true_state -= env.n_states

        # Plot heatmap of beliefs
        print_heatmap_with_state_action(agent, belief, f'Belief Distribution Over States (Step {i})', true_state, action, ax=axs[0])

        # Plot heatmap of probabilities
        sns.heatmap(np.array(probs).reshape(1, -1), cmap='Blues', vmin=0, vmax=np.max(probs), annot=True, xticklabels=actions_str, ax=axs[1])
        axs[1].set_title('Probabilities Over Actions')
        axs[1].set_yticks([])

        if save_dir is not None:
            png_filename = f"{save_dir}/{i}.png"
            plt.savefig(png_filename, format='png', bbox_inches='tight', pad_inches=0.1)
            print(f"Graph saved as {png_filename}")

        # Show the plot
        plt.show()

def print_visuals(exp_names, sub_exp_names):
    for exp_name in exp_names:
        # Directory of the experiment of which we want to print and save policies and heatmaps
        results_directory = os.path.join("Results", exp_name)
        for sub_exp_name in sub_exp_names:
            print("Saving Visuals for " + exp_name + '_' + sub_exp_name)
            # Directory path where pickles are stored
            sub_exp_folder = results_directory + '/SubExperiments/' + sub_exp_name
            # Instantiate directory to save the visuals
            visuals_folder = os.path.join(sub_exp_folder, "Visuals")
            os.makedirs(visuals_folder)
            # List all files in the directory
            pickle_files = [f for f in os.listdir(sub_exp_folder) if f.startswith('agent_') and f.endswith('.pkl')]

            # Sort the files to ensure they are loaded in order
            pickle_files.sort()
            # Iterator
            ag_num = 0 
            # Iterate through pickle files and load objects
            for pickle_file in pickle_files:
                pickle_path = os.path.join(sub_exp_folder, pickle_file)
                final_folder = os.path.join(visuals_folder, str(ag_num))
                os.makedirs(final_folder)
                with open(pickle_path, 'rb') as file:
                    agent = pickle.load(file)
                    agent.print_visuals(env=agent.env, n_traj=12, save_dir=final_folder)
                ag_num = ag_num + 1

## UTILITY METHODS ##


def export_horizontal_legend(names, plot_folder_path):
    if plot_folder_path != None:
        fig, ax = plt.subplots(figsize=(10, 2))  # Adjust the figure size as needed

        # Create a dummy plot with empty data to generate the legend
        for i, name in enumerate(names):
            color = colors[i % len(colors)]  # Cyvle through colors
            ax.plot([], label=name, color=color)

        # Create the legend with a horizontal layout
        legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=len(names))

        # Set the correct linestyle for each legend item
        for i, line in enumerate(legend.get_lines()):
            line.set_linestyle(line_styles[i % len(line_styles)])  # Set linestyle for legend lines (you can adjust this)

        # Remove the axes for a clean image
        ax.axis('off')

        # Adjust layout to minimize white space
        fig.tight_layout()

        # Put the correct path to the file
        filename = os.path.join(plot_folder_path, "legend.png")

        # Save the figure to a PNG file
        plt.savefig(filename, bbox_inches='tight')

def get_first_folder_name(directory):
    # Get a list of all entries in the directory
    entries = os.listdir(directory)
    
    # Iterate through the entries and find the first folder
    for entry in entries:
        # Check if the entry is a directory
        if os.path.isdir(os.path.join(directory, entry)):
            # Return the name of the first folder found
            return entry
    
    # If no folders were found, return None
    return None
                
def get_distances_matrix(env):
    '''
    Given a MatrixGridworld environment, this function returns the distance in steps between the points in the grid. 
    The distance between two points is -1 if unreachable.
    '''
    # Get number of states in the environment
    n_states = env.n_states
    # Initialize the distances matrix
    distances_matrix = np.zeros((n_states, n_states)) - 2
    # Compute the matrix
    for s in range(n_states):
        for s_i in range(n_states):
            # Check for symmetry
            if distances_matrix[s_i][s] != -2:
                distances_matrix[s][s_i] = distances_matrix[s_i][s]
            else:
                distances_matrix[s][s_i] = dijkstra(env.world_matrix, env.index_to_state(s), env.index_to_state(s_i))

    return distances_matrix

## MATH ##

def mode(ar_sorted):
    ar_sorted.sort()
    idx = np.flatnonzero(ar_sorted[1:] != ar_sorted[:-1])
    count = np.empty(idx.size+1,dtype=int)
    count[1:-1] = idx[1:] - idx[:-1]
    try:
        count[0] = idx[0] + 1
    except:
        return ar_sorted[0], len(ar_sorted)
    count[-1] = len(ar_sorted) - idx[-1] - 1
    argmax_idx = count.argmax()

    if argmax_idx==len(idx):
        modeval = ar_sorted[-1]
    else:
        modeval = ar_sorted[idx[argmax_idx]]
    modecount = count[argmax_idx]
    return modeval-1, modecount

def weighted_mean(arr):
    n_rows, n_cols = arr.shape
    weights = np.arange(n_rows) + 1  # Linearly increasing weights
    weighted_sum = np.sum(arr * weights, axis=0)
    total_weight = np.sum(weights)
    return weighted_sum / total_weight

def compute_entropy(prob_array):
        '''
        This method computes the entropy of a probability distribution.
        Args:
         - prob_array: the probability distribution array.
        '''
        # Create the condition to apply the log
        condition = prob_array > 0
        # Compute log of prob_array
        log_prob_array = -np.log(prob_array, where=condition)
        # Compute the entropy
        entropy = np.sum(np.multiply(prob_array, log_prob_array))
        return entropy
    
def compute_intervals(n_episodes, n_run, confidence, values):
    # Initialize the arrays to plot
    under_line = []
    over_line = []

    for i in range(n_episodes):
        freedom_deg = n_run - 1
        t_crit = np.abs(t.ppf((1 - confidence) / 2, freedom_deg))
        under_line.append(values[0][i] - values[1][i] * t_crit / np.sqrt(n_run))
        over_line.append(values[0][i] + values[1][i] * t_crit / np.sqrt(n_run))

    return under_line, over_line

    
def compute_lower_bound_observations(observations_entropies, reg_term, env):
    """
    Lower bound computation.
    """
    # Convert to numpy arrays
    observations_entropies = np.array(observations_entropies)
    reg_term = np.array(reg_term)
    # Calculate singular values
    _, S, _ = np.linalg.svd(env.observation_matrix)
    # Extract the max singular value
    sigma_max = np.max(S)
    print(f"Sigma max: {sigma_max}")
    # Calculate lower bound
    lower_bound = observations_entropies - reg_term + np.log(sigma_max)
    return lower_bound

def compute_spectral_lower_bound_observations(observations_entropies, env):
    """
    Spectral lower bound computation.
    """
    # Convert to numpy arrays
    observations_entropies = np.array(observations_entropies)
    # Calculate singular values
    _, S, _ = np.linalg.svd(env.observation_matrix)
    # Extract the max singular value
    sigma_max = np.max(S)
    # Calculate lower bound
    lower_bound = observations_entropies + np.log(sigma_max)
    return lower_bound

def compute_upper_bound_observations(observations_entropies, env):
    """
    Spectral upper bound computation.
    """
    inverse_observation_matrix = matrix_element_invert(env.observation_matrix)
    _, S, _ = np.linalg.svd(inverse_observation_matrix)
    sigma_max = np.max(S)
    upper_bound = observations_entropies + np.log(sigma_max)
    return upper_bound


def matrix_element_invert(observation_matrix):
    """
    Given a numpy matrix, this method returns the element-wise inverted matrix.
    O[x,y] = 1 / O[x,y]  
    """
    large_number = 1e6
    inverted_O = np.where(observation_matrix != 0, 1 / observation_matrix, large_number)

    # Find maximum value among the correctly inverted values
    max_value = np.max(inverted_O[np.isfinite(inverted_O)]) * 2

    # Bound the inverted values based on the maximum value
    bounded_inverted_O = np.where(inverted_O > max_value, max_value, inverted_O)

    return bounded_inverted_O

def dijkstra(grid, start, target):
    '''
    Implementation of the Dijkstra algorithm.
    '''
    # Adapt from x,y to y,x
    start = (start[1], start[0])
    target = (target[1], target[0])
    # Get rows and cols
    rows, cols = len(grid), len(grid[0])
    #print(f"Cols: {cols} Rows: {rows}")
    distances = [[float('inf')] * cols for _ in range(rows)]
    distances[start[0]][start[1]] = 0
    pq = PriorityQueue()
    pq.put((0, start))

    while not pq.empty():
        dist, curr = pq.get()
        if curr == target:
            # Print for debugging
            #print(f"Start: {start} | Target: {target} -> Distance: {distances[curr[0]][curr[1]]}")
            return distances[curr[0]][curr[1]]
        
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_r, new_c = curr[0] + dr, curr[1] + dc
            if 0 <= new_r < rows and 0 <= new_c < cols and grid[new_r][new_c] != 0: # Important with the weighted version to be != 0 and not == 1
                new_dist = dist + 1
                if new_dist < distances[new_r][new_c]:
                    distances[new_r][new_c] = new_dist
                    pq.put((new_dist, (new_r, new_c)))

    # Target not reachable
    return -1


## TRAINING FUNCTIONS ##

def generate_log_content(variables):
    log_content = ""

    for variable, value in variables.items():
        log_content += f"{variable} = {value}\n"

    return log_content

def create_sub_experiment_folder(sub_exp_folder, sub_exp_name, log_content):
    sub_exp_folder_name = os.path.join(sub_exp_folder, sub_exp_name)
    
    # Remove the existing folder if it exists and make it
    if os.path.exists(sub_exp_folder_name):
        shutil.rmtree(sub_exp_folder_name)
    os.makedirs(sub_exp_folder_name)
    
    # Save the log info to a file in the experiment folder
    log_filename = os.path.join(sub_exp_folder_name, "log_" + sub_exp_name + ".txt")
    with open(log_filename, 'w') as file:
        file.write(log_content)

    print(f"Log has been saved to {log_filename}")
    return sub_exp_folder_name

## MAZES ##

def build_biroom(grid_size, length_corridor):
    matrix = np.ones((grid_size, grid_size * 2 + length_corridor))
    matrix[:, grid_size : grid_size + length_corridor] = 0
    matrix[grid_size // 2, grid_size : grid_size + length_corridor] = 1
    return matrix

def build_maze(grid_size):
    matrix = np.ones((grid_size, grid_size))
    return matrix