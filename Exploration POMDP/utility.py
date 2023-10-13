from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import t

keys = ['Total Reward', 'Believed Entropy', 'Learned Entropy']

def print_gridworld_with_policy(policy_params, env, figsize=(6, 6), title="notitle", ax=None):
    if ax is None:
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

def print_heatmap(agent, d_t, title, ax=None):
    '''
    This method serves the purpose of creating a heatmap of the state distribution given a trajectory of states drawn by
    an agent acting inside an environment.
    :param agent: the agent that acted in the environment;
    :param d_t: the state visitation distribution.
    :param title: the title for the heatmap.
    :param ax: the Axes object to use for plotting (optional).
    '''
    if ax is None:
        ax = plt.gca()
    # Check corridor
    if hasattr(agent.env, 'length_corridor'):
        heatmap_data = np.full((agent.env.grid_size + agent.env.length_corridor, agent.env.grid_size), np.nan)
    else:
        heatmap_data = np.full((agent.env.grid_size, agent.env.grid_size), np.nan)
    # Prepare data for plotting
    for index in range(d_t.size):
        col, row = agent.env.index_to_state(index)
        heatmap_data[row][col] = d_t[index]
    # Plot the heatmap
    sns.heatmap(data=heatmap_data, annot=True, ax=ax)
    ax.set_title(title)

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

def plot_graph(n_run, n_episodes, plot_args, confidence, objective, title):
    # Initialize the plotting vectors
    fig, ax = plt.subplots(figsize=(30, 10))
    # Add a title to the plot
    plt.title(title)        
    # Clear the canvas
    ax.clear()
    # Plot all the arrays
    for key, value in plot_args.items():
        ax.plot(value[0], label=key)
    # Plot the confidence intervals of the learned objective
    for key in keys:
        if key in plot_args.keys():
            under_line, over_line = compute_intervals(n_episodes, n_run, confidence, plot_args.get(key))
            ax.fill_between(np.arange(n_episodes), under_line, over_line, color='b', alpha=0.1)
    #ax.plot(entropies_means, label='Believed Entropy')
    #ax.plot(true_entropies_means, label='True Entropy')
    ax.set_xlabel('Episode')
    ax.set_ylabel(objective)
    ax.legend()
    fig.canvas.draw()

## MATH

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