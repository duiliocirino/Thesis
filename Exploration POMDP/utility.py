from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import t


def print_gridworld_with_policy(agent, figsize=(6, 6), title="notitle"):
    plt.figure(figsize=figsize)

    for state in range(agent.env.state_space.n):
        col, row = agent.env.index_to_state(state)

        plt.gca().add_patch(
            plt.Rectangle((col, agent.env.grid_size - row - 1), 1, 1, facecolor='white', edgecolor='black'))

        for a in range(agent.env.action_space.n):
            prob = np.exp(agent.policy_params[state, a]) / np.sum(np.exp(agent.policy_params[state]))
            # Calculate the center coordinates of the state cell
            center_x = col + 0.5
            center_y = agent.env.grid_size - row - 0.5

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
            plt.arrow(arrow_start_x, arrow_start_y, arrow_end_x - arrow_start_x, arrow_end_y - arrow_start_y,
                      head_width=head_width, head_length=head_length, fc='black', ec='black', linewidth=arrow_thickness)

    plt.axis('scaled')
    plt.axis('off')
    plt.show()

def print_heatmap(agent, d_t, title):
    '''
    This method serves the purpose of printing a heatmap of the state distribution given a trajectory of states drawn by
    an agent acting inside an environment.
    :param agent: the agent that acted in the environment;
    :param d_t: the state visitation distribution.
    '''
    # Create the subplot
    ax = plt.gca()
    # Check corridor
    if hasattr(agent.env, 'length_corridor'):
        heatmap_data = np.full((agent.env.grid_size + agent.env.length_corridor, agent.env.grid_size), np.nan)
    else:
        heatmap_data = np.full((agent.env.grid_size, agent.env.grid_size), np.nan)
    # Prepare data for plotting
    for index in range(agent.env.state_space.n):
        col, row = agent.env.index_to_state(index)
        heatmap_data[row][col] = d_t[index]
    # Plot the heatmap
    sns.heatmap(data=heatmap_data, annot=True, ax=ax)
    ax.set_title(title)
    plt.show()

def plot_graph(n_run, n_episodes, list_entropies, list_true_entropies, confidence):
    # Initialize the plotting vectors
    fig, ax = plt.subplots(figsize=(30, 10))
    plt.title("Big number of episodes normal learning rate, normal gaussian")
    entropies_means = []
    true_entropies_means = []
    under_line = []
    over_line = []

    for i in range(n_episodes):
        entropies_mean = np.mean(list_entropies[i])
        true_entropies_mean = np.mean(list_true_entropies[i])
        entropies_std = np.std(list_entropies[i])
        freedom_deg = n_run - 1
        t_crit = np.abs(t.ppf((1 - confidence) / 2, freedom_deg))
        entropies_means.append(entropies_mean)
        true_entropies_means.append(true_entropies_mean)
        under_line.append(entropies_mean - entropies_std * t_crit / np.sqrt(n_run))
        over_line.append(entropies_mean + entropies_std * t_crit / np.sqrt(n_run))

    ax.clear()
    ax.plot(entropies_means, label='Believed Entropy')
    ax.plot(true_entropies_means, label='True Entropy')
    ax.fill_between(np.arange(n_episodes), under_line, over_line, color='b', alpha=0.1)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Entropy')
    ax.legend()
    fig.canvas.draw()