from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np


def print_gridworld_with_policy(agent, figsize=(6, 6)):
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

def print_heatmap(agent, states):
    '''
    This method serves the purpose of printing a heatmap of the state distribution given a trajectory of states drawn by
    an agent acting inside an environment.
    :param agent: the agent that acted in the environment;
    :param states: the states over the trajectory that has been drawn by the agent acting in the environment, these must
        be passed in the form of a vector of probabilities of the length of the states of the underlying MDP.
    '''
    d_t = agent.compute_state_distribution(states)
    print(d_t)
    heatmap_data = np.zeros((agent.env.grid_size, agent.env.grid_size))
    for index in range(agent.env.state_space.n):
        col, row = agent.env.index_to_state(index)
        heatmap_data[row][col] = d_t[col][row]
    sns.heatmap(data=heatmap_data, annot=True)