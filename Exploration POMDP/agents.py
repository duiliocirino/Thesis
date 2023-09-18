import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.stats import t, entropy
import multiprocessing

from utility import print_gridworld_with_policy, print_heatmap, plot_graph, mode

## REWARD MDP ##

class REINFORCEAgent():
    '''
    This class is the implementation of a REINFORCE agent that tries to maximize the objective function J(θ)=E_(τ ~ p_π)[R(τ)]

    Args:
     - env: a copy of the environment on which the agent is acting.
     - alpha: the value of the learning rate to compute the policy update.
     - gamma: the value of the discount for the reward in order to compute the discounted reward.
    '''
    def __init__(self, env, alpha=0.1, gamma=0.9):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        # Policy of the agent
        self.policy_params = np.zeros((env.observation_space.n, env.action_space.n))
        # Identity matrix to get a one hot encoded representation of the states of the environment
        self.ohe_states = np.identity(env.observation_space.n)

    def get_probability(self, state):
        '''
        This method is used to get the action probabilities from the policy given a state.

        Args:
         - state: an array representing the state from which we take an action. Sums up to 1, being a probability distribution over the states.
        '''
        # Compute the dot product
        params = np.dot(state, self.policy_params)
        # Compute the softmax
        probs = np.exp(params) / np.sum(np.exp(params))
        return probs
    
    def get_action(self, state):
        '''
        This method is used to get the action the agent will given a certain state.

        Args:
         - state: an array representing the state from which we take an action. Sums up to 1, being a probability distribution over the states.
        '''
        # Get probability vector
        probs = self.get_probability(state)
        # Sample the action
        action = random.choices(range(len(probs)), weights=probs)[0]
        #action = np.random.choice(len(probs), p=probs)
        return action, probs

    def compute_returns(self, trajectory):
        '''
        This method is used in order to compute the returns of the agent given a trajectory.

        Args:
         - episode: an array representing the trajectory of the agent during the episode
        '''
        G = 0
        returns = []
        for t in reversed(range(len(trajectory))):
            _, _, _, reward, _ = trajectory[t]
            G = self.gamma * G + reward
            returns.append(G)
        returns = np.array(list(reversed(returns)))
        return returns

    def update_single_sampling(self, episode):
        '''
        This version of the update is the Monte Carlo sampling version of the REINFORCE algorithm.

        Args:
         - episode: the sampled trajectory from which we compute the policy gradient.
        '''
        # Compute returns
        returns = self.compute_returns(episode)
        # Compute the policy gradient
        grad = np.zeros_like(self.policy_params)
        for t in range(len(episode)):
            state, action, probs, _, _ = episode[t]
            dlogp = np.zeros(self.env.action_space.n)
            for i in range(self.env.action_space.n):
                dlogp[i] = 1.0 - probs[i] if i == action else -probs[i]
            grad += np.outer(state, dlogp) * returns[t]
        # Update the policy parameters
        self.policy_params += self.alpha * grad

    def update_multiple_sampling(self, trajectories):
        '''
        This version of the update takes into consideration the approximation of the gradient by sampling multiple trajectories. Instead
        of working with only one trajectory it works with multiple trajectories in order to have a more accurate representation of the expected
        value of the ∇J(θ).

        Args:
         - trajectories: a list of sampled trajectories from which we compute the policy gradient.
        '''
        # Compute the policy gradient
        grad = np.zeros_like(self.policy_params)
        for episode in trajectories:
            returns = self.compute_returns(episode)
            for t in range(len(episode)):
                state, action, probs, _, _ = episode[t]
                dlogp = np.zeros(self.env.action_space.n)
                for i in range(self.env.action_space.n):
                    dlogp[i] = 1.0 - probs[i] if i == action else -probs[i]
                grad += np.outer(state, dlogp) * returns[t]
        grad /= len(trajectories)
        # Update the policy parameters
        self.policy_params += self.alpha * grad

    def play(self, env, n_traj=1):
        '''
        This method samples n trajectories from the environment for the agent.

        Args:
         - env: the instance of the environment in which the agent samples the trajectories.
         - n_traj: the number of trajectories that we have to sample in one episode.
        '''
        if n_traj == 1:
            episode = []
            total_reward = 0
            # Reset environment
            state, _ = self.env.reset()
            done = False
            while not done:
                # Get the one hot encoding
                ohe_state = self.ohe_states[state, :]
                # Sample the action from the state
                action, probs = self.get_action(ohe_state)
                # Compute the step
                next_state, reward, done, _ = env.step(action)
                # Save step
                episode.append((ohe_state, action, probs, reward, next_state))
                # Update state 
                state = next_state
                # Sum step reward
                total_reward += reward
            return episode, total_reward
        else:
            episodes = []
            total_rewards = []
            for _ in range(n_traj):
                episode = []
                total_reward = 0
                # Reset environment
                state, _ = env.reset()
                done = False
                while not done:
                    # Get the one hot encoding
                    ohe_state = self.ohe_states[state, :]
                    # Sample the action from the state
                    action, probs = self.get_action(ohe_state)
                    # Compute the step
                    next_state, reward, done, _ = env.step(action)
                    # Save step
                    episode.append((ohe_state, action, probs, reward, next_state))
                    # Update state 
                    state = next_state
                    # Sum step reward
                    total_reward += reward
                # Save episode
                episodes.append(episode)
                # Save total reward
                total_rewards.append(total_reward)
            return episodes, total_rewards


## ENTROPY MDP ##


class REINFORCEAgentE(REINFORCEAgent):
    '''
    This class is the extension of the previous Reinforce agent. The only change is the objective followed by the agent that this time is going
    to be the state-action visitation entropy J(θ) = E_(τ ~ p_π)[H(d_τ(s,a))] .

    Args:
     - env: a copy of the environment on which the agent is acting.
     - alpha: the value of the learning rate to compute the policy update.
     - gamma: the value of the discount for the reward in order to compute the discounted reward.
     - policy: an int option representing how to initialize the agent. If 0 initialize with the dummy policy; if 1 initialize with the optimal policy;
               if nothing initialize with the uniform policy.
    '''
    def __init__(self, env, alpha=0.1, gamma=0.9, policy=-1):
        super().__init__(env=env, alpha=alpha, gamma=gamma)
        # Optional initialization to have a non-uniform initial policy
        self.init_policy(policy)

    def init_policy(self, policy):
        if policy == 0:
            self.policy_params = self.create_dummy_policy()
        elif policy == 1:
            self.policy_params = self.create_optimal_policy()
        else:
            self.policy_params = np.ones((self.env.observation_space.n, self.env.action_space.n))

    def create_dummy_policy(self):
        params = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        for i in range(self.env.observation_space.n):
                params[i, 0] = 1
        return params

    def create_optimal_policy(self):
        params = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        k = 1
        actions = [3,3,3,2]
        for i in range(self.env.observation_space.n):
                if i in range(self.env.grid_size**2, self.env.grid_size**2 + self.env.length_corridor):
                    params[i, 1] = 500
                elif i % self.env.grid_size == self.env.grid_size - 1 and (k == 1):
                    params[i, 1] = 500
                    k += 1
                elif i % self.env.grid_size == self.env.grid_size - 1 and (k == 3):
                    params[i, actions[k]] = 500
                    k = 0
                elif i % self.env.grid_size == 0 and k == 2:
                    params[i, 1] = 500
                    k += 1
                elif i % self.env.grid_size == 0 and k == 0:
                    params[i, actions[k]] = 500
                    k += 1
                else:
                    params[i, actions[k]] = 500
        return params

    def compute_entropy(self, d_t):
        '''
        This method computes the entropy of a probability distribution.

        Args:
         - d_t: the probability distribution array.
        '''
        # Create the condition to apply the log
        condition = d_t > 0
        # Compute log of d_t
        log_d_t = -np.log(d_t, where=condition)
        # Compute the entropy
        entropy = np.sum(np.multiply(d_t, log_d_t))
        return entropy

    def update_single_sampling(self, trajectory, d_t):
        '''
        This version of the update is the Monte Carlo sampling version of the REINFORCE algorithm with an entropy objective.

        Args:
         - trajectory: the sampled trajectory from which we compute the policy gradient.
         - d_t: the state visitation over the trajectory.
        '''
        # Compute entropy of d_t
        entropy = self.compute_entropy(d_t)
        # Update policy parameters using the gradient of the entropy of the t-step state distribution
        grad = np.zeros_like(self.policy_params)
        for t in range(len(trajectory)):
            state, action, probs, _, _ = trajectory[t]
            # Compute the policy gradient
            dlogp = np.zeros(self.env.action_space.n)
            for i in range(self.env.action_space.n):
                dlogp[i] = 1.0 - probs[i] if i == action else -probs[i]
            grad += np.outer(state, dlogp) * entropy
        # Update the policy parameters
        self.policy_params += self.alpha * grad
        return entropy

    def update_multiple_sampling(self, trajectories):
        '''
        This version of the update takes into consideration the approximation of the gradient by sampling multiple trajectories. Instead
        of working with only one trajectory it works with multiple trajectories in order to have a more accurate representation of the expected
        value of the ∇J(θ).

        Args:
         - trajectories: a list of sampled trajectories from which we compute the policy gradient.
        '''
        entropies = []
        # Update policy parameters using the approximated gradient of the entropy objective function
        grad = np.zeros_like(self.policy_params)
        for trajectory, d_t in trajectories:
            # Initialize the gradient of the k-th sampled trajectory
            grad_k = np.zeros_like(self.policy_params)
            # Compute entropy
            entropy = self.compute_entropy(d_t)
            # Compute the gradient
            for t in range(len(trajectory)):
                state, action, probs, _, _ = trajectory[t]
                # Compute the policy gradient
                dlogp = np.zeros(self.env.action_space.n)
                for i in range(self.env.action_space.n):
                    dlogp[i] = 1.0 - probs[i] if i == action else -probs[i]
                grad_k += np.outer(state, dlogp) * entropy
            # Sum the k-th gradient to the final gradient
            grad += grad_k
            # Save in entropies
            entropies.append(entropy)
        # Divide the gradient by the number of trajectories sampled
        grad /= len(trajectories)
        # Update the policy parameters
        self.policy_params += self.alpha * grad
        return entropies

    def play(self, env, n_traj=1):
        '''
        This method samples n trajectories from the environment for the agent.

        Args:
         - env: the instance of the environment in which the agent samples the trajectories.
         - n_traj: the number of trajectories that we have to sample in one episode.
        '''
        if n_traj == 1:
            # Initialize episode array
            episode = []
            # Initialize the state visitation array
            d_t = np.zeros(env.observation_space.n)
            # Reset the environment
            state, _ = self.env.reset()
            done = False
            while not done:
                # Update state visitation
                d_t[state] += 1
                # Get the one hot encoding
                ohe_state = self.ohe_states[state, :]
                # Sample the action from the state
                action, probs = self.get_action(ohe_state)
                # Compute the step
                next_state, reward, done, _ = env.step(action)
                episode.append((ohe_state, action, probs, reward, next_state))
                state = next_state
            d_t /= len(episode)
            return episode, d_t
        else:
            # Initialize episodes array
            episodes = []
            for _ in range(n_traj):
                # Initialize episode array
                episode = []
                # Initialize the state visitation array
                d_t = np.zeros(env.observation_space.n)
                # Reset the environment
                state, _ = self.env.reset()
                done = False
                while not done:
                    # Update state visitation
                    d_t[state] += 1
                    # Get the one hot encoding
                    ohe_state = self.ohe_states[state, :]
                    # Sample the action from the state
                    action, probs = self.get_action(ohe_state)
                    # Compute the step
                    next_state, reward, done, _ = env.step(action)
                    episode.append((ohe_state, action, probs, reward, next_state))
                    # Update state
                    state = next_state
                d_t /= len(episode)
                episodes.append((episode, d_t))
            return episodes
        
    def print_visuals(self, env, n_traj):
        # Visualization of policy and expected state visitation
        d_t = np.zeros(env.observation_space.n)
        for _ in range(n_traj):
            # Reset the environment
            state, _ = env.reset()
            done = False
            while not done:
                # Update state visitation
                d_t[state] += 1
                # Get the one hot encoding
                ohe_state = self.ohe_states[state, :]
                # Sample the action from the state
                action, _ = self.get_action(ohe_state)
                # Compute the step
                next_state, _, done, _ = env.step(action)
                # Update state
                state = next_state
        # Normalize the state visitation
        d_t /= (env.time_horizon * n_traj)
        # Print the final true state heatmap
        print_heatmap(self, d_t, "Final State Distribution")
        # Print the ending policy
        print_gridworld_with_policy(self, env, title="Ending Policy")


## REWARD POMDP ##


class REINFORCEAgentPOMDP(REINFORCEAgent):
    '''
    This calls implements the Reinforce Agent for the POMDP defined above.
    It is implemented as the extension of the former MDP agent but adds the belief state and what is concerned by it.
    '''
    def __init__(self, env, alpha=0.1, gamma=0.9):
        super().__init__(env=env, alpha=alpha, gamma=gamma)
        self.belief_state = np.ones(env.observation_space.n) / env.observation_space.n
        self.n_expected_value = 30

    def belief_update(self, action, observation):
        '''
        This method updates the belief of the agent in a Bayesian way.
        '''
        # Get the observation probabilities for all states
        obs_probabilities = self.env.observation_matrix[:, observation]
        # Calculate the transition probability from the previous belief
        sum_value = np.sum(self.env.transition_matrix[:, :, action] * self.belief_state, axis=1)
        # Calculate the numerator: Obs(o|s) * sum_s' (P(s|s',a) * belief_state[s'])
        numerator = obs_probabilities * sum_value
        # Calculate the denominator: sum_s' (Obs(o|s') * sum_s'' (P(s'|s'',a) * belief_state[s''))
        denominator = np.sum(numerator)
        # Update the belief state
        self.belief_state = numerator / denominator

    def get_state(self, belief, behaviour):
        '''
        This method is used to sample or compute the expected state given the current belief.
        Args:
         - behaviour: if 0 it returns the state from a single sampling.
                 if 1 it returns the state from multiple samplings.
                 No other values allowed.
        '''
        if behaviour != 0 and behaviour != 1:
            raise Exception("You have to pass me 0 or 1, read :/")
        if behaviour == 0:
            state = random.choices(range(belief.size), weights=belief)[0]
            #state = np.random.choice(belief.size, p=belief, size=1)
        elif behaviour == 1:
            states = random.choices(range(belief.size), k=self.n_expected_value, weights=belief)
            #states = np.random.choice(belief.size, p=belief, size=self.n_expected_value)
            state = mode(states)[0]
        state = self.env.index_to_state(state)
        return state

    def play(self, env, n_traj):
        # Initialize episodes array
        episodes = []
        # Initialize rewards array
        total_rewards = []
        # Sample trajectories
        for k in range(n_traj):
            # Initialize episode array
            episode = []
            total_reward = 0
            # Initialize the belief state
            self.belief_state = np.ones(env.observation_space.n) / env.observation_space.n
            # Reset the environment
            env.reset()
            done = False
            while not done:
                # Sample action and get probabilities from the belief
                action, probs = self.get_action(self.belief_state)
                # Take a step of the environment
                next_obs, reward, done, _, true_state = env.step(action)
                # Save the step of the environment
                episode.append((self.belief_state, action, probs, reward, true_state))
                # Update the belief
                self.belief_update(action, next_obs)
                # Sum step reward
                total_reward += reward
            # Save episode
            episodes.append(episode)
            # Save total reward
            total_rewards.append(total_reward)
        return episodes, total_rewards


## ENTROPY POMDP ##

class REINFORCEAgentEPOMDP(REINFORCEAgentE):
    '''
    This calls implements the Reinforce Agent for the POMDP defined above.
    It is implemented as the extension of the former MDP agent but adds the belief state and what is concerned by it.
    '''
    def __init__(self, env, alpha=0.1, gamma=0.9):
        super().__init__(env=env, alpha=alpha, gamma=gamma)
        self.belief_state = np.ones(env.observation_space.n) / env.observation_space.n
        self.n_expected_value = 30

    def belief_update(self, action, observation):
        '''
        This method updates the belief of the agent in a Bayesian way.
        '''
        # Get the observation probabilities for all states
        obs_probabilities = self.env.observation_matrix[:, observation]

        # Calculate the transition probability from the previous belief
        sum_value = np.sum(self.env.transition_matrix[:, :, action] * self.belief_state, axis=1)

        # Calculate the numerator: Obs(o|s) * sum_s' (P(s|s',a) * belief_state[s'])
        numerator = obs_probabilities * sum_value

        # Calculate the denominator: sum_s' (Obs(o|s') * sum_s'' (P(s'|s'',a) * belief_state[s''))
        denominator = np.sum(numerator)

        # Update the belief state
        self.belief_state = numerator / denominator

    def get_state(self, belief, behaviour):
        '''
        This method is used to sample or compute the expected state given the current belief.
        Args:
         - behaviour: if 0 it returns the state from a single sampling.
                 if 1 it returns the state from multiple samplings.
                 No other values allowed.
        '''
        if behaviour != 0 and behaviour != 1:
            raise Exception("You have to pass me 0 or 1, read :/")
        if behaviour == 0:
            state = random.choices(range(belief.size), weights=belief)[0]
            #state = np.random.choice(belief.size, p=belief, size=1)
        elif behaviour == 1:
            states = random.choices(range(belief.size), k=self.n_expected_value, weights=belief)
            #states = np.random.choice(belief.size, p=belief, size=self.n_expected_value)
            state = mode(states)[0]
        state = self.env.index_to_state(state)
        return state

    def play(self, env, n_traj):
        # Initialize episodes array
        episodes = []
        # Initialize entropies arrays
        entropies = []
        true_entropies = []
        for k in range(n_traj):
            # Initialize episode array
            episode = []
            # Initialize the state visitations arrays
            d_t = np.zeros(env.observation_space.n)
            true_d_t = np.zeros(env.observation_space.n)
            # Initialize the belief states
            self.belief_state = np.ones(env.observation_space.n) / env.observation_space.n
            # Reset the environment
            env.reset()
            done = False
            while not done:
                # Sample action and get probabilities from the belief
                action, probs = self.get_action(self.belief_state)
                # Sample state
                sampled_state = self.get_state(self.belief_state, 1)
                # Get the index of the state
                state_index = self.env.state_to_index(sampled_state)
                # Take a step of the environment
                next_obs, reward, done, _ ,true_state = env.step(action)
                # Get true_state from dict
                true_state_index = true_state['true_state']
                # Update state visitation
                d_t[state_index] += 1
                # Update state visitation
                true_d_t[true_state_index] += 1
                episode.append((self.belief_state, action, probs, reward, true_state))
                self.belief_update(action, next_obs)
            # Compute true entropy of the trajectory
            true_d_t /= len(episode)
            true_entropies.append(self.compute_entropy(true_d_t))
            # Compute believed entropy
            d_t /= len(episode)
            
            episodes.append((episode, d_t))
        return episodes, true_entropies

    def print_visuals(self, env, n_traj):
        num_rows = 2  # Number of rows in the grid
        num_cols = 3  # Number of columns in the grid
        # Create a larger figure with subplots
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 12))
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        # Visualization of policy and expected state visitation
        d_t = np.zeros(env.observation_space.n)
        true_d_t = np.zeros(env.observation_space.n)
        for i in range(n_traj):
            # Initialize the belief states
            self.belief_state = np.ones(env.observation_space.n) / env.observation_space.n
            # Reset the environment
            env.reset()
            done = False
            while not done:
                # Sample action and get probabilities from the belief
                action, _ = self.get_action(self.belief_state)
                # Sample state
                sampled_state = self.get_state(self.belief_state, 1)
                # Get the index of the state
                state_index = self.env.state_to_index(sampled_state)
                # Take a step of the environment
                next_obs, _, done, _, true_state = env.step(action)
                # Get true_state from dict
                true_state_index = true_state['true_state']
                # Update state visitation
                d_t[state_index] += 1
                # Update state visitation
                true_d_t[true_state_index] += 1
                self.belief_update(action, next_obs)
        # Normalize the state visitations
        true_d_t /= (env.time_horizon * n_traj)
        d_t /= (env.time_horizon * n_traj)
        # Plot the final true state heatmap
        print_heatmap(self, true_d_t, "Final True State Distribution", ax=axes[0, 0])
        # Plot the final believed state heatmap
        print_heatmap(self, d_t, "Final Believed State Distribution", ax=axes[0, 1])
        # Plot KL divergence between d_t and true d_t
        kl_divergence1 = entropy(d_t, true_d_t)
        axes[0, 2].text(0.5, 0.5, f"KL divergence(d_t, true_d_t):\n{kl_divergence1:.4f}", ha='center', va='center')
        axes[0, 2].axis('off')
        # Plot KL divergence between true_d_t and d_t
        kl_divergence2 = entropy(true_d_t, d_t)
        axes[1, 0].text(0.5, 0.5, f"KL divergence(true_d_t, d_t):\n{kl_divergence2:.4f}", ha='center', va='center')
        axes[1, 0].axis('off')
        # Plot the ending policy
        print_gridworld_with_policy(self, env, title="Ending Policy", ax=axes[1, 1])
        # Use plt.show() once at the end to display the entire combined visualization
        plt.show()


class REINFORCEAgentEPOMDPVec(REINFORCEAgentE):

    def __init__(self, env, alpha=0.1, gamma=0.9, n_traj=50):
        super().__init__(env=env, alpha=alpha, gamma=gamma)
        # Initialization of the belief for each parallel environment
        self.beliefs = np.ones((n_traj, env.observation_space.n)) / env.observation_space.n
        # Number of times the agent samples from the belief to get the believed state
        self.n_expected_value = 30

    def belief_update(self, actions, observations):
        batch_size = len(actions)
        new_beliefs = np.zeros_like(self.beliefs)
        for i in range(batch_size):
            # Get the observation probabilities for all states
            obs_probabilities = self.env.observation_matrix[:, observations[i]]
            # Calculate the transition probability from the previous belief
            sum_value = np.sum(self.env.transition_matrix[:, :, actions[i]] * self.beliefs[i], axis=1)
            # Calculate the numerator: Obs(o|s) * sum_s' (P(s|s',a) * belief_state[s'])
            numerators = obs_probabilities * sum_value
            # Calculate the denominator: sum_s' (Obs(o|s') * sum_s'' (P(s'|s'',a) * belief_state[s''))
            denominators = np.sum(numerators)
            # Update the belief state for this trajectory
            new_beliefs[i, :] = numerators / denominators
        # Update the beliefs for all trajectories
        self.beliefs = new_beliefs

    def get_states(self):
        '''
        This method is used to sample the believed states of each parallel environment given the current beliefs.
        '''
        # Sample the believed state from the belief of each parallel environment
        return np.array([self.get_state(belief, 1) for belief in self.beliefs])
    
    def get_state(self, belief, behaviour):
        '''
        This method is used to sample or compute the expected state given the current belief.
        Args:
         - behaviour: if 0 it returns the state from a single sampling.
                 if 1 it returns the state from multiple samplings.
                 No other values allowed.
        '''
        if behaviour != 0 and behaviour != 1:
            raise Exception("You have to pass me 0 or 1, read :/")
        if behaviour == 0:
            state = random.choices(range(belief.size), weights=belief)[0]
            #state = np.random.choice(belief.size, p=belief, size=1)
        elif behaviour == 1:
            states = random.choices(range(belief.size), k=self.n_expected_value, weights=belief)
            #states = np.random.choice(belief.size, p=belief, size=self.n_expected_value)
            state = mode(states)[0]
        state = self.env.index_to_state(state)
        return state

    def get_actions(self):
        '''
        This method is used to sample the actions of each parallel environment given the current beliefs.
        '''
        # Get the policy probabilities for each state of each parallel environment
        probs = np.array([self.get_probability(state) for state in self.beliefs])
        # Sample the actions from the policy for each parallel environment
        actions = np.array([random.choices(range(len(prob)), weights=prob)[0] for prob in probs])
        #actions = np.array([np.random.choice(len(prob), p=prob) for prob in probs])
        return actions, probs
        
    def print_visuals(self, envs, env, n_traj):
        num_rows = 2  # Number of rows in the grid
        num_cols = 3  # Number of columns in the grid
        # Create a larger figure with subplots
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        # Initialization of true and believed state visitation
        d_t = np.zeros(env.observation_space.n)
        true_d_t = np.zeros(env.observation_space.n)
        # Reset the environments
        envs.reset()
        # Reset the belief for the agent in each environment
        self.beliefs = np.ones((n_traj, env.observation_space.n)) / env.observation_space.n
        # Reset the done flags for each environment 
        done = np.zeros(n_traj, dtype=bool)
        while not np.all(done):
            # Sample action and get probabilities from the belief
            actions, _ = self.get_actions()
            # Sample state
            sampled_states = self.get_states()
            # Take a step in the parallel environments
            next_obs, _, done, _, true_states = envs.step(actions)
            # Get the indices of the states for all parallel environments
            state_indices = [env.state_to_index(state) for state in sampled_states]
            true_state_indices = true_states['true_state']
            # Update state visitation counts for all parallel environments
            for state_index in state_indices:
                d_t[state_index] += 1
            for true_state_index in true_state_indices:
                true_d_t[true_state_index] += 1
            # Update belief
            self.belief_update(actions, next_obs)
        # Normalize the state visitation
        true_d_t /= (env.time_horizon * n_traj)
        d_t /= (env.time_horizon * n_traj)
        # Plot the final true state heatmap
        print_heatmap(self, true_d_t, "Final True State Distribution", ax=axes[0, 0])

        # Plot the final believed state heatmap
        print_heatmap(self, d_t, "Final Believed State Distribution", ax=axes[0, 1])

        # Plot KL divergence between d_t and true d_t
        kl_divergence1 = entropy(d_t, true_d_t)
        axes[0, 2].text(0.5, 0.5, f"KL divergence(d_t, true_d_t):\n{kl_divergence1:.4f}", ha='center', va='center')
        axes[0, 2].axis('off')

        # Plot KL divergence between true_d_t and d_t
        kl_divergence2 = entropy(true_d_t, d_t)
        axes[1, 0].text(0.5, 0.5, f"KL divergence(true_d_t, d_t):\n{kl_divergence2:.4f}", ha='center', va='center')
        axes[1, 0].axis('off')

        # Plot the ending policy
        print_gridworld_with_policy(self, env, title="Ending Policy", ax=axes[1, 1])

        # Use plt.show() once at the end to display the entire combined visualization
        plt.show()


## DEEP PART


import tensorflow as tf

class Buffer():
    def __init__(self):
        self.observations = []
        self.actions = []
        self.entropies = []

    def store(self, temp_traj, entropy):
        if len(temp_traj) > 0:
            self.observations.extend([item[0] for item in temp_traj])
            self.actions.extend([item[1] for item in temp_traj])
            self.entropies.extend(np.full(len(temp_traj), entropy))

    def get_batch(self):
        return np.array(self.observations), np.array(self.actions), np.array(self.entropies)

    def __len__(self):
        assert (len(self.observations) == len(self.actions) == len(self.entropies))
        return len(self.observations)


def create_policy_network(num_observations, num_actions):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(num_observations,)),  # Define the input shape
        tf.keras.layers.Reshape((1, num_observations)),  # Reshape for LSTM input
        tf.keras.layers.LSTM(64, return_sequences=True),  # Add LSTM layer
        tf.keras.layers.Flatten(),  # Flatten the LSTM output
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_actions, activation='softmax')
    ])
    
    return model
    '''model = tf.keras.Sequential([
        #tf.keras.layers.Input(shape=(num_observations,)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(num_actions, activation='softmax')
    ])
    return model'''

class DeepREINFORCEAgentE():
    def __init__(self, obs_dim, act_dim, time_horizon, learning_rate):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.time_horizon = time_horizon
        # Create the model
        self.policy_network = create_policy_network(obs_dim, act_dim)
        # Set the policy params for visualization
        self.policy_params = self.get_params_from_network()
        # Define the Adam optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)

    def get_params_from_network(self):
        policy_params = np.empty((self.obs_dim, self.act_dim))
        for i in range(self.obs_dim):
            probs = self.policy_network.predict(np.eye(self.obs_dim)[i].reshape(1, -1))[0]
            policy_params[i, :] = probs
        return policy_params

    def compute_entropy(self, d_t):
        '''
        This method computes the entropy of a probability distribution.

        Args:
         - d_t: the probability distribution array.
        '''
        # Create the condition to apply the log
        condition = d_t > 0
        # Compute log of d_t
        log_d_t = -np.log(d_t, where=condition)
        # Compute the entropy
        entropy = np.sum(np.multiply(d_t, log_d_t))
        return entropy

    def get_action(self, state):
        # Given a state, sample an action from the policy network
        action_probabilities = self.policy_network.predict(state.reshape(1, -1))[0]
        action = np.random.choice(len(action_probabilities), p=action_probabilities)
        return action

    def train(self, states, actions, rewards):
        with tf.GradientTape() as tape:
            action_probabilities = self.policy_network(states)
            print(action_probabilities)
            action_masks = tf.one_hot(actions, depth=len(action_probabilities[0]))
            print(action_masks)
            selected_action_probabilities = tf.reduce_sum(action_probabilities * action_masks, axis=1)
            print(selected_action_probabilities)
            loss = -tf.reduce_sum(tf.math.log(selected_action_probabilities) * rewards)
            print(loss)

        gradients = tape.gradient(loss, self.policy_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.policy_network.trainable_variables))
        # Set the policy params for visualization
        self.policy_params = self.get_params_from_network()

'''
class DeepREINFORCEAgentE():
    def __init__(self, obs_dim, act_dim, learning_rate):
        # Define the input placeholders (similar to your old code)
        self.obs_input = tf.keras.layers.Input(shape=(obs_dim,))
        self.act_input = tf.keras.layers.Input(shape=(), dtype=tf.int32)
        self.return_input = tf.keras.layers.Input(shape=(), dtype=tf.float32)

        # Define the policy network using TensorFlow layers
        x = tf.keras.layers.Dense(64, activation=tf.nn.tanh)(self.obs_input)
        p_logits = tf.keras.layers.Dense(act_dim, activation=None)(x)

        # Sample action using tf.random.categorical
        act_multn = tf.squeeze(tf.random.categorical(p_logits, 1))

        # Calculate action mask and log probabilities
        action_mask = tf.one_hot(self.act_input, depth=act_dim)
        p_log = tf.reduce_sum(action_mask * tf.nn.log_softmax(p_logits), axis=1)

        # Calculate policy loss
        p_loss = -tf.reduce_mean(p_log * self.return_input)

        # Define the policy optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate)

        # Define the model
        self.policy_params = tf.keras.Model(inputs=[self.obs_input, self.act_input, self.return_input], outputs=[p_logits, p_loss])

        # Compile the model with the optimizer
        self.policy_params.compile(optimizer=optimizer, loss=[None, lambda _, loss: loss])

    def get_action(self, obs):
        # Add a batch dimension to the observation
        obs_with_batch = tf.expand_dims(obs, axis=0)

        # Forward pass through the policy network
        policy_logits = self.policy_params(obs_with_batch)
        
        # Sample an action using the policy logits (e.g., using tf.random.categorical)
        action = tf.random.categorical(policy_logits, 1)
        return action.numpy()[0]  # Return as a single integer

    def Train(self, obs_batch, act_batch, ret_batch):
        # Train the policy network
        self.policy_params.train_on_batch([obs_batch, act_batch, ret_batch], [None, None])

# Example usage:
# agent = DeepREINFORCEAgentE(obs_dim, act_dim, learning_rate)
# action = agent.GetAction(observation)
# agent.Train(obs_batch, act_batch, ret_batch)
'''