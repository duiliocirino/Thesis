import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.stats import t, entropy, mode
import itertools

from utility import print_gridworld_with_policy, print_heatmap, weighted_mean

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
        #action = random.choices(range(len(probs)), weights=probs)[0]
        action = np.random.choice(len(probs), p=probs)
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
               if nothing initialize with the uniform policy;
     - true_feedback: boolean to pass the true states as feedback from the trajectories;
     - belief: boolean for whether the policy in use will be a belief based one or observation based one;
     - states_policy: boolean to use the markovian on the states or markovian on the observations when belief=False;
     - random_init_policy: boolean to initialize the policy with random values;
     - noise_oracle: approximation factor that controls the variance of the gaussian for the belief approximation;
     - sample_belief: boolean to turn on state sampling from the belief.
    '''
    def __init__(self, env, alpha=0.1, gamma=0.9, policy=-1, true_feedback=False, belief=False, states_policy=False, random_init_policy=False, noise_oracle=0, sample_belief=False, obs_bound= False):
        super().__init__(env=env, alpha=alpha, gamma=gamma)
        # Optional initialization to have a non-uniform initial policy
        self.random_init_policy = random_init_policy
        self.true_feedback = true_feedback
        self.enable_belief_policy = False
        self.states_policy = states_policy
        self.beliefs_dict = {}
        self.belief = belief
        self.noise_oracle = noise_oracle
        self.sample_belief = sample_belief
        self.init_policy(policy)
        # Handle case with belief
        if self.belief:
            # Initialize the belief states
            self.belief_state = np.ones(env.n_states * (self.env.goggles + 1)) / env.n_states
            if env.goggles:
                self.belief_state[env.n_states:] = 0
            self.policy_params = np.zeros((self.belief_state.size, env.action_space.n))
            
            print(self.policy_params.shape)
        if self.states_policy:
            self.ohe_states = np.identity(env.n_states * (self.env.goggles + 1))

    def init_policy(self, policy):
        if policy == 0:
            self.policy_params = self.create_dummy_policy()
        elif policy == 1:
            self.policy_params = self.create_optimal_policy()
        else:
            if not self.states_policy:
                self.policy_params = np.ones((self.env.observation_space.n, self.env.action_space.n))
            else:
                # This line handles the case in which you have goggles that multiply the n_states * 2, but also the case in which the observation are different from the n_states
                self.policy_params = np.ones((self.env.n_states * (self.env.goggles + 1), self.env.action_space.n))
        if self.random_init_policy:
            self.policy_params = np.random.rand(*self.policy_params.shape)*2

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

    def add_noise(self):
        '''
        This function is the implementation of the approximation oracle that changes the belief by applying Gaussian noise to the real belief update.
        '''
        # Generate random noise with the same shape as the input distribution
        noise = np.random.normal(loc=0, scale=self.noise_oracle, size=self.belief_state.shape)
        # Add the noise to the original distribution
        self.belief_state = self.belief_state + noise
        # Clip values to ensure they remain within [0, 1]
        self.belief_state = np.clip(self.belief_state, 0, 1)
        # Normalize the distribution to ensure it sums to 1
        self.belief_state /= np.sum(self.belief_state)

    def belief_update(self, action, observation):
        '''
        This method updates the belief of the agent in a Bayesian way.
        '''
        # Get the observation probabilities for all states
        obs_probabilities = self.env.observation_matrix[:, observation]

        #print("Obs probability: " + str(obs_probabilities))
        #print("Belief: " + str(self.belief_state))

        # Calculate the transition probability from the previous belief
        sum_value = np.sum(self.env.transition_matrix[:, :, action] * self.belief_state, axis=1)

        #print("Sum Value: " + str(sum_value))

        # Calculate the numerator: Obs(o|s) * sum_s' (P(s|s',a) * belief_state[s'])
        numerator = obs_probabilities * sum_value

        #print("Numerator: " + str(numerator))

        # Calculate the denominator: sum_s' (Obs(o|s') * sum_s'' (P(s'|s'',a) * belief_state[s''))
        denominator = np.sum(numerator)

        #print("Denominator: " + str(denominator))

        # Check if denominator is zero to avoid division by zero
        if denominator != 0:
            # Update the belief state
            self.belief_state = numerator / denominator
        else:
            # Handle the case where denominator is zero (you can set belief_state to some default value)
            # For example, setting belief_state to a uniform distribution:
            self.belief_state = np.ones_like(self.belief_state) / len(self.belief_state)
            print("division by zero avoided")

        # Add noise if specified TODO:check correctness
        if hasattr(self, 'noise_oracle'):
            if self.noise_oracle != 0:
                old_belief = self.belief_state.copy()
                self.add_noise()
                diff = np.sum(np.abs(self.belief_state - old_belief))
                return diff

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
            #state = random.choices(range(belief.size), weights=belief)[0]
            state = np.random.choice(belief.size, p=belief, size=1)
        elif behaviour == 1:
            #states = random.choices(range(belief.size), k=self.n_expected_value, weights=belief)
            states = np.random.choice(belief.size, p=belief, size=self.n_expected_value)
            state = mode(states)[0]
        state = self.env.index_to_state(state)
        return state
    
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

    def update_multiple_sampling(self, trajectories, objective=0, baseline_type=0, entropy_coefficient=1, belief_reg=0):
        '''
        This version of the update takes into consideration the approximation of the gradient by sampling multiple trajectories. Instead
        of working with only one trajectory it works with multiple trajectories in order to have a more accurate representation of the expected
        value of the ∇J(θ).

        Args:
         - trajectories: a list of sampled trajectories from which we compute the policy gradient;
         - objective: parameter to change the objective function of the learning between the baselines we discussed in the document
            0: proxy objective and lower bound
            1: belief entropy objective
         - baseline_type: parameter which changes the type of baseline used for the baselined version of the algorithm
            0: optimal baseline
            1: mean baseline
         - entropy coefficient: a multiplier for the entropy of the trajectory;
         - belief_reg: a multiplier for the belief regularization factor.
        '''
        # Initialise the policy gradient over the trajectories
        grad = np.zeros_like(self.policy_params)
        # Initialise array of policy gradients of the k trajectories 
        list_grads = []
        # Initialise array to save the entropies of the k trajectories
        entropies = []
        # Initialise array to save the entropies of the belief/obervation function of the k trajectories
        list_reg_entropies = []

        # Compute all entropies and gradients
        for episode, d_t in trajectories:
            if self.belief:                                          ## Calculate the sum of belief's entropies for the bound
                # Extract beliefs per each step along the trajectory
                step_beliefs = [step[0] for step in episode]
                belief_entropies = []
                # Compute the entropy of the belief for each step of the trajectory
                for belief in step_beliefs:
                    belief_entropies.append(self.compute_entropy(belief))
                # Save the entropies
                list_reg_entropies.append(belief_entropies)
                # Compute the sum of the belief vectors
                sum_beliefs = np.sum(step_beliefs, axis=0)

                # Normalize for the length of the trajectory
                sum_beliefs = sum_beliefs / len(episode)
            elif self.states_policy == False:                            ## Calculate the values for the new bound
                # Compute entropy for each observation
                entropies_o = np.zeros(self.env.observation_space.n)
                for obs in range(self.env.observation_space.n):
                    entropies_o[obs] = self.compute_entropy(self.env.observation_matrix[:, obs])

                # Weighted sum of entropies
                sum_entropies_o = np.dot(d_t, entropies_o)

                # Append to the list
                list_reg_entropies.append(sum_entropies_o)

            # Handle goggles by adding the states with and without glasses to be the same state
            if self.env.goggles == True:
                if objective == 0:
                    d_t = d_t[:self.env.n_states] + d_t[self.env.n_states:]
                elif objective == 1:
                    sum_beliefs = sum_beliefs[:self.env.n_states] + sum_beliefs[self.env.n_states:]
            # Compute and save entropy based on the objective we want to maximise
            if objective == 0:
                entropies.append(self.compute_entropy(d_t))
            elif objective == 1:        # this works with belief=True
                entropies.append(self.compute_entropy(sum_beliefs))
            
            # Initialize the gradient of the k-th sampled trajectory
            grad_k = np.zeros_like(self.policy_params)
            # Compute the gradient
            for t in range(len(episode)):
                state, action, probs, _, _, _, _ = episode[t]
                # Compute the policy gradient
                dlogp = np.zeros(self.env.action_space.n)
                for i in range(self.env.action_space.n):
                    dlogp[i] = 1.0 - probs[i] if i == action else -probs[i]
                if self.enable_belief_policy:
                    grad_k[self.beliefs_dict[tuple(state)]] += dlogp
                else:
                    grad_k += np.outer(state, dlogp)
            # Save the gradient of the trajectory
            list_grads.append(grad_k)
            # Sum the k-th gradient to the final gradient
            grad += grad_k
        
        if self.belief:
            # Calculate the sum of the entropies beliefs
            sum_entropies_belief = (np.sum(list_reg_entropies, axis=1))
            # Calculate the bound
            bound = (np.log(self.env.observation_space.n) / np.log(self.env.observation_space.n - 1) * sum_entropies_belief -
                    np.log(self.env.observation_space.n) * np.sqrt(np.log(2 / 0.95) / (2*len(trajectories))))
        else:
            if list_reg_entropies:  # Check if list_reg_entropies is not empty
                bound = np.array(list_reg_entropies)
            else:
                bound = np.zeros_like(entropies)
            # print(f"Entropy={entropies} and Bound={bound}")

        # Regularize entropies if needed
        if entropy_coefficient == 0:
            reg_entropies = - belief_reg * bound
        elif entropy_coefficient:
            reg_entropies = entropy_coefficient * entropies - belief_reg * bound
            #print(f"D_t: {d_t}\nEntropies: {entropies}\nBound: {bound}\nReg Entropies: {reg_entropies}")

        # Compute baseline based on the type requested
        baseline = 0
        if baseline_type == 0:
            # Product between gradient of the policy and the entropy
            sum_prod = np.sum([grad * entropy for grad, entropy in zip(list_grads, reg_entropies)])
            baseline = (sum_prod / grad)
        elif baseline_type == 1:
            # Mean of the regularized entropies
            baseline = np.mean(reg_entropies)

        ## Update policy parameters using the approximated gradient of the entropy objective function
        grad = np.zeros_like(self.policy_params)
        for k, episode in enumerate(trajectories):
            # Compute entropy
            entropy = reg_entropies[k]
            # Compute advantage
            advantage = entropy - baseline
            # Sum the k-th gradient to the final gradient
            grad += list_grads[k] * advantage
        # Divide the gradient by the number of trajectories sampled
        grad /= len(trajectories)
        # Update the policy parameters
        self.policy_params += self.alpha * grad
        return entropies, np.mean(bound)

    def play(self, env, n_traj=1, seed=None):
        '''
        This method samples n trajectories from the environment for the agent.

        Args:
         - env: the instance of the environment in which the agent samples the trajectories.
         - n_traj: the number of trajectories that we have to sample in one episode.
        '''
        if n_traj == -1: ## Old case: to remove
            # Initialize episode array
            episode = []
            # Initialize the state visitation array
            d_t = np.zeros(env.observation_space.n)
            # Reset the environment
            state, _ = env.reset(seed=seed)
            done = False
            while not done:
                # Update state visitation
                d_t[state] += 1
                # Get the one hot encoding
                ohe_state = self.ohe_states[state, :]
                # Sample the action from the state
                action, probs = self.get_action(ohe_state)
                # Compute the step
                next_state, reward, done, _, _ = env.step(action)
                episode.append((ohe_state, action, probs, reward, next_state, state, next_state))
                state = next_state
            d_t /= len(episode)
            return episode, d_t
        else:
            # Initialize episodes array
            episodes = []
            true_entropies = []
            for _ in range(n_traj):
                # Initialize episode array
                episode = []
                # Initialize the state visitation array
                d_t = np.zeros(env.observation_space.n)
                true_d_t = np.zeros(env.n_states * (self.env.goggles + 1))
                # Reset the environment
                if self.belief:
                    # Initialize the belief states
                    self.belief_state = np.ones(env.n_states * (self.env.goggles + 1)) / env.n_states
                    if env.goggles:
                        self.belief_state[env.n_states:] = 0
                # Get initial state (observation)
                state, initial_true_state = env.reset(seed=seed)
                if self.states_policy:
                    state = initial_true_state['true_state']
                done = False
                
                while not done:
                    # Get the one hot encoding
                    ohe_state = self.ohe_states[state, :]
                    # Sample the action from the state
                    if self.belief:
                        if self.sample_belief:
                            # Sample state
                            sampled_state = self.get_state(self.belief_state, 0)
                            # Get the index of the state
                            state_index = self.env.state_to_index(sampled_state)
                            action, probs = self.get_action(self.ohe_states[state_index, :])
                        else:
                            action, probs = self.get_action(self.belief_state)
                    else:
                        action, probs = self.get_action(ohe_state)
                    # Update state visitation
                    if not self.states_policy:
                        d_t[state] += 1
                    # Compute the step
                    next_state, reward, done, _ ,true_state = env.step(action)
                    # Get true_state from dict
                    true_state_index = true_state['true_state']
                    # Update true state visitation
                    true_d_t[true_state_index] += 1
                    if self.belief:
                        episode.append((self.belief_state, action, probs, reward, initial_true_state, state, next_state)) # In this case state=obs and next_state=next_obs
                    else:
                        episode.append((ohe_state, action, probs, reward, initial_true_state, state, next_state))
                    # Update belief state
                    if self.belief:
                        self.belief_update(action, next_state)
                    # Update latest observation
                    if not self.states_policy:
                        state = next_state
                    else: 
                        state = true_state_index
                    # Update the current initial state for the next timestep
                    initial_true_state = true_state
                if not self.states_policy:
                    d_t /= len(episode)
                # Compute true entropy of the trajectory
                true_d_t /= len(episode)
                if self.env.goggles:
                    true_entropies.append(self.compute_entropy(true_d_t[:self.env.n_states] + true_d_t[self.env.n_states:]))
                else:
                    true_entropies.append(self.compute_entropy(true_d_t))
                # Save trajectory to the list 
                if self.true_feedback:
                    episodes.append((episode, true_d_t))
                else:
                    episodes.append((episode, d_t)) # A problem here would be caused by d_t never initialized since self.states_policy is True
            return episodes, true_entropies
        
    def print_visuals(self, env, n_traj, seed=None, save_dir=None):
        # Visualization of policy and expected state visitation
        d_t = np.zeros(env.observation_space.n)
        true_d_t = np.zeros(env.n_states * (self.env.goggles + 1))
        '''episodes, true_entropies = self.play(self.env, n_traj, seed)
        for episode in episodes:
            traj = episode[0]
            d_t += episode[1]
            for step in len(traj):
                true_d_t[true_state["true_state"]] += 1'''
        # Stats init
        stats = {}
        goggles_on_at = []
        for _ in range(n_traj):
            # Reset the environment
            if self.belief:
                # Initialize the belief states
                self.belief_state = np.ones(env.n_states * (self.env.goggles + 1)) / env.n_states
                if env.goggles:
                    self.belief_state[env.n_states:] = 0
            # Get initial state (observation)
            if not self.states_policy:
                state, _ = env.reset(seed=seed)
            else:
                _, state = env.reset(seed=seed)
                state = state['true_state']
            iter=0
            goggles=False
            done = False
            while not done:
                if not self.states_policy:
                    # Update state visitation
                    d_t[state] += 1
                # Get the one hot encoding
                ohe_state = self.ohe_states[state, :]
                # Sample the action from the state
                if self.belief:
                    if self.sample_belief:
                        # Sample state
                        sampled_state = self.get_state(self.belief_state, 0)
                        # Get the index of the state
                        state_index = self.env.state_to_index(sampled_state)
                        action, probs = self.get_action(self.ohe_states[state_index, :])
                    else:
                        action, probs = self.get_action(self.belief_state)
                else:
                    action, probs = self.get_action(ohe_state)
                # Compute the step
                next_state, _, done, _, true_state = env.step(action)
                # Get true_state from dict
                true_state_index = true_state['true_state']
                # Update state visitation
                true_d_t[true_state_index] += 1
                # Update state
                if self.belief:
                    self.belief_update(action, next_state)
                if not self.states_policy:
                    state = next_state
                else: 
                    state = true_state_index
                iter+=1
                # Stats update
                if env.goggles_on and goggles == False:
                    goggles_on_at.append(iter)
                    goggles = True
        env.reset(seed=seed)
        # Normalize the state visitation
        true_d_t /= (env.time_horizon * n_traj)
        # Handle states policy
        if not self.states_policy:
            d_t /= (env.time_horizon * n_traj)
        else:
            d_t = true_d_t

        ## Plotting
            
        if self.env.goggles == False:
            num_rows = 2  # Number of rows in the grid
            num_cols = 2  # Number of columns in the grid
            # Create a larger figure with subplots
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 12))
            plt.subplots_adjust(wspace=0.5, hspace=0.5)
            # Plot the final true state heatmap
            print_heatmap(self, true_d_t, "Final True State Distribution", ax=axes[0, 0], save_dir=save_dir)
            # Plot the final believed state heatmap
            print_heatmap(self, d_t, "Final Believed State Distribution", ax=axes[0, 1], save_dir=save_dir)
            # Plot KL divergences
            kl_divergence1 = 0
            kl_divergence2 = 0
            axes[1, 0].text(0.5, 0.5, f"KL divergence(d_t, true_d_t):\n{kl_divergence1:.4f}\nKL divergence(true_d_t, d_t):\n{kl_divergence2:.4f}", ha='center', va='center')
            axes[1, 0].axis('off')
            # Plot the ending policy
            print_gridworld_with_policy(self.policy_params, env, title="Ending Policy", ax=axes[1, 1], save_dir=save_dir)
        else:
            num_rows = 2  # Number of rows in the grid
            num_cols = 3  # Number of columns in the grid
            # Create a larger figure with subplots
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 12))
            plt.subplots_adjust(wspace=0.5, hspace=0.5)
            # Adapt the state visitations
            d_t = d_t[:int(env.observation_space.n/2)] + d_t[int(env.observation_space.n/2):]
            true_d_t = true_d_t[:int(env.observation_space.n/2)] + true_d_t[int(env.observation_space.n/2):]
            # Plot the final true state heatmap
            print_heatmap(self, true_d_t, "Final True State Distribution", ax=axes[0, 0], save_dir=save_dir)
            # Plot the final believed state heatmap
            print_heatmap(self, d_t, "Final Believed State Distribution", ax=axes[0, 1], save_dir=save_dir)
            # Plot some stats on the single run
            kl_divergence1 = 0
            kl_divergence2 = 0
            if len(goggles_on_at) > 0:
                average_str = f"\n Average time step goggles: {np.mean(goggles_on_at):.2f}"
                std_str = f"\n Std: {np.std(goggles_on_at):.2f}"
            else:
                average_str = ""
                std_str = ""
            axes[0, 2].text(0.5, 0.5, f"KL divergence(d_t, true_d_t):\n{kl_divergence1:.4f}\nKL divergence(true_d_t, d_t):\n{kl_divergence2:.5f}\n\nGoggles put {len(goggles_on_at)}/{n_traj} times{average_str}{std_str}", ha='center', va='center')
            axes[0, 2].axis('off')
            # Other stats
            # axes[1, 0].text(0.5, 0.5)
            # Plot the ending policy
            print_gridworld_with_policy(self.policy_params[:int(env.observation_space.n/2), :], env, title="Ending Policy no goggles", ax=axes[1, 1], save_dir=save_dir)
            print_gridworld_with_policy(self.policy_params[int(env.observation_space.n/2):, :], env, title="Ending Policy goggles on", ax=axes[1, 2], save_dir=save_dir)
        # Use plt.show() once at the end to display the entire combined visualization
        plt.show()

## REWARD POMDP ##

# TODO: OUTDATED should be put up to date if wanted to be used
class REINFORCEAgentPOMDP(REINFORCEAgent):
    '''
    This calls implements the Reinforce Agent for the POMDP defined above.
    It is implemented as the extension of the former MDP agent but adds the belief state and what is concerned by it.
    '''
    def __init__(self, env, alpha=0.1, gamma=0.9, noise_oracle=False):
        super().__init__(env=env, alpha=alpha, gamma=gamma)
        self.belief_state = np.ones(env.observation_space.n) / env.observation_space.n
        self.noise_oracle = noise_oracle
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

    def generate_all_belief_states(self):
        '''
        This method calculates all the possible beliefs that can be generated given any pair action/observation
        starting from the initial belief.
        TODO: Add check for the feasibility, in some cases it could generate nans for this reason
        '''
        # Get the number of states and observations
        num_actions = self.env.action_space.n
        num_observations = self.env.observation_space.n
        
        # Generate all possible combinations of actions and observations
        all_combinations = list(itertools.product(range(num_actions), repeat=num_observations))
        
        all_belief_states = [self.belief_state]
        unchanged = False
        while not unchanged:
            old_all_belief_states = all_belief_states.copy()
            for belief in old_all_belief_states:
                for combination in all_combinations:
                    self.belief_state = belief
                    self.belief_update(combination[0], combination[1])
                    if not any(np.array_equal(self.belief_state, b) for b in all_belief_states) and (not np.isnan(self.belief_state).any()):
                        # print("Adding " + str(self.belief_state))
                        all_belief_states.append(self.belief_state)
            if len(old_all_belief_states) == len(all_belief_states):
                unchanged = True
        return all_belief_states

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
            #state = random.choices(range(belief.size), weights=belief)[0]
            state = np.random.choice(belief.size, p=belief, size=1)
        elif behaviour == 1:
            #states = random.choices(range(belief.size), k=self.n_expected_value, weights=belief)
            states = np.random.choice(belief.size, p=belief, size=self.n_expected_value)
            state = mode(states)[0]
        state = self.env.index_to_state(state)
        return state

    def play(self, env, n_traj, seed=None):
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
            obs, _ = env.reset(seed=seed)
            done = False
            while not done:
                # Sample action and get probabilities from the belief
                action, probs = self.get_action(self.belief_state)
                # Take a step of the environment
                next_obs, reward, done, _, true_state = env.step(action)
                # Save the step of the environment
                episode.append((self.belief_state, action, probs, reward, true_state, obs, next_obs))
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
    def __init__(self, env, alpha=0.1, gamma=0.9, true_feedback=False, belief=True, states_policy=False, enable_belief_policy=False, random_init_policy=False, noise_oracle=0, sample_belief=False):
        super().__init__(env=env, alpha=alpha, gamma=gamma, true_feedback=true_feedback, belief=belief, states_policy=states_policy, random_init_policy=random_init_policy, noise_oracle=noise_oracle, sample_belief=sample_belief)
            
        # Adapt the ohe_states
        self.ohe_states = np.identity(env.n_states * (self.env.goggles + 1))
        # Initialize the belief states
        self.belief_state = np.ones(env.n_states * (self.env.goggles + 1)) / env.n_states
        if env.goggles:
            self.belief_state[env.n_states:] = 0
        # Set policy params
        self.enable_belief_policy = enable_belief_policy
        self.policy_params = np.zeros((self.belief_state.size, env.action_space.n))
        if enable_belief_policy:
            self.policy_params = np.empty((0, env.action_space.n))
        self.n_expected_value = 30
        
        self.beliefs_dict = {}
        self.beliefs_states = {}
        self.beliefs_visits = {}
        self.n_beliefs = 0
        self.n_beliefs_story = []

    def get_probability(self, state):
        '''
        This method is an override of the previous in order to implement the possibility of the direct parametrization of every belief.
        '''
        if not self.enable_belief_policy:
            return super().get_probability(state)
        else:
            state_t = tuple(state)
            if state_t in self.beliefs_dict:
                params = self.policy_params[self.beliefs_dict[state_t]]
                return np.exp(params) / np.sum(np.exp(params))
            else:
                self.beliefs_dict[state_t] = self.n_beliefs
                self.beliefs_states[state_t] = []
                self.beliefs_visits[state_t] = 0
                self.n_beliefs += 1
                self.policy_params = np.vstack([self.policy_params, np.random.rand(self.env.action_space.n) * 3])
                return np.ones(self.env.action_space.n) / self.env.action_space.n

    def play(self, env, n_traj, seed=None, printing=False):
        # Initialize episodes array
        episodes = []
        true_entropies = []
        for k in range(n_traj):
            # Initialize episode array
            episode = []
            # Initialize the belief states
            self.belief_state = np.ones(env.n_states * (self.env.goggles + 1)) / env.n_states
            if env.goggles:
                self.belief_state[env.n_states:] = 0
            # Initialize the state visitations arrays
            d_t = np.zeros(self.belief_state.shape)
            true_d_t = np.zeros(self.belief_state.shape)
            # Reset the environment
            obs, _ = env.reset(seed=seed)
            done = False
            while not done:
                state_index = None
                if not self.true_feedback:
                    if self.belief:
                        # Sample state
                        sampled_state = self.get_state(self.belief_state, 0)
                        # Get the index of the state
                        state_index = self.env.state_to_index(sampled_state)
                        # Update state visitation
                        d_t[state_index] += 1
                    else:
                        # Update state visitation with the observation
                        d_t[obs] += 1
                # Sample action and get probabilities from the belief
                if not self.belief:
                    action, probs = self.get_action(self.ohe_states[obs, :])    # Sample from the obs case
                elif self.sample_belief:
                    if state_index == None:
                        # Sample state
                        sampled_state = self.get_state(self.belief_state, 0)
                        # Get the index of the state
                        state_index = self.env.state_to_index(sampled_state)
                    action, probs = self.get_action(self.ohe_states[state_index, :])    # Sample from the belief case
                else:
                    action, probs = self.get_action(self.belief_state)                  # Mean policy case
                # Take a step of the environment
                next_obs, reward, done, _ ,true_state = env.step(action)
                # Get true_state from dict
                true_state_index = true_state['true_state']
                # Update true state visitation
                true_d_t[true_state_index] += 1

                # Update the belief visitation (to keep count of how many times we get to a belief state) NOT NEEDED if problems comment this
                if self.enable_belief_policy:
                    states = self.beliefs_states[tuple(self.belief_state)]
                    if true_state_index not in states:
                        states.append(true_state_index)
                        self.beliefs_states[tuple(self.belief_state)] = states
                    self.beliefs_visits[tuple(self.belief_state)] += 1
                episode.append((self.belief_state, action, probs, reward, true_state, obs, next_obs))
                # Update belief
                self.belief_update(action, next_obs)
                # Update last observation
                obs = next_obs
                
            # Compute true entropy of the trajectory
            true_d_t /= len(episode)
            true_entropies.append(self.compute_entropy(true_d_t))
            # Compute believed entropy
            d_t /= len(episode)
            if self.true_feedback:
                episodes.append((episode, true_d_t))
            else:
                episodes.append((episode, d_t))
        if self.enable_belief_policy:
            self.n_beliefs_story.append(self.n_beliefs)
        return episodes, true_entropies

    def print_visuals(self, env, n_traj, seed=None, save_dir=None):
        # Visualization of policy and expected state visitation
        d_t = np.zeros(self.belief_state.shape)
        true_d_t = np.zeros(self.belief_state.shape)
        # Stats init
        if self.enable_belief_policy:
            print(f"We found {self.n_beliefs} different beliefs")
            '''
            for key, value in self.beliefs_states.items():
                if len(value) > 1:
                    print(f"{key}: {value}")
            '''
            # Sort the dictionary items by their values in descending order
            sorted_items = sorted(self.beliefs_visits.items(), key=lambda x: x[1], reverse=True)

            # Get the top 50 items
            top_50_items = sorted_items[:20]

            # Print the top 50 items
            for key, value in top_50_items:
                print(f"{key}: {value}")

        stats = {}
        goggles_on_at = []
        for i in range(n_traj):
            # Initialize the belief states
            self.belief_state = np.ones(env.n_states * (self.env.goggles + 1)) / env.n_states
            if env.goggles:
                self.belief_state[env.n_states:] = 0
            # Reset the environment
            obs, _ = env.reset(seed=seed)
            iter=0
            goggles=False
            done = False
            while not done:
                state_index = None
                if not self.true_feedback:
                    if self.belief:
                        # Sample state
                        sampled_state = self.get_state(self.belief_state, 0)
                        # Get the index of the state
                        state_index = self.env.state_to_index(sampled_state)
                        # Update state visitation
                        d_t[state_index] += 1
                    else:
                        # Update state visitation with the observation
                        d_t[obs] += 1
                # Sample state
                sampled_state = self.get_state(self.belief_state, 0)
                # Get the index of the state
                state_index = self.env.state_to_index(sampled_state)

                if not self.belief:
                    action, probs = self.get_action(self.ohe_states[obs, :])    # Sample from the obs case
                elif self.sample_belief:
                    if state_index == None:
                        # Sample state
                        sampled_state = self.get_state(self.belief_state, 0)
                        # Get the index of the state
                        state_index = self.env.state_to_index(sampled_state)
                    action, probs = self.get_action(self.ohe_states[state_index, :])    # Sample from the belief case
                else:
                    action, probs = self.get_action(self.belief_state)                  # Mean policy case
                
                # Take a step of the environment
                next_obs, _, done, _, true_state = env.step(action)
                # Get true_state from dict
                true_state_index = true_state['true_state']
                # Update state visitation
                d_t[state_index] += 1
                # Update state visitation
                true_d_t[true_state_index] += 1
                self.belief_update(action, next_obs)
                iter+=1
                # Update last observation
                obs = next_obs
                # Stats update
                if env.goggles_on and goggles == False:
                    goggles_on_at.append(iter)
                    goggles = True
        env.reset(seed=seed)
        # Normalize the state visitations
        true_d_t /= (env.time_horizon * n_traj)
        d_t /= (env.time_horizon * n_traj)
        if self.env.goggles == False:
            num_rows = 2  # Number of rows in the grid
            num_cols = 2  # Number of columns in the grid
            # Create a larger figure with subplots
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 12))
            plt.subplots_adjust(wspace=0.5, hspace=0.5)
            # Plot the final true state heatmap
            print_heatmap(self, true_d_t, "Final True State Distribution", ax=axes[0, 0], save_dir=save_dir)
            # Plot the final believed state heatmap
            print_heatmap(self, d_t, "Final Believed State Distribution", ax=axes[0, 1], save_dir=save_dir)
            # Plot KL divergences
            kl_divergence1 = 0
            kl_divergence2 = 0
            axes[1, 0].text(0.5, 0.5, f"KL divergence(d_t, true_d_t):\n{kl_divergence1:.4f}\nKL divergence(true_d_t, d_t):\n{kl_divergence2:.4f}", ha='center', va='center')
            axes[1, 0].axis('off')
            # Plot the ending policy
            print_gridworld_with_policy(self.policy_params, env, title="Ending Policy", ax=axes[1, 1], save_dir=save_dir)
        else:
            num_rows = 2  # Number of rows in the grid
            num_cols = 3  # Number of columns in the grid
            # Create a larger figure with subplots
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 12))
            plt.subplots_adjust(wspace=0.5, hspace=0.5)
            # Adapt the state visitations
            d_t = d_t[:int(env.observation_space.n/2)] + d_t[int(env.observation_space.n/2):]
            true_d_t = true_d_t[:int(env.observation_space.n/2)] + true_d_t[int(env.observation_space.n/2):]
            # Plot the final true state heatmap
            print_heatmap(self, true_d_t, "Final True State Distribution", ax=axes[0, 0], save_dir=save_dir)
            # Plot the final believed state heatmap
            print_heatmap(self, d_t, "Final Believed State Distribution", ax=axes[0, 1], save_dir=save_dir)
            # Plot some stats on the single run
            kl_divergence1 = 0
            kl_divergence2 = 0
            if len(goggles_on_at) > 0:
                average_str = f"\n Average time step goggles: {np.mean(goggles_on_at):.2f}"
                std_str = f"\n Std: {np.std(goggles_on_at):.2f}"
            else:
                average_str = ""
                std_str = ""
            axes[0, 2].text(0.5, 0.5, f"KL divergence(d_t, true_d_t):\n{kl_divergence1:.4f}\nKL divergence(true_d_t, d_t):\n{kl_divergence2:.5f}\n\nGoggles put {len(goggles_on_at)}/{n_traj} times{average_str}{std_str}", ha='center', va='center')
            axes[0, 2].axis('off')
            # Other stats
            # axes[1, 0].text(0.5, 0.5)
            # Plot the ending policy
            print_gridworld_with_policy(self.policy_params[:int(env.observation_space.n/2), :], env, title="Ending Policy no goggles", ax=axes[1, 1], save_dir=save_dir)
            print_gridworld_with_policy(self.policy_params[int(env.observation_space.n/2):, :], env, title="Ending Policy goggles on", ax=axes[1, 2], save_dir=save_dir)
        # Use plt.show() once at the end to display the entire combined visualization
        plt.show()
        stats = self.n_beliefs_story
        return stats


## DOUBLE INPUT POLICY


class REINFORCEDoubleAgentE(REINFORCEAgentEPOMDP):
    '''
    This class is the implementation of a REINFORCE agent that tries to maximize the objective function J(θ)=E_(τ ~ p_π)[R(τ)]

    Args:
     - env: a copy of the environment on which the agent is acting;
     - alpha: the value of the learning rate to compute the policy update;
     - gamma: the value of the discount for the reward in order to compute the discounted reward;
     - policy_type: a parameter that states which representation of the policy the agent will have;
     - history_type: a parameter that states whether the agent's history is the believed state visitation or the mean belief
        - 0: state visitation
        - 1: mean belief
    '''
    def __init__(self, env, alpha=0.1, gamma=0.9, true_feedback=False, enable_belief_policy=False, policy_type=0, weight_policy=0.5, history_type=0,
                 importance_belief=False, learnable_weight=0, random_init_policy=False, noise_oracle=0):
        super().__init__(env=env, alpha=alpha, gamma=gamma, true_feedback=true_feedback, random_init_policy=random_init_policy, noise_oracle=noise_oracle)
        # Policy of the agent
        self.policy_type = policy_type
        self.history_type = history_type
        self.learnable_weight = learnable_weight
        if self.learnable_weight == 1:
            self.weights_policy = np.zeros(self.env.time_horizon)
        elif self.learnable_weight == 2:
            self.weights_policy = np.zeros((2, self.env.time_horizon))
        else:
            ## May be wrong, the problem
            self.weight_policy = weight_policy
        self.importance_belief = importance_belief
        if self.policy_type == 0:
            self.policy_params = np.zeros((env.observation_space.n, env.observation_space.n, env.action_space.n))
        else:
            self.policy_params_belief = np.zeros((env.observation_space.n, env.action_space.n))
            self.policy_params_history = np.zeros((env.observation_space.n, env.action_space.n))
        # Timestep in the environment
        self.ohe_time = np.identity(self.env.time_horizon)
        self.timestep = 0
        

    def get_probability_prod(self, history, state):
        '''
        This method is used to get the action probabilities from the policy given a state.

        Args:
         - history: the history of the trajectory fed to the agent in form of the medium belief (or the state visitation);
         - state: an array representing the state from which we take an action. Sums up to 1, being a probability distribution over the states.
        '''
        # Compute the first multiplication between the mean vector and the 3D policy
        first_prod = np.dot(history, self.policy_params)
        # Compute the dot product
        params = np.dot(state, first_prod)
        # Compute the softmax
        probs = np.exp(params) / np.sum(np.exp(params))
        return probs
    
    def get_probability_mean(self, history, state, printing=False):
        '''
        This method is used to get the action probabilities from the policy given a state.

        Args:
         - history: the history of the trajectory fed to the agent in form of the medium belief (or the state visitation);
         - state: an array representing the state from which we take an action. Sums up to 1, being a probability distribution over the states.
        '''
        # Compute the dot product for the history
        history_params = np.dot(history, self.policy_params_history)
        # Compute the dot product for the state (belief)
        state_params = np.dot(state, self.policy_params_belief)
        # Compute the softmaxes
        if printing:
            print(self.policy_params_belief)
            print(state_params)
        history_probs = np.exp(history_params) / np.sum(np.exp(history_params))
        state_probs = np.exp(state_params) / np.sum(np.exp(state_params))
        # Get parametrized weights or use the fixed weight
        if self.learnable_weight == 1:
            weight_prob = 1/(1 + np.exp(-self.weights_policy))
            weight_policy = np.dot(self.ohe_time[self.timestep], weight_prob)
            weight_policy = [weight_policy, 1 - weight_policy]
        elif self.learnable_weight == 2:
            weight_prob_h = 1/(1 + np.exp(-self.weights_policy[0][self.timestep]))
            weight_prob_b = 1/(1 + np.exp(-self.weights_policy[1][self.timestep]))
            weight_policy = [weight_prob_h, weight_prob_b]
        else:
            weight_policy = [self.weight_policy, 1 - self.weight_policy]
        probs = [weight_policy, history_probs, state_probs]
        return probs
    
    def get_action(self, history, state, printing=False):
        '''
        This method is used to get the action the agent will given a certain state.

        Args:
         - history: the history of the trajectory fed to the agent in form of the medium belief (or the state visitation);
         - state: an array representing the state from which we take an action. Sums up to 1, being a probability distribution over the states.
        '''
        # Get probability vector
        if self.policy_type == 0:
            probs = self.get_probability_prod(history, state)
            action = np.random.choice(len(probs), p=probs)
        else:
            probs = self.get_probability_mean(history, state, printing=printing)
            prob_vec = probs[0][0] * probs[1] + probs[0][1] * probs[2]
            softmax = np.exp(prob_vec)/np.sum(np.exp(prob_vec))
            action = np.random.choice(len(probs[1]), p=softmax)
        # Sample the action
        
        return action, probs

    def update_multiple_sampling(self, trajectories, objective=0, baseline_type=0, entropy_coefficient=1, belief_reg=0):
        '''
        This version of the update takes into consideration the approximation of the gradient by sampling multiple trajectories. Instead
        of working with only one trajectory it works with multiple trajectories in order to have a more accurate representation of the expected
        value of the ∇J(θ).

        Args:
         - trajectories: a list of sampled trajectories from which we compute the policy gradient;
         - objective: parameter to change the objective function of the learning between the baselines we discussed in the document
            0: proxy objective and lower bound
            1: belief entropy objective
         - baseline_type: parameter which changes the type of baseline used for the baselined version of the algorithm
            0: optimal baseline
            1: mean baseline
         - entropy coefficient: a multiplier for the entropy of the trajectory;
         - belief_reg: a multiplier for the belief regularization factor.
        '''
        # Initialise array of policy gradients of the k trajectories
        if self.policy_type == 0:
            list_grads = []
        elif self.policy_type == 1:
            list_grads_his = []
            list_grads_bel = []
        # Initialise array to save the entropies of the k trajectories
        entropies = []
        # Initialise array to save the entropies of the belief of the k trajectories
        list_belief_entropies = []

        # Compute all entropies and gradients
        for episode, d_t in trajectories:
            # Extract beliefs per each step along the trajectory
            step_beliefs = [step[0] for step in episode]
            belief_entropies = []
            # Compute the entropy of the belief for each step of the trajectory
            for belief in step_beliefs:
                belief_entropies.append(self.compute_entropy(belief))
            # Save the entropies
            list_belief_entropies.append(belief_entropies)
            # Compute the sum of the belief vectors
            sum_beliefs = np.sum(step_beliefs, axis=0)

            # Handle goggles by adding the states with and without glasses to be the same state
            if self.env.goggles == True:
                if objective == 0:
                    d_t = d_t[:self.env.n_states] + d_t[self.env.n_states:]
                elif objective == 1:
                    sum_beliefs = sum_beliefs[:self.env.n_states] + sum_beliefs[self.env.n_states:]
                    sum_beliefs = sum_beliefs / len(episode)
            
            # Cpumpute and save entropy based on the objective we want to maximise
            if objective == 0:
                entropies.append(self.compute_entropy(d_t))
            elif objective == 1:
                entropies.append(self.compute_entropy(sum_beliefs))

            ## Initialize the gradient of the k-th sampled trajectory
            if self.policy_type == 0:
                grad_k = np.zeros_like(self.policy_params)
            elif self.policy_type == 1:
                grad_k_his = np.zeros_like(self.policy_params_history)
                grad_k_bel = np.zeros_like(self.policy_params_belief)
            
            # Compute the gradient
            for t in range(len(episode)):
                state, history, action, probs, _, _, _, _ = episode[t]
                # Compute the policy gradient
                if self.policy_type == 0:
                    dlogp = np.zeros(self.env.action_space.n)
                    for i in range(self.env.action_space.n):
                        dlogp[i] = 1.0 - probs[i] if i == action else -probs[i]
                    first_prod = np.outer(state, dlogp)
                    grad_k += np.dot(history.T, first_prod)
                elif self.policy_type == 1:
                    # Get probs from array
                    h_probs = probs[1]
                    s_probs = probs[2]
                    # Initialize the derivatives
                    dp_h = np.zeros(self.env.action_space.n)
                    dp_b = np.zeros(self.env.action_space.n)
                    for i in range(self.env.action_space.n):
                        dp_h[i] = (1.0 - h_probs[i]) * h_probs[action] if i == action else -h_probs[i] * h_probs[action]
                        dp_b[i] = (1.0 - s_probs[i]) * s_probs[action] if i == action else -s_probs[i] * s_probs[action]
                    u = probs[0][0] * h_probs[action] + (1 - h_probs[action]) * s_probs[action]
                    dlogp_h = 1 / u * probs[0][0] * dp_h
                    dlogp_b = 1 / u * probs[0][1] * dp_b
                    # Add step gradient to the trajectory gradient
                    grad_k_his += np.outer(history, dlogp_h)
                    grad_k_bel += np.outer(state, dlogp_b)

            # Save the gradient of the trajectory
            if self.policy_type == 0:
                list_grads.append(grad_k)
            elif self.policy_type == 1:
                list_grads_his.append(grad_k_his)
                list_grads_bel.append(grad_k_bel)
        
        # Calculate the sum of the entropies beliefs
        sum_entropies_belief = (np.sum(list_belief_entropies, axis=1))
        # Calculate the bound
        bound = (np.log(self.env.observation_space.n) / np.log(self.env.observation_space.n - 1) * sum_entropies_belief -
                 np.log(self.env.observation_space.n) * np.sqrt(np.log(2 / 0.95) / (2*len(trajectories))))
        
        # Regularize entropies if needed
        if entropy_coefficient == 0:
            reg_entropies = - belief_reg * bound
        elif entropy_coefficient:
            reg_entropies = entropy_coefficient * entropies - belief_reg * bound

        # Compute baseline based on the type requested
        baseline = 0
        if baseline_type == 0:
            # TODO: aggiustare i grad nel caso multiple policy
            # Product between gradient of the policy and the entropy
            sum_prod = np.sum([grad * entropy for grad, entropy in zip(list_grads, reg_entropies)])
            baseline = (sum_prod / grad)
        elif baseline_type == 1:
            # Mean of the regularized entropies
            baseline = np.mean(reg_entropies)

        ## Update policy parameters using the approximated gradient of the entropy objective function
        if self.policy_type == 0:
            grad = np.zeros_like(self.policy_params)
        elif self.policy_type == 1:
            grad_his = np.zeros_like(self.policy_params_history) 
            grad_bel = np.zeros_like(self.policy_params_belief)
        for k, episode in enumerate(trajectories):
            # Compute entropy
            entropy = reg_entropies[k]
            # Compute advantage
            advantage = entropy - baseline
            # Sum the k-th gradient to the final gradient    
            if self.policy_type == 0:
                grad += list_grads[k] * advantage
            elif self.policy_type == 1:
                grad_his += list_grads_his[k] * advantage
                grad_bel += list_grads_bel[k] * advantage
        # Divide the gradient by the number of trajectories sampled
        if self.policy_type == 0:
            grad /= len(trajectories)
        elif self.policy_type == 1:
            grad_his /= len(trajectories)
            grad_bel /= len(trajectories)
        # Update the policy parameters
        if self.policy_type == 0:
            self.policy_params += self.alpha * grad
        elif self.policy_type == 1:
            self.policy_params_history += self.alpha * grad_his
            self.policy_params_belief += self.alpha * grad_bel
        return entropies, np.mean(bound)
    
    def update_weights(self, episodes, objective, belief_reg=0, entropy_coefficient=1, baseline_type=1):
        '''
        This method is used in order to update the weights for each time step for the mixed policy gradient.
        For more info over the gradient look at the document.
        '''
        list_grads = []
        # Initialise array to save the entropies of the k trajectories
        entropies = []
        # Initialise array to save the entropies of the belief of the k trajectories
        list_belief_entropies = []

        for episode, d_t in episodes:
            grad_k = np.zeros_like(self.weights_policy)
            
            # Extract beliefs per each step along the trajectory
            step_beliefs = [step[0] for step in episode]
            belief_entropies = []
            # Compute the entropy of the belief for each step of the trajectory
            for belief in step_beliefs:
                belief_entropies.append(self.compute_entropy(belief))
            # Save the entropies
            list_belief_entropies.append(belief_entropies)
            # Compute the sum of the belief vectors
            sum_beliefs = np.sum(step_beliefs, axis=0)

            # Handle goggles by adding the states with and without glasses to be the same state
            if self.env.goggles == True:
                if objective == 0:
                    d_t = d_t[:self.env.n_states] + d_t[self.env.n_states:]
                elif objective == 1:
                    sum_beliefs = sum_beliefs[:self.env.n_states] + sum_beliefs[self.env.n_states:]
                    sum_beliefs = sum_beliefs / len(episode)
            
            # Cpumpute and save entropy based on the objective we want to maximise
            if objective == 0:
                entropies.append(self.compute_entropy(d_t))
            elif objective == 1:
                entropies.append(self.compute_entropy(sum_beliefs))

            if self.learnable_weight == 1:
                for t in range(len(episode)):
                    state, history, action, probs, _, _, _, _ = episode[t]
                    weight = probs[0][0]
                    prob_h = probs[1]
                    prob_s = probs[2]

                    u = prob_h[action] - prob_s[action]
                    v = weight * u + prob_s[action]
                    d_w = weight * (1 - weight)
                    dlog_w = 1/v *  d_w * u
                    grad_k[t] = dlog_w
            elif self.learnable_weight == 2:
                for t in range(len(episode)):
                    state, history, action, probs, _, _, _, _ = episode[t]
                    weight_h = probs[0][0]
                    weight_b = probs[0][1]
                    prob_h = probs[1]
                    prob_s = probs[2]

                    for p in [0, 1]:
                        u = prob_h[action] - prob_s[action]
                        v = probs[0][p] * u + prob_s[action]
                        d_w = probs[0][p] * (1 - probs[0][p])
                        dlog_w = 1/v *  d_w * u
                        grad_k[p][t] = dlog_w
            list_grads.append(grad_k)
            
        # Calculate the sum of the entropies beliefs
        sum_entropies_belief = (np.sum(list_belief_entropies, axis=1))
        # Calculate the bound
        bound = (np.log(self.env.observation_space.n) / np.log(self.env.observation_space.n - 1) * sum_entropies_belief -
                 np.log(self.env.observation_space.n) * np.sqrt(np.log(2 / 0.95) / (2*len(episodes))))
        
        # Regularize entropies if needed
        if entropy_coefficient == 0:
            reg_entropies = - belief_reg * bound
        elif entropy_coefficient:
            reg_entropies = entropy_coefficient * entropies - belief_reg * bound

        # Compute baseline based on the type requested
        baseline = 0
        if baseline_type == 0:
            # TODO: aggiustare i grad nel caso multiple policy
            # Product between gradient of the policy and the entropy
            sum_prod = np.sum([grad * entropy for grad, entropy in zip(list_grads, reg_entropies)])
            baseline = (sum_prod / grad)
        elif baseline_type == 1:
            # Mean of the regularized entropies
            baseline = np.mean(reg_entropies)

        # Update weights parameters using the approximated gradient of the entropy objective function
        grad = np.zeros_like(self.weights_policy)
        
        for k, episode in enumerate(episodes):
            # Compute entropy
            entropy = reg_entropies[k]
            # Compute advantage
            advantage = entropy - baseline
            # Sum the k-th gradient to the final gradient
            grad += list_grads[k] * advantage
        # Divide the gradient by the number of trajectories sampled
        grad /= len(episodes)
        # Update the policy parameter
        self.weights_policy += self.alpha * grad
    
    def play(self, env, n_traj, seed=None, printing=False):
        # Initialize episodes array
        episodes = []
        true_entropies = []
        for k in range(n_traj):
            # Initialize episode array
            episode = []
            # Initialize the state visitations arrays
            d_t = np.zeros(env.observation_space.n)
            true_d_t = np.zeros(env.n_states * (self.env.goggles + 1))
            # Initialize the belief states
            self.belief_state = np.ones(env.observation_space.n) / env.observation_space.n
            # Initialize timestep
            self.timestep = 0
            # Initialize belief history
            step_beliefs = []
            # Reset the environment
            obs, _ = env.reset(seed=seed)
            done = False
            while not done:
                # Sample state
                sampled_state = self.get_state(self.belief_state, 0)
                # Get the index of the state
                state_index = env.state_to_index(sampled_state)
                # Update state visitation
                d_t[state_index] += 1
                # Sample action and get probabilities from the history and belief
                if self.history_type == 0:
                    history = d_t
                    action, probs = self.get_action(history, self.belief_state)
                else:
                    step_beliefs.append(self.belief_state)
                    if self.importance_belief:
                        history = weighted_mean(step_beliefs)
                    else:
                        history = np.mean(step_beliefs, axis=0)
                    action, probs = self.get_action(history, self.belief_state, printing=printing)
                # Take a step of the environment
                next_obs, reward, done, _ ,true_state = env.step(action)
                # Get true_state from dict
                true_state_index = true_state['true_state']
                # Update true state visitation
                true_d_t[true_state_index] += 1
                episode.append((self.belief_state, history, action, probs, reward, true_state, obs, next_obs))
                # Update belief
                self.belief_update(action, next_obs)
                # Increase timestep
                self.timestep += 1
            # Compute true entropy of the trajectory
            true_d_t /= len(episode)
            true_entropies.append(self.compute_entropy(true_d_t))
            # Compute believed entropy
            d_t /= len(episode)
            if self.true_feedback:
                episodes.append((episode, true_d_t))
            else:
                episodes.append((episode, d_t))
        return episodes, true_entropies

    def print_visuals(self, env, n_traj, seed=None, save_dir=None):
        # Visualization of policy and expected state visitation
        d_t = np.zeros(env.observation_space.n)
        true_d_t = np.zeros(env.n_states * (self.env.goggles + 1))
        # Stats init
        stats = {}
        goggles_on_at = []
        for i in range(n_traj):
            # Initialize the belief states
            self.belief_state = np.ones(env.observation_space.n) / env.observation_space.n
            # Initialize timestep
            self.timestep = 0
            # Reset the environment
            obs, _ = env.reset(seed=seed)
            step_beliefs = []
            goggles=False
            done = False
            while not done:
                # Sample state
                sampled_state = self.get_state(self.belief_state, 0)
                # Get the index of the state
                state_index = env.state_to_index(sampled_state)
                # Update state visitation
                d_t[state_index] += 1
                # Sample action and get probabilities from the history and belief
                if self.history_type == 0:
                    history = d_t
                    action, probs = self.get_action(d_t, self.belief_state)
                else:
                    step_beliefs.append(self.belief_state)
                    if self.importance_belief:
                        history = weighted_mean(step_beliefs)
                    else:
                        history = np.mean(step_beliefs, axis=0)
                    action, probs = self.get_action(history, self.belief_state)
                # Take a step of the environment
                next_obs, reward, done, _ ,true_state = env.step(action)
                # Get true_state from dict
                true_state_index = true_state['true_state']
                # Update true state visitation
                true_d_t[true_state_index] += 1
                # Update belief
                self.belief_update(action, next_obs)
                # Increase timestep
                self.timestep += 1
                # Stats update
                if env.goggles_on and goggles == False:
                    goggles_on_at.append(self.timestep)
                    goggles = True
        # If you dont reset the printing dies
        env.reset(seed=seed)
        # Normalize the state visitations
        true_d_t /= (env.time_horizon * n_traj)
        d_t /= (env.time_horizon * n_traj)
        if self.env.goggles == False:
            num_rows = 3  # Number of rows in the grid
            num_cols = 2  # Number of columns in the grid
            # Create a larger figure with subplots
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(24, 12))
            plt.subplots_adjust(wspace=0.5, hspace=0.5)
            # Plot the final true state heatmap
            print_heatmap(self, true_d_t, "Final True State Distribution", ax=axes[0, 0], save_dir=save_dir)
            # Plot the final believed state heatmap
            print_heatmap(self, d_t, "Final Believed State Distribution", ax=axes[0, 1], save_dir=save_dir)
            # Plot KL divergences
            kl_divergence1 = 0
            kl_divergence2 = 0
            axes[1, 0].text(0.5, 0.5, f"KL divergence(d_t, true_d_t):\n{kl_divergence1:.4f}\nKL divergence(true_d_t, d_t):\n{kl_divergence2:.4f}", ha='center', va='center')
            axes[1, 0].axis('off')
            if self.learnable_weight > 0:
                weights = 1 / (1 + np.exp(-self.weights_policy))
                # Convert the weights array to a string representation
                weights_string = np.array2string(weights, precision=4, separator=', ', suppress_small=True)
                axes[1, 1].text(0.5, 0.5, f"Weights for policy mix:\n{weights_string}", ha='center', va='center')
                axes[1, 1].axis('off')
            # Plot the ending policy
            if self.policy_type == 0:
                print_gridworld_with_policy(self.policy_params[0], env, title="Ending Policy 0", ax=axes[1, 1])
                print_gridworld_with_policy(self.policy_params[0], env, title="Ending Policy 10", ax=axes[2, 0])
                print_gridworld_with_policy(self.policy_params[0], env, title="Ending Policy 24", ax=axes[2, 1])
            else:
                print_gridworld_with_policy(self.policy_params_history, env, title="Ending History Policy", ax=axes[2, 0])
                print_gridworld_with_policy(self.policy_params_belief, env, title="Ending Belief Policy", ax=axes[2, 1])
        else:
            num_rows = 2  # Number of rows in the grid
            num_cols = 3  # Number of columns in the grid
            # Create a larger figure with subplots
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 12))
            plt.subplots_adjust(wspace=0.5, hspace=0.5)
            # Adapt the state visitations
            d_t = d_t[:int(env.observation_space.n/2)] + d_t[int(env.observation_space.n/2):]
            true_d_t = true_d_t[:int(env.observation_space.n/2)] + true_d_t[int(env.observation_space.n/2):]
            # Plot the final true state heatmap
            print_heatmap(self, true_d_t, "Final True State Distribution", ax=axes[0, 0])
            # Plot the final believed state heatmap
            print_heatmap(self, d_t, "Final Believed State Distribution", ax=axes[0, 1])
            # Plot some stats on the single run
            kl_divergence1 = 0
            kl_divergence2 = 0
            if len(goggles_on_at) > 0:
                average_str = f"\n Average time step goggles: {np.mean(goggles_on_at):.2f}"
                std_str = f"\n Std: {np.std(goggles_on_at):.2f}"
            else:
                average_str = ""
                std_str = ""
            axes[0, 2].text(0.5, 0.5, f"KL divergence(d_t, true_d_t):\n{kl_divergence1:.4f}\nKL divergence(true_d_t, d_t):\n{kl_divergence2:.5f}\n\nGoggles put {len(goggles_on_at)}/{n_traj} times{average_str}{std_str}", ha='center', va='center')
            axes[0, 2].axis('off')
            # Other stats
            # axes[1, 0].text(0.5, 0.5)
            # Plot the ending policy
            print_gridworld_with_policy(self.policy_params[:int(env.observation_space.n/2), :], env, title="Ending Policy no goggles", ax=axes[1, 1])
            print_gridworld_with_policy(self.policy_params[int(env.observation_space.n/2):, :], env, title="Ending Policy goggles on", ax=axes[1, 2])
        # Use plt.show() once at the end to display the entire combined visualization
        plt.show()
        return stats
