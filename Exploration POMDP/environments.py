import gym
from gym import spaces
import random
import numpy as np
from scipy.stats import t, mode

class GridworldEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    """
    This class implements a custom made Gym environment of a simple Gridworld, a NxN matrix where the agent starts from a starting position I and 
    has to reach the goal position G.
    
    The set of states S = {[x,y]| x,y ∈ [0, ..., grid_size]} or {}, represent the possible positions in the environment.
    The set of actions A = [up, down, left, right].
    The transition to a new state T(s_t+1 | s_t, a_t).
    The reward R(s, a) = {1 if s == G else -0.1}.
    
    In this version the initial state is the position 0 ([0, 0]), I = [0, 0]. The goal state is the position 24 ([4, 4]) G = [4,4]. 
    
    Args:
    - grid_size: the size in height and length of the grid, N of the NxN matrix.
    - time_horizon: the maximum number of time steps the agent can take to get to the goal. If set to -1 the time horizon is ∞.
    - prob: the probability with which the environment takes the chosen action. If set to 0 the actions taken by the agent are deterministic.
    - randomize: a variable that if at 1 sets a random initial position, at 0 makes the initial position the top left corner.
    - length_corridor: the length of a corridor placed on the right bottom corner of the environment.
    - goggles: a 0-1 value that states whether to give the environment a pair of glasses that the agent receives by going on a state, which
        improves the quality of the observations.
    """
    
    def __init__(self, grid_size=5, time_horizon=-1, prob=0.1, randomize=0, length_corridor=0, goggles=False):
        self.grid_size = grid_size
        self.length_corridor = length_corridor
        self.goggles = goggles
        self.observation_space = spaces.Discrete((self.grid_size ** 2 + self.length_corridor)*(1 + self.goggles))
        self.action_space = spaces.Discrete(4)
        self.reward_range = (-0.1, 1.0)
        self.observation_range = np.array(range(self.observation_space.n))
        self.goal = (grid_size-1, grid_size-1)
        self.goggles_pos = (0, grid_size-1)
        self.n_states = int(self.observation_space.n / 2) if goggles else self.observation_space.n
        self.goggles_on = False
        self.done = False
        self.time_horizon = time_horizon
        self.steps_taken = 0
        self.prob = prob
        self.transition_matrix = self._build_transition_matrix()
        self.randomize = randomize
        if self.randomize == 1:
            self.current_pos = self.index_to_state(np.random.randint(0, self.observation_space.n))
        else:
            self.current_pos = (0, 0)
        
    def _build_transition_matrix(self):
        '''
        This method builds the transition matrix for the MDP, T(s'|a, s).
        The function should be used with the following order of operands:
         - first parameter: s, the state where the agent takes the action in
         - second parameter: s', the state where the agent arrives taking action a from state s
         - third parameter: a, the action taken by the agent
        '''
        transition_matrix = np.zeros((self.observation_space.n, self.observation_space.n, self.action_space.n))
        # For every state (i,j)
        for index in self.observation_range:
            # Handle goggles
            if self.goggles==True and index >= self.n_states:
                self.goggles_on = True
            s = self.index_to_state(index)
            # For every action 'a' chosen as action from the agent
            for a in range(self.action_space.n):
                # For all actions 'a_'
                for a_ in range(self.action_space.n):
                    # Calculate the probability of
                    prob = 1 - self.prob if a_ == a else self.prob / (self.action_space.n - 1)
                    s_ = self._sample_new_position(a_, s)
                    if s_==self.goggles_pos and self.goggles_on == False and self.goggles:
                        transition_matrix[self.state_to_index(s_) + self.n_states, self.state_to_index(s), a] += prob
                    else:
                        transition_matrix[self.state_to_index(s_), self.state_to_index(s), a] += prob
        self.goggles_on = False
        return transition_matrix
    
    def _sample_new_position(self, action, state):
        if action == 0:  # up
            if state[1] >= self.grid_size:
                new_pos = (state[0], state[1] - 1) # if you are in the corridor you just go one y less
            else:
                new_pos = (state[0], max(0, state[1]-1))
        elif action == 1:  # down
            if state == (self.grid_size - 1, self.grid_size - 1) or state[1] >= self.grid_size:
                new_pos = (state[0], min(self.grid_size + self.length_corridor - 1, state[1] + 1)) # if you are in the corridor, or you are entering the corridor, you add one unless you are at the end of the corridor
            else:
                new_pos = (state[0], min(self.grid_size-1, state[1]+1))
        elif action == 2:  # left
            if state[1] >= self.grid_size:
                new_pos = state # if you are in the corridor you don't move
            else:
                new_pos = (max(0, state[0]-1), state[1])
        elif action == 3:  # right
            if state[1] >= self.grid_size:
                new_pos = state # if you are in the corridor you don't move
            else:
                new_pos = (min(self.grid_size-1, state[0]+1), state[1])
        else:
            raise ValueError("Invalid action.")
        return new_pos
    
    def _get_info(self):
        return {"coord": self.current_pos}
    
    def state_to_index(self, state):
        index = 0
        if state[1] >= self.grid_size:
            index = self.grid_size ** 2 + state[1] - self.grid_size
        else:
            index = state[0] + state[1] * self.grid_size
        return int(index + self.n_states * self.goggles_on)

    def index_to_state(self, index):
        """Converts an index to a state tuple (i, j)."""
        ind = int(index - (self.n_states * self.goggles_on))
        if ind >= self.grid_size ** 2:
            i = self.grid_size - 1
            j = self.grid_size + ind - self.grid_size ** 2
            return (i, j)
        i = ind % self.grid_size
        j = ind // self.grid_size
        #print("Converting %d => (%d, %d)" %(index, i, j))
        return (i, j)
    
    def sample_next_state(self, action):
        current_state = self.state_to_index(self.current_pos)
        action_probabilities = self.transition_matrix[:, current_state, action]
        next_state_index = random.choices(self.observation_range, weights=action_probabilities)[0]
        #next_state_index = np.random.choice(self.observation_space.n, p=action_probabilities)
        next_pos = self.index_to_state(next_state_index)
        # TODO: This could generate problems with a small environment
        if self.goggles and next_pos[1] > self.grid_size + self.length_corridor:
            next_pos = self.goggles_pos
        #print("Taking action %d from cur_pos index:%d (%d, %d): going to index %d which is state => (%d, %d)" %(action, current_state, self.current_pos[0], self.current_pos[1], next_state_index, next_pos[0], next_pos[1]))
        return next_pos
        
    def step(self, action):
        #print("Time-step: %d" %self.steps_taken)
        if self.done:
            raise ValueError("Episode has already ended.")
        new_pos = self.sample_next_state(action)
        reward = -0.1  # default reward for moving
        if new_pos == self.goal:
            reward = 1.0
            self.done = True
        if new_pos == self.goggles_pos and self.goggles == True:
            self.goggles_on = True
        self.current_pos = new_pos
        self.steps_taken += 1
        if self.time_horizon != -1 and self.steps_taken >= self.time_horizon:
            self.done = True
        info = self._get_info
        return self.state_to_index(self.current_pos), reward, self.done, False, info
    
    def reset(self, seed=None, options=None):
        if self.randomize == 1:
            self.current_pos = self.index_to_state(np.random.randint(0, self.observation_space.n))
        else:
            self.current_pos = (0, 0)
        self.goggles_on = False
        self.done = False
        self.steps_taken = 0
        info = {}
        return self.state_to_index(self.current_pos), info

    def render(self, mode='human'):
        if mode == 'human':
            for j in range(self.grid_size):
                for i in range(self.grid_size):
                    if (i, j) == self.current_pos:
                        print("X ", end='')
                    elif (i, j) == self.goal:
                        print("G ", end='')
                    else:
                        print("_ ", end='')
                print()
            print()

class GridworldEnvGoalless(GridworldEnv):
    '''
    This extension of the environment work functionally like the previous, but it removes the ending goal state.
    '''
    def step(self, action):
        #print("Time-step: %d" %self.steps_taken)
        if self.done:
            raise ValueError("Episode has already ended.")
        new_pos = self.sample_next_state(action)
        reward = -0.1  # default reward for moving
        if new_pos == self.goal:
            reward = 1.0
        if new_pos == self.goggles_pos and self.goggles == True:
            self.goggles_on = True
        if new_pos == self.goggles_pos and self.goggles == True:
            self.goggles_on = True
        self.current_pos = new_pos
        self.steps_taken += 1
        if self.time_horizon != -1 and self.steps_taken >= self.time_horizon:
            self.done = True
        info = {}
        return self.state_to_index(self.current_pos), reward, self.done, False, info


## POMDP ##


class GridworldPOMDPEnv(GridworldEnv):
    '''
    This class implements the extension of the upper Gridworld MDP, making it become a POMDP by inserting the observation mechanism.
    The observations here are modeled as a probability of returning the real state of the MDP, in a Gaussian fashion based on the distance of the observed state from the real current state.
    
    Args:
     - grid_size: the size in height and length of the grid, N of the NxN matrix.
     - time_horizon: the maximum number of time steps the agent can take to get to the goal. If set to -1 the time horizon is ∞.
     - prob: the probability with which the environment takes the chosen action. If set to 0 the actions taken by the agent are deterministic.
     - steepness: a parameter to control the steepness of the Gaussian distribution that models the observation probability from the real state. A higher value makes it steeper.
     - uniform: a variable that must be between 0 and 1 that, when greater than 0, makes the observation matrix be uniformly distributed but in the true state, where it has probability
       1-uniform
    '''
    def __init__(self, grid_size=5, time_horizon=-1, prob=0.1, randomize=0, length_corridor=0, goggles=False, steepness=15, uniform=0):
        # Initialize the underlying Gridworld MDP
        super().__init__(grid_size=grid_size, time_horizon=time_horizon, length_corridor=length_corridor, goggles=goggles, randomize=randomize, prob=prob)
        # Initialize all the POMDP specific variables
        self.steepness = steepness
        self.uniform = uniform
        self.observation_matrix = self.build_observation_matrix()
        
    def build_observation_matrix(self):
        '''
        This method creates the observation matrix for our environment.
        The observation function indexes are to be used in order:
         - first param: true state s
         - second param: observation o
        '''
        # Initialize the observation function with zeros
        observation_matrix = np.zeros((self.observation_space.n, self.observation_space.n))
        # Calculate the variance of the Gaussian distribution based on the grid size
        variance = (self.grid_size // 2) ** 2

        if self.uniform > 0:
            prob = 1 - self.uniform
            prob_oth = self.uniform / (self.observation_space.n - 1)
            for s in self.observation_range:
                # Assign probability based on Gaussian distribution with adjusted steepness
                observation_matrix[s, :] = prob_oth
                observation_matrix[s, s] = prob
        else:
            for s_i in range(self.n_states):
                for s in range(self.n_states):
                    # Calculate the distance between the observed position and the true state
                    distance = np.linalg.norm(np.array(self.index_to_state(s_i)) - np.array(self.index_to_state(s)))
                    # Assign probability based on Gaussian distribution with adjusted steepness
                    observation_matrix[s_i][s] += np.exp(-self.steepness * distance**2 / (2 * variance))
                # Normalize the probabilities for each row
                observation_matrix[s_i] /= np.sum(observation_matrix[s_i])
        # Goggles variables
        if self.goggles == True:
            observation_matrix[int(self.observation_space.n/2):, int(self.observation_space.n/2):] = np.eye(int(self.observation_space.n/2))
        # Return the built matrix
        return observation_matrix

    def reset(self, seed=None, options=None):
        # Call the upper reset of the environment
        super().reset()
        info = self._get_info()
        return self.state_to_index(self.current_pos), info

    def step(self, action):
        # Save the true state
        true_state = self._get_info()
        # Make the step of the underlying MDP
        next_state, reward, done, info = super().step(action)
        # Get the observation probabilities for the state
        obs_prob = self.observation_matrix[next_state]
        # Sample the next observation from the probabilities
        obs = random.choices(self.observation_range, weights=obs_prob)[0]
        #obs = np.random.choice(self.grid_size**2, p=obs_prob)
        return obs, reward, done, False, true_state
    
    def _get_info(self):
        return {"true_state": self.current_pos - self.n_states * self.goggles_on}

class GridworldPOMDPEnvGoalless(GridworldEnvGoalless):
    '''
    This extension of the environment work functionally like the previous, but it removes the ending goal state.
    In the POMDP case it also adds the observation mechanism.
    '''
    def __init__(self, grid_size=5, time_horizon=-1, prob=0.1, randomize=0, length_corridor=0, goggles=False, steepness=15, uniform=0):
        # Initialize the underlying Gridworld MDP
        super().__init__(grid_size=grid_size, time_horizon=time_horizon, length_corridor=length_corridor, goggles=goggles, prob=prob, randomize=randomize)
        # Initialize all the POMDP specific variables
        self.steepness = steepness
        self.uniform = uniform
        self.observation_matrix = self.build_observation_matrix()

    def build_observation_matrix(self):
        '''
        This method creates the observation matrix for our environment.
        The observation function indexes are to be used in order:
         - first param: true state s
         - second param: observation o
        '''
        # Initialize the observation function with zeros
        observation_matrix = np.zeros((self.observation_space.n, self.observation_space.n))
        # Calculate the variance of the Gaussian distribution based on the grid size
        variance = (self.grid_size // 2) ** 2

        if self.uniform > 0:
            prob = 1 - self.uniform
            prob_oth = self.uniform / (self.observation_space.n - 1)
            for s in self.observation_range:
                # Assign probability based on Gaussian distribution with adjusted steepness
                observation_matrix[s, :] = prob_oth
                observation_matrix[s, s] = prob
        else:
            for s_i in range(self.n_states):
                for s in range(self.n_states):
                    # Calculate the distance between the observed position and the true state
                    distance = np.linalg.norm(np.array(self.index_to_state(s_i)) - np.array(self.index_to_state(s)))
                    # Assign probability based on Gaussian distribution with adjusted steepness
                    observation_matrix[s_i][s] += np.exp(-self.steepness * distance**2 / (2 * variance))
                # Normalize the probabilities for each row
                observation_matrix[s_i] /= np.sum(observation_matrix[s_i])
        # Goggles variables
        if self.goggles == True:
            observation_matrix[self.n_states:, self.n_states:] = np.eye(self.n_states)
        # Return the built matrix
        return observation_matrix

    def reset(self, seed=None, options=None):
        # Call the upper reset of the environment
        super().reset()
        info = self._get_info()
        return self.state_to_index(self.current_pos), info

    def step(self, action):
        # Save the true state
        true_state = self._get_info()
        # Make the step of the underlying MDP
        next_state, reward, done, _, info = super().step(action)
        # Get the observation probabilities for the state
        obs_prob = self.observation_matrix[next_state]
        # Sample the next observation from the probabilities
        obs = random.choices(self.observation_range, weights=obs_prob)[0]
        #obs = np.random.choice(self.grid_size**2, p=obs_prob)
        return obs, reward, done, False, true_state
    
    def _get_info(self):
        return {"true_state": self.state_to_index(self.current_pos) - self.n_states * self.goggles_on}


## BiModal ##


class GridworldPOMDPEnvBiModal(GridworldEnvGoalless):
    '''
    This class implements the extension of the upper Gridworld MDP, making it become a POMDP by inserting the observation mechanism.
    The observations here are modeled as a probability of returning the real state of the MDP, in a Gaussian fashion based on the distance of the observed state from the real current state.
    Moreover in this instance the observatino will be shifted in order to be bimodal with respect to the true state.

    Args:
     - grid_size: the size in height and length of the grid, N of the NxN matrix.
     - time_horizon: the maximum number of time steps the agent can take to get to the goal. If set to -1 the time horizon is ∞.
     - prob: the probability with which the environment takes the chosen action. If set to 0 the actions taken by the agent are deterministic.
     - steepness: a parameter to control the steepness of the Gaussian distribution that models the observation probability from the real state. A higher value makes it steeper.
     - uniform: a variable that must be between 0 and 1 that, when greater than 0, makes the observation matrix be uniformly distributed but in the true state, where it has probability
       1-uniform
     - shift_amount: a number that represents how much to shift the gaussian from the mean.
    '''
    def __init__(self, grid_size=5, time_horizon=-1, prob=0.1, randomize=0, length_corridor=0, goggles=False, steepness=15, uniform=0, shift_amount=0.80):
        # Initialize the underlying Gridworld MDP
        super().__init__(grid_size=grid_size, time_horizon=time_horizon, length_corridor=length_corridor, goggles=goggles, randomize=randomize, prob=prob)
        # Initialize all the POMDP specific variables
        self.shift_amount = shift_amount
        self.steepness = steepness
        self.uniform = uniform
        self.observation_matrix = self.build_observation_matrix()

    def build_observation_matrix(self):
        '''
        This method creates the observation matrix for our environment.
        The observation function indexes are to be used in order:
         - first param: true state s
         - second param: observation o
        '''
        # Initialize the observation function with zeros
        observation_matrix = np.zeros((self.observation_space.n, self.observation_space.n))
        # Calculate the variance of the Gaussian distribution based on the grid size
        variance = (self.grid_size // 2) ** 2

        if self.uniform > 0:
            prob = 1 - self.uniform
            prob_oth = self.uniform / (self.observation_space.n - 1)
            for s in self.observation_range:
                # Assign probability based on Gaussian distribution with adjusted steepness
                observation_matrix[s, :] = prob_oth
                observation_matrix[s, s] = prob
        else:
            for s_i in self.observation_range:
                for s in self.observation_range:
                    # Calculate the distance between the observed position and the true state
                    distance = np.linalg.norm(np.array(self.index_to_state(s_i)) - np.array(self.index_to_state(s)))
                    # Calculate the probability for the left-shifted Gaussian
                    prob_left = np.exp(-self.steepness * (distance + self.shift_amount) ** 2 / (2 * variance))
                    # Calculate the probability for the right-shifted Gaussian
                    prob_right = np.exp(-self.steepness * (distance - self.shift_amount) ** 2 / (2 * variance))
                    # Assign probability as the sum of left-shifted and right-shifted Gaussians
                    observation_matrix[s_i][s] += prob_left + prob_right
                # Normalize the probabilities for each row
                observation_matrix[s_i] /= np.sum(observation_matrix[s_i])

        # Return the built matrix
        return observation_matrix

    def reset(self, seed=None, options=None):
        # Call the upper reset of the environment
        super().reset()
        info = self._get_info()
        return self.state_to_index(self.current_pos), info

    def step(self, action):
        # Save the true state
        true_state = self._get_info()
        # Make the step of the underlying MDP
        next_state, reward, done, _, info = super().step(action)
        # Get the observation probabilities for the state
        obs_prob = self.observation_matrix[next_state]
        # Sample the next observation from the probabilities
        obs = random.choices(self.observation_range, weights=obs_prob)[0]
        #obs = np.random.choice(self.grid_size**2, p=obs_prob)
        return obs, reward, done, False, true_state
    
    def _get_info(self):
        return {"true_state": self.state_to_index(self.current_pos)}
    
class GridworldPOMDPEnvAsymm(GridworldEnvGoalless):
    '''
    This class implements the extension of the upper Gridworld MDP, making it become a POMDP by inserting the observation mechanism.
    The observations here are modeled as a probability of returning the real state of the MDP, in a Gaussian fashion based on the distance of the observed state from the real current state.
    Moreover in this instance the observatino will be shifted in order to be asymmetric with respect to the true state.

    Args:
     - grid_size: the size in height and length of the grid, N of the NxN matrix.
     - time_horizon: the maximum number of time steps the agent can take to get to the goal. If set to -1 the time horizon is ∞.
     - prob: the probability with which the environment takes the chosen action. If set to 0 the actions taken by the agent are deterministic.
     - steepness: a parameter to control the steepness of the Gaussian distribution that models the observation probability from the real state. A higher value makes it steeper.
     - uniform: a variable that must be between 0 and 1 that, when greater than 0, makes the observation matrix be uniformly distributed but in the true state, where it has probability
       1-uniform
     - shift_amount: a number that represents how much to shift the gaussian from the mean.
    '''
    def __init__(self, grid_size=5, time_horizon=-1, prob=0.1, randomize=0, length_corridor=0, goggles=False, steepness=15, uniform=0, shift_amount=0.1):
        # Initialize the underlying Gridworld MDP
        super().__init__(grid_size=grid_size, time_horizon=time_horizon, length_corridor=length_corridor, goggles=goggles, randomize=randomize, prob=prob)
        # Initialize all the POMDP specific variables
        self.shift_amount = shift_amount
        self.observation_space = spaces.Discrete(self.grid_size ** 2 + self.length_corridor)
        self.steepness = steepness
        self.uniform = uniform
        self.observation_matrix = self.build_observation_matrix()

    def build_observation_matrix(self):
        '''
        This method creates the observation matrix for our environment.
        The observation function indexes are to be used in order:
         - first param: true state s
         - second param: observation o
        '''
        # Initialize the observation function with zeros
        observation_matrix = np.zeros((self.observation_space.n, self.observation_space.n))
        # Calculate the variance of the Gaussian distribution based on the grid size
        variance = (self.grid_size // 2) ** 2

        if self.uniform > 0:
            prob = 1 - self.uniform
            prob_oth = self.uniform / (self.observation_space.n - 1)
            for s in self.observation_range:
                # Assign probability based on Gaussian distribution with adjusted steepness
                observation_matrix[s, :] = prob_oth
                observation_matrix[s, s] = prob
        else:
            for s_i in self.observation_range:
                for s in self.observation_range:
                    # Calculate the distance between the observed position and the true state
                    distance = np.linalg.norm(np.array(self.index_to_state(s_i)) - np.array(self.index_to_state(s)))
                    # Calculate the probability for the left-shifted Gaussian
                    probability = np.exp(-self.steepness * (distance - self.shift_amount) ** 2 / (2 * variance))
                    # Introduce an asymmetric bias
                    if self.index_to_state(s_i)[0] < self.index_to_state(s)[0]:
                        # Increase probability for positions to the right of the true state
                        probability *= 1.6
                    elif self.index_to_state(s_i)[0] > self.index_to_state(s)[0]:
                        # Decrease probability for positions to the left of the true state
                        probability *= 0.4
                    observation_matrix[s_i][s] += probability
                # Normalize the probabilities for each row
                observation_matrix[s_i] /= np.sum(observation_matrix[s_i])

        # Return the built matrix
        return observation_matrix

    def reset(self, seed=None, options=None):
        # Call the upper reset of the environment
        super().reset()
        info = self._get_info()
        return self.state_to_index(self.current_pos), info

    def step(self, action):
        # Save the true state
        true_state = self._get_info()
        # Make the step of the underlying MDP
        next_state, reward, done, _, info = super().step(action)
        # Get the observation probabilities for the state
        obs_prob = self.observation_matrix[next_state]
        # Sample the next observation from the probabilities
        obs = random.choices(self.observation_range, weights=obs_prob)[0]
        #obs = np.random.choice(self.grid_size**2, p=obs_prob)
        return obs, reward, done, False, true_state
    
    def _get_info(self):
        return {"true_state": self.state_to_index(self.current_pos)}

## SPECIAL ENVIRONMENTS

