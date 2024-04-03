import gym
from gym import spaces
import random
import math
import numpy as np
from scipy.stats import t, mode
from utility import get_distances_matrix

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
    - randomize: a variable that if at 1 sets a random initial position, at 0 makes the initial position the top left corner;
    - length_corridor: the length of a corridor placed on the right bottom corner of the environment;
    - goggles: a boolean value that states whether to give the environment a pair of glasses that the agent receives by going on a state, which
        improves the quality of the observations;
    - goggles_corridor_shift: a boolean that if True shifts the initial position to the center right of the grid and gives the agent the 
        possibility to enter the corridor to take the glasses.
    - multi_room: a boolean that if True turns the environment into the four room environment.
    """
    
    def __init__(self, grid_size=5, time_horizon=-1, prob=0.1, randomize=0, length_corridor=0, goggles=False, goggles_pos=None, goggles_corridor_shift=False, multi_room=False):
        ## Environment defining variables
        self.grid_size = grid_size
        self.length_corridor = length_corridor
        self.goggles = goggles
        self.goggles_corridor_shift = goggles_corridor_shift
        self.multi_room = multi_room
        self.observation_space = spaces.Discrete((self.grid_size ** 2 + self.length_corridor)*(1 + self.goggles))
        self.action_space = spaces.Discrete(4)
        self.reward_range = (-0.1, 1.0)
        self.observation_range = np.array(range(self.observation_space.n))
        self.goal = (grid_size-1, grid_size-1)
        if goggles_pos != None:
            if goggles_pos is tuple:
                self.goggles_pos = goggles_pos
            elif goggles_pos is int:
                self.goggles_pos = self.index_to_state(goggles_pos)
            else:
                raise("Goggles pos must be either None, int or tuple type")
        else:
            self.goggles_pos = (0, grid_size - 1)
        if goggles_corridor_shift:
            self.goggles_pos = (grid_size-1, grid_size + length_corridor - 1)
        self.n_states = int(self.observation_space.n / 2) if goggles else self.observation_space.n
        self.goggles_on = False
        self.done = False
        self.time_horizon = time_horizon
        self.steps_taken = 0
        self.prob = prob
        self.randomize = randomize
        if self.randomize == 1:
            self.current_pos = self.index_to_state(np.random.randint(0, self.n_states))
        else:
            self.current_pos = (0, 0)
        if(self.goggles_corridor_shift):
            self.current_pos = (grid_size-1, grid_size//2)
        
        self.transition_matrix = self._build_transition_matrix()
        
    def _build_transition_matrix(self):
        '''
        This method builds the transition matrix for the MDP, T(s'|a, s).
        The function should be used with the following order of operands:
         - first parameter: s, the state where the agent takes the action from
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
            if self.multi_room and state[1] == self.grid_size / 2 and (state[0] in range(1, self.grid_size - 1)): # handle multi room
                new_pos = state
            elif state[1] >= self.grid_size: # if you are in the corridor you just go one y less
                new_pos = (state[0], state[1] - 1)
            else:
                new_pos = (state[0], max(0, state[1]-1))
        elif action == 1:  # down
            if self.multi_room and  state[1] == self.grid_size / 2 - 1 and (state[0] in range(1, self.grid_size - 1)): # handle multi room
                new_pos = state
            elif state == (self.grid_size - 1, self.grid_size - 1) or state[1] >= self.grid_size: # if you are in the corridor, or you are entering the corridor, you add one unless you are at the end of the corridor
                new_pos = (state[0], min(self.grid_size + self.length_corridor - 1, state[1] + 1))
            else:
                new_pos = (state[0], min(self.grid_size-1, state[1]+1))
        elif action == 2:  # left
            if self.multi_room and state[0] == self.grid_size / 2 and (state[1] in range(1, self.grid_size - 1)):
                new_pos = state
            elif state[1] >= self.grid_size: # if you are in the corridor you don't move
                new_pos = state
            else:
                new_pos = (max(0, state[0]-1), state[1])
        elif action == 3:  # right
            if self.multi_room and state[0] == self.grid_size / 2 - 1 and (state[1] in range(1, self.grid_size - 1)):
                new_pos = state
            elif state[1] >= self.grid_size: # if you are in the corridor you don't move
                new_pos = state
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
        #next_state_index = random.choices(self.observation_range, weights=action_probabilities)[0]
        next_state_index = np.random.choice(self.observation_space.n, p=action_probabilities)
        next_pos = self.index_to_state(next_state_index)
        # TODO: This could generate problems with a small environment
        if self.goggles and next_pos[1] > self.grid_size + self.length_corridor:
            next_pos = self.goggles_pos
        # print("Taking action %d from cur_pos index:%d (%d, %d): going to index %d which is state => (%d, %d)" %(action, current_state, self.current_pos[0], self.current_pos[1], next_state_index, next_pos[0], next_pos[1]))
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
        self.goggles_on = False
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        #np.random.seed(seed=seed)
        if self.randomize == 1:
            self.current_pos = self.index_to_state(np.random.randint(0, self.n_states))
        else:
            self.current_pos = (0, 0)
        if(self.goggles_corridor_shift):
            self.current_pos = (self.grid_size-1, self.grid_size//2)
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
    def __init__(self, grid_size=5, time_horizon=-1, prob=0.1, randomize=0, length_corridor=0, goggles=False, goggles_pos=None, goggles_corridor_shift=False, multi_room=False, steepness=15, uniform=0):
        # Initialize the underlying Gridworld MDP
        super().__init__(grid_size=grid_size, time_horizon=time_horizon, length_corridor=length_corridor, goggles=goggles, goggles_pos=goggles_pos, goggle_corridor_shift=goggles_corridor_shift, multi_room=multi_room, randomize=randomize, prob=prob)
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
        variance = 1 # To control the steepness directly thanks to the steepness parameter
        #variance = (self.grid_size // 2) ** 2

        # Handle uniform distribution case
        if self.uniform > 0:
            prob = 1 - self.uniform
            prob_oth = self.uniform / (self.observation_space.n - 1)
            for s in range(self.n_states):
                # Handle goggles
                if self.goggles and self.index_to_state == self.goggles_pos:
                    observation_matrix[s, :] = (1 - 0.99)/(self.observation_space.n - 1)
                    observation_matrix[s, s] = 0.99
                else:
                    observation_matrix[s, :] = prob_oth
                    observation_matrix[s, s] = prob
        # Normal distribution case
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
            observation_matrix[self.n_states:, self.n_states:] = (1 - 0.99)/(self.n_states - 1)
            for s in range(self.n_states, self.n_states*2):
                observation_matrix[s, s] = 0.99 
        # Return the built matrix
        return observation_matrix

    def reset(self, seed=None, options=None):
        # Call the upper reset of the environment
        super().reset()
        info = self._get_info()
        # Get the observation probabilities for the state
        obs_prob = self.observation_matrix[self.state_to_index(self.current_pos)]
        # Sample the next observation from the probabilities
        #obs = random.choices(self.observation_range, weights=obs_prob)[0]
        obs = np.random.choice(self.observation_space.n, p=obs_prob)
        return obs, info

    def step(self, action):
        # Make the step of the underlying MDP
        next_state, reward, done, info = super().step(action)
        # Save the true state
        true_state = self._get_info()
        # Get the observation probabilities for the state
        obs_prob = self.observation_matrix[next_state]
        # Sample the next observation from the probabilities
        #obs = random.choices(self.observation_range, weights=obs_prob)[0]
        obs = np.random.choice(self.observation_space.n, p=obs_prob)
        return obs, reward, done, False, true_state
    
    def _get_info(self):
        return {"true_state": self.state_to_index(self.current_pos)}

class GridworldPOMDPEnvGoalless(GridworldEnvGoalless):
    '''
    This extension of the environment work functionally like the previous, but it removes the ending goal state.
    In the POMDP case it also adds the observation mechanism.
    '''
    def __init__(self, grid_size=5, time_horizon=-1, prob=0.1, randomize=0, length_corridor=0, goggles=False, goggles_pos=None, goggles_corridor_shift=False, multi_room=False, steepness=15, manhattan_obs=True, uniform=0):
        # Initialize the underlying Gridworld MDP
        super().__init__(grid_size=grid_size, time_horizon=time_horizon, length_corridor=length_corridor, goggles=goggles, goggles_pos=goggles_pos, goggles_corridor_shift=goggles_corridor_shift, multi_room=multi_room, prob=prob, randomize=randomize)
        # Initialize all the POMDP specific variables
        self.steepness = steepness
        self.uniform = uniform
        self.manhattan_obs = manhattan_obs
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
        variance = 1 # To control the steepness directly thanks to the steepness parameter
        #variance = (self.grid_size // 2) ** 2

        # Handle uniform distribution case
        if self.uniform > 0:
            prob = 1 - self.uniform
            prob_oth = self.uniform / (self.observation_space.n - 1)
            for s in range(self.n_states):
                # Handle goggles
                if self.goggles and self.index_to_state == self.goggles_pos:
                    observation_matrix[s, :] = (1 - 0.99)/(self.observation_space.n - 1)
                    observation_matrix[s, s] = 0.99
                else:
                    observation_matrix[s, :] = prob_oth
                    observation_matrix[s, s] = prob
        # Normal distribution case
        else:
            if self.manhattan_obs:
                distances = get_distances_matrix(self)
                for s_i in range(self.n_states):
                    state = self.index_to_state(s_i)
                    variance = 1 / (self.steepness * self.world_matrix[state[1]][state[0]])
                    for s in range(self.n_states):
                        # Calculate the distance between the observed position and the true state
                        distance = distances[s][s_i]
                        # Assign probability based on Gaussian distribution with adjusted steepness. State used for allowing higher cell visibility
                        observation_matrix[s_i][s] += np.exp(-distance**2 / (2 * variance))
                    # Normalize the probabilities for each row
                    observation_matrix[s_i] /= np.sum(observation_matrix[s_i])
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
            observation_matrix[self.n_states:, self.n_states:] = (1 - 0.99)/(self.n_states - 1)
            for s in range(self.n_states, self.n_states*2):
                observation_matrix[s, s] = 0.99 
        # Return the built matrix
        return observation_matrix

    def reset(self, seed=None, options=None):
        # Call the upper reset of the environment
        super().reset()
        info = self._get_info()
        # Get the observation probabilities for the state
        obs_prob = self.observation_matrix[self.state_to_index(self.current_pos)]
        # Sample the next observation from the probabilities
        #obs = random.choices(self.observation_range, weights=obs_prob)[0]
        obs = np.random.choice(self.observation_space.n, p=obs_prob)
        return obs, info

    def step(self, action):
        # Make the step of the underlying MDP
        next_state, reward, done, _, info = super().step(action)
        # Save the true state
        true_state = self._get_info()
        # Get the observation probabilities for the state
        obs_prob = self.observation_matrix[next_state]
        # Sample the next observation from the probabilities
        #obs = random.choices(self.observation_range, weights=obs_prob)[0]
        obs = np.random.choice(self.observation_space.n, p=obs_prob)
        return obs, reward, done, False, true_state
    
    def _get_info(self):
        return {"true_state": self.state_to_index(self.current_pos)}


## BiModal ##

# To update
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
    def __init__(self, grid_size=5, time_horizon=-1, prob=0.1, randomize=0, length_corridor=0, goggles=False, goggles_pos=None, goggles_corridor_shift=False, multi_room=False, steepness=15, uniform=0, shift_amount=0.80):
        # Initialize the underlying Gridworld MDP
        super().__init__(grid_size=grid_size, time_horizon=time_horizon, length_corridor=length_corridor, goggles=goggles, goggles_pos=goggles_pos, goggles_corridor_shift=goggles_corridor_shift, multi_room=multi_room, randomize=randomize, prob=prob)
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
        variance = 1 # To control the steepness directly thanks to the steepness parameter
        #variance = (self.grid_size // 2) ** 2

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
        # Get the observation probabilities for the state
        obs_prob = self.observation_matrix[self.state_to_index(self.current_pos)]
        # Sample the next observation from the probabilities
        #obs = random.choices(self.observation_range, weights=obs_prob)[0]
        obs = np.random.choice(self.observation_space.n, p=obs_prob)
        return obs, info

    def step(self, action):
        # Make the step of the underlying MDP
        next_state, reward, done, _, info = super().step(action)
        # Save the true state
        true_state = self._get_info()
        # Get the observation probabilities for the state
        obs_prob = self.observation_matrix[next_state]
        # Sample the next observation from the probabilities
        #obs = random.choices(self.observation_range, weights=obs_prob)[0]
        obs = np.random.choice(self.observation_space.n, p=obs_prob)
        return obs, reward, done, False, true_state
    
    def _get_info(self):
        return {"true_state": self.state_to_index(self.current_pos)}
# To update   
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
    def __init__(self, grid_size=5, time_horizon=-1, prob=0.1, randomize=0, length_corridor=0, goggles=False, goggles_pos=None, goggles_corridor_shift=False, multi_room=False, steepness=15, uniform=0, shift_amount=0.1):
        # Initialize the underlying Gridworld MDP
        super().__init__(grid_size=grid_size, time_horizon=time_horizon, length_corridor=length_corridor, goggles=goggles, goggles_pos=goggles_pos, goggles_corridor_shift=goggles_corridor_shift, multi_room=multi_room, randomize=randomize, prob=prob)
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
        # Get the observation probabilities for the state
        obs_prob = self.observation_matrix[self.state_to_index(self.current_pos)]
        # Sample the next observation from the probabilities
        #obs = random.choices(self.observation_range, weights=obs_prob)[0]
        obs = np.random.choice(self.observation_space.n, p=obs_prob)
        return obs, info

    def step(self, action):
        # Make the step of the underlying MDP
        next_state, reward, done, _, info = super().step(action)
        # Save the true state
        true_state = self._get_info()
        # Get the observation probabilities for the state
        obs_prob = self.observation_matrix[next_state]
        # Sample the next observation from the probabilities
        #obs = random.choices(self.observation_range, weights=obs_prob)[0]
        obs = np.random.choice(self.observation_space.n, p=obs_prob)
        return obs, reward, done, False, true_state
    
    def _get_info(self):
        return {"true_state": self.state_to_index(self.current_pos)}

class GridworldPOMDPEnvGoalless4Obs(GridworldEnvGoalless):
    '''
    This extension of the environment work functionally like the previous, but it removes the ending goal state.
    In the POMDP case it also adds the observation mechanism.
    '''
    def __init__(self, grid_size=6, time_horizon=-1, prob=0.1, randomize=0, length_corridor=0, goggles=False, goggles_pos=None, 
                 goggles_corridor_shift=False, multi_room=True, steepness=10, manhattan_obs=False, uniform=0):
        # Initialize the underlying Gridworld MDP
        super().__init__(grid_size=grid_size, time_horizon=time_horizon, length_corridor=length_corridor, goggles=goggles, goggles_pos=goggles_pos, 
                         goggles_corridor_shift=goggles_corridor_shift, multi_room=multi_room, prob=prob, randomize=randomize)
        # Initialize all the POMDP specific variables
        self.steepness = steepness
        self.uniform = uniform
        self.observation_space = spaces.Discrete(4)
        self.observation_matrix = self.build_observation_matrix()

    def build_observation_matrix(self):
        '''
        This method creates the observation matrix for our environment.
        The observation function indexes are to be used in order:
         - first param: true state s
         - second param: observation o
        '''
        observation_matrix = np.zeros((self.n_states, self.observation_space.n))
        for s in range(self.n_states):
            if s < self.grid_size ** 2 / 2:
                if s % self.grid_size < self.grid_size / 2:
                    observation_matrix[s, 0] = 1
                else: 
                    observation_matrix[s, 1] = 1
            else:
                if s % self.grid_size < self.grid_size / 2:
                    observation_matrix[s, 2] = 1
                else: 
                    observation_matrix[s, 3] = 1
        return observation_matrix
    
    def sample_next_state(self, action):
        current_state = self.state_to_index(self.current_pos)
        action_probabilities = self.transition_matrix[:, current_state, action]
        #next_state_index = random.choices(self.observation_range, weights=action_probabilities)[0]
        next_state_index = np.random.choice(self.n_states, p=action_probabilities)
        next_pos = self.index_to_state(next_state_index)
        # TODO: This could generate problems with a small environment
        if self.goggles and next_pos[1] > self.grid_size + self.length_corridor:
            next_pos = self.goggles_pos
        # print("Taking action %d from cur_pos index:%d (%d, %d): going to index %d which is state => (%d, %d)" %(action, current_state, self.current_pos[0], self.current_pos[1], next_state_index, next_pos[0], next_pos[1]))
        return next_pos

    def reset(self, seed=None, options=None):
        # Call the upper reset of the environment
        super().reset()
        info = self._get_info()
        # Get the observation probabilities for the state
        obs_prob = self.observation_matrix[self.state_to_index(self.current_pos)]
        # Sample the next observation from the probabilities
        #obs = random.choices(self.observation_range, weights=obs_prob)[0]
        obs = np.random.choice(self.observation_space.n, p=obs_prob)
        return obs, info

    def step(self, action):
        # Make the step of the underlying MDP
        next_state, reward, done, _, info = super().step(action)
        # Save the true state
        true_state = self._get_info()
        # Get the observation probabilities for the state
        obs_prob = self.observation_matrix[next_state]
        # Sample the next observation from the probabilities
        #obs = random.choices(self.observation_range, weights=obs_prob)[0]
        obs = np.random.choice(self.observation_space.n, p=obs_prob)
        return obs, reward, done, False, true_state
    
    def _get_info(self):
        return {"true_state": self.state_to_index(self.current_pos)}
    

class GridworldPOMDPEnvGoalless2Obs(GridworldEnvGoalless):
    '''
    This extension of the environment work functionally like the previous, but it removes the ending goal state.
    In the POMDP case it also adds the observation mechanism.
    '''
    def __init__(self, grid_size=6, time_horizon=-1, prob=0.1, randomize=0, length_corridor=0, goggles=False, goggles_pos=None, 
                 goggles_corridor_shift=False, multi_room=True, steepness=10, manhattan_obs=False, uniform=0):
        # Initialize the underlying Gridworld MDP
        super().__init__(grid_size=grid_size, time_horizon=time_horizon, length_corridor=length_corridor, goggles=goggles, goggles_pos=goggles_pos, 
                         goggles_corridor_shift=goggles_corridor_shift, multi_room=multi_room, prob=prob, randomize=randomize)
        # Initialize all the POMDP specific variables
        self.steepness = steepness
        self.uniform = uniform
        self.observation_space = spaces.Discrete(4)
        self.observation_matrix = self.build_observation_matrix()

    def build_observation_matrix(self):
        '''
        This method creates the observation matrix for our environment.
        The observation function indexes are to be used in order:
         - first param: true state s
         - second param: observation o
        '''
        observation_matrix = np.zeros((self.n_states, self.observation_space.n))
        for s in range(self.n_states):
            if s < self.grid_size ** 2:
                if s % self.grid_size < self.grid_size / 2:
                    observation_matrix[s, 0] = 1
                else: 
                    observation_matrix[s, 1] = 1
        return observation_matrix

    def sample_next_state(self, action):
        current_state = self.state_to_index(self.current_pos)
        action_probabilities = self.transition_matrix[:, current_state, action]
        #next_state_index = random.choices(self.observation_range, weights=action_probabilities)[0]
        next_state_index = np.random.choice(self.n_states, p=action_probabilities)
        next_pos = self.index_to_state(next_state_index)
        # TODO: This could generate problems with a small environment
        if self.goggles and next_pos[1] > self.grid_size + self.length_corridor:
            next_pos = self.goggles_pos
        # print("Taking action %d from cur_pos index:%d (%d, %d): going to index %d which is state => (%d, %d)" %(action, current_state, self.current_pos[0], self.current_pos[1], next_state_index, next_pos[0], next_pos[1]))
        return next_pos
    
    def reset(self, seed=None, options=None):
        # Call the upper reset of the environment
        super().reset()
        info = self._get_info()
        # Get the observation probabilities for the state
        obs_prob = self.observation_matrix[self.state_to_index(self.current_pos)]
        # Sample the next observation from the probabilities
        #obs = random.choices(self.observation_range, weights=obs_prob)[0]
        obs = np.random.choice(self.observation_space.n, p=obs_prob)
        return obs, info

    def step(self, action):
        # Make the step of the underlying MDP
        next_state, reward, done, _, info = super().step(action)
        # Save the true state
        true_state = self._get_info()
        # Get the observation probabilities for the state
        obs_prob = self.observation_matrix[next_state]
        # Sample the next observation from the probabilities
        #obs = random.choices(self.observation_range, weights=obs_prob)[0]
        obs = np.random.choice(self.observation_space.n, p=obs_prob)
        return obs, reward, done, False, true_state
    
    def _get_info(self):
        return {"true_state": self.state_to_index(self.current_pos)}



## SPECIAL ENVIRONMENTS

class MatrixGridworld(gym.Env):
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
    - grid_size: in this instance this variable is not needed but for the way we print the policy in the utils functions, although it is not used and must not be used in other cases.
    - time_horizon: the maximum number of time steps the agent can take to get to the goal. If set to -1 the time horizon is ∞.
    - prob: the probability with which the environment takes the chosen action. If set to 0 the actions taken by the agent are deterministic.
    - randomize: a variable that if at 1 sets a random initial position, at 0 makes the initial position the top left corner;
    - length_corridor: the length of a corridor placed on the right bottom corner of the environment;
    - goggles: a boolean value that states whether to give the environment a pair of glasses that the agent receives by going on a state, which
        improves the quality of the observations;
    - goggles_corridor_shift: a boolean that if True shifts the initial position to the center right of the grid and gives the agent the 
        possibility to enter the corridor to take the glasses.
    - multi_room: a boolean that if True turns the environment into the four room environment.
    """
    
    def __init__(self, grid_size=5, time_horizon=-1, prob=0.1, randomize=0, initial_position=(0, 0), length_corridor=0, goggles=False, goggles_pos=None, goggles_corridor_shift=False, world_matrix=[]):
        ## Environment defining variables
        self.grid_size = grid_size
        self.length_corridor = length_corridor
        self.goggles = goggles
        self.observation_space = spaces.Discrete(int(np.count_nonzero(world_matrix)*(1 + self.goggles)))
        self.n_states = int(self.observation_space.n / 2) if goggles else self.observation_space.n
        self.action_space = spaces.Discrete(4)
        self.reward_range = (-0.1, 1.0)
        self.observation_range = np.array(range(self.observation_space.n))
        # Define the reachable spots
        self.world_matrix = np.array(world_matrix)
        # Define the indexes from the world matrix
        self.world_indexes = self._build_indexes()
        # TODO: in case of wanting to put a real goal this should be handled like the goggles int the next lines
        self.goal = tuple(x - 1 for x in self.world_matrix.shape)
        # Set the goggles to the correct value if present
        if goggles_pos != None:
            print(goggles_pos)
            if isinstance(goggles_pos, tuple):
                self.goggles_pos = goggles_pos
            elif isinstance(goggles_pos, int):
                self.goggles_pos = self.index_to_state(goggles_pos)
            else:
                raise("Goggles pos must be either None, int or tuple type")
        else:
            self.goggles_pos = tuple(x - 1 for x in self.world_matrix.shape)
        self.goggles_on = False
        self.done = False
        self.time_horizon = time_horizon
        self.steps_taken = 0
        self.prob = prob
        self.initial_position = initial_position
        self.randomize = randomize
        self.transition_matrix = self._build_transition_matrix()

    def _build_indexes(self):
        '''
        This functions transforms the list passed as world_matrix into a numpy array for ease of operations.
        '''
        world_indexes = np.zeros_like(self.world_matrix)
        num_rows, num_columns = self.world_matrix.shape
        cur_num = 0
        for i in range(num_rows):
            for j in range(num_columns):
                if self.world_matrix[i, j] != 0:
                    world_indexes[i, j] = cur_num
                    cur_num += 1
        return world_indexes
        
    def _build_transition_matrix(self):
        '''
        This method builds the transition matrix for the MDP, T(s'|a, s).
        The function should be used with the following order of operands:
         - first parameter: s, the state where the agent takes the action from
         - second parameter: s', the state where the agent arrives taking action a from state s
         - third parameter: a, the action taken by the agent
        '''
        transition_matrix = np.zeros((self.observation_space.n, self.observation_space.n, self.action_space.n))
        # For every state (i,j)
        for index in self.observation_range:
            # Debug
            #print("Analyzing state: " + str(index))
            # Handle goggles
            if self.goggles==True and index >= self.n_states:
                self.goggles_on = True
            s = self.index_to_state(index)
            #print(f"State {s} from index {index}")
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
            new_pos = (state[0], max(0, state[1]-1))
            if self.world_matrix[new_pos[1], new_pos[0]] == 0:
                new_pos = state
        elif action == 1:  # down
            new_pos = (state[0], min(state[1]+1, self.world_matrix.shape[0] - 1))
            if self.world_matrix[new_pos[1], new_pos[0]] == 0:
                new_pos = state
        elif action == 2:  # left
            new_pos = (max(0, state[0]-1), state[1])
            if self.world_matrix[new_pos[1], new_pos[0]] == 0:
                new_pos = state
        elif action == 3:  # right
            new_pos = (min(state[0]+1, self.world_matrix.shape[1] - 1), state[1])
            if self.world_matrix[new_pos[1], new_pos[0]] == 0:
                new_pos = state
        else:
            raise ValueError("Invalid action.")
        return new_pos
    
    def _get_info(self):
        return {"coord": self.current_pos}
    
    def state_to_index(self, state):
        #print(f"Analyzing x={state[0]} and y={state[1]}")
        index = self.world_indexes[state[1], state[0]]
        #print("Converting (%d, %d) => %d" %(state[0], state[1], index))
        return int(index + self.n_states * self.goggles_on)

    def index_to_state(self, index):
        """Converts an index to a state tuple (i, j)."""
        if index >= self.n_states:
            index = index - self.n_states
        state = np.where(self.world_indexes == index)
        #print("index: " + str(index) + " => state: " + str(state))
        state = (state[1][0], state[0][0])
        #print("Converting %d => (%d, %d)" %(index, i, j))
        return state
    
    def sample_next_state(self, action):
        current_state = self.state_to_index(self.current_pos)
        action_probabilities = self.transition_matrix[:, current_state, action]
        #next_state_index = random.choices(self.observation_range, weights=action_probabilities)[0]
        next_state_index = np.random.choice(self.observation_space.n, p=action_probabilities)
        next_pos = self.index_to_state(next_state_index)
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
        info = self._get_info()
        return self.state_to_index(self.current_pos), reward, self.done, False, info
    
    def reset(self, seed=None, options=None):
        self.goggles_on = False
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        #np.random.seed(seed=seed)
        if self.randomize == 1:
            self.current_pos = self.index_to_state(np.random.randint(0, self.n_states))
        else:
            self.current_pos = self.initial_position
        self.done = False
        self.steps_taken = 0
        info = {}
        return self.state_to_index(self.current_pos), info
    
class MatrixGridworldGoalless(MatrixGridworld):
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
    
class MatrixGridworldPOMDPEnvGoalless(MatrixGridworldGoalless):
    '''
    This extension of the environment work functionally like the previous, but it removes the ending goal state.
    In the POMDP case it also adds the observation mechanism.
    '''
    def __init__(self, grid_size=5, time_horizon=-1, prob=0.1, randomize=0, initial_position=(0, 0), length_corridor=0, goggles=False, goggles_pos=None, goggles_corridor_shift=False, multi_room=False, world_matrix=[], steepness=15, manhattan_obs=True, uniform=0):
        # Initialize the underlying Gridworld MDP
        super().__init__(grid_size=grid_size, time_horizon=time_horizon, length_corridor=length_corridor, goggles=goggles, goggles_pos=goggles_pos, goggles_corridor_shift=goggles_corridor_shift, world_matrix=world_matrix, prob=prob, randomize=randomize, initial_position=initial_position)
        # Initialize all the POMDP specific variables
        self.steepness = steepness
        self.uniform = uniform
        self.manhattan_obs = manhattan_obs
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
        variance = 1

        if self.uniform > 0:
            prob = 1 - self.uniform
            prob_oth = self.uniform / (self.observation_space.n - 1)
            for s in range(self.n_states):
                if self.goggles and self.index_to_state == self.goggles_pos:
                    observation_matrix[s, :] = (1 - 0.99)/(self.observation_space.n - 1)
                    observation_matrix[s, s] = 0.99
                else:
                    observation_matrix[s, :] = prob_oth
                    observation_matrix[s, s] = prob
        else:
            if self.manhattan_obs:
                distances = get_distances_matrix(self)
                for s_i in range(self.n_states):
                    state = self.index_to_state(s_i)
                    variance = 1 / (self.steepness * self.world_matrix[state[1]][state[0]])
                    for s in range(self.n_states):
                        # Calculate the distance between the observed position and the true state
                        distance = distances[s][s_i]
                        # Assign probability based on Gaussian distribution with adjusted steepness. State used for allowing higher cell visibility
                        observation_matrix[s_i][s] += np.exp(-distance**2 / (2 * variance))
                        #print(f"Distance {self.index_to_state(s)} {self.index_to_state(s_i)}: {distance}")
                    # Normalize the probabilities for each row
                    #print(observation_matrix[s_i])
                    observation_matrix[s_i] /= np.sum(observation_matrix[s_i])
                    #print(observation_matrix[s_i])
                    #print(f"State: ({state[1]}, {state[0]}) - Variance: {variance} - Weight: {self.world_matrix[state[1]][state[0]]}")
            else:
                for s_i in range(self.n_states): # old code based on coordinates distance
                    for s in range(self.n_states):
                        # Calculate the distance between the observed position and the true state
                        distance = np.linalg.norm(np.array(self.index_to_state(s_i)) - np.array(self.index_to_state(s)))
                        # Assign probability based on Gaussian distribution with adjusted steepness
                        observation_matrix[s_i][s] += np.exp(-self.steepness * distance**2 / (2 * variance))
                    # Normalize the probabilities for each row
                    observation_matrix[s_i] /= np.sum(observation_matrix[s_i])
        # Goggles variables
        if self.goggles == True:
            observation_matrix[self.n_states:, self.n_states:] = (1 - 0.99)/(self.n_states - 1)
            for s in range(self.n_states, self.n_states*2):
                observation_matrix[s, s] = 0.99 
        # Return the built matrix
        return observation_matrix

    def reset(self, seed=None, options=None):
        # Call the upper reset of the environment
        super().reset()
        info = self._get_info()
        # Get the observation probabilities for the state
        obs_prob = self.observation_matrix[self.state_to_index(self.current_pos)]
        # Sample the next observation from the probabilities
        #obs = random.choices(self.observation_range, weights=obs_prob)[0]
        obs = np.random.choice(self.observation_space.n, p=obs_prob)
        return obs, info

    def step(self, action):
        # Make the step of the underlying MDP
        next_state, reward, done, _, info = super().step(action)
        # Save the true state
        true_state = self._get_info()
        # Get the observation probabilities for the state
        obs_prob = self.observation_matrix[next_state]
        # Sample the next observation from the probabilities
        #obs = random.choices(self.observation_range, weights=obs_prob)[0]
        obs = np.random.choice(self.observation_space.n, p=obs_prob)
        return obs, reward, done, False, true_state
    
    def _get_info(self):
        return {"true_state": self.state_to_index(self.current_pos)}


## TOY ENV ##


class ToyPOMDPEnv(gym.Env):
    def __init__(self):
        # Define the state space, observation space, and action space
        self.state_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(2)
        self.action_space = spaces.Discrete(2)
        
        # Initialize the state
        self.state = self.state_space.sample()
        
        # Transition probabilities (s, a, s')
        self.transition_matrix = np.array([[[1, 0], [0, 1]], [[1, 0], [0, 1]]])
        
        # Observation probabilities (s, o)
        self.observation_matrix = np.array([[0, 1], [1, 0]])

    def step(self, action):
        # Ensure action is within the action space
        action = self.action_space.sample() if action not in [0, 1] else action
        
        # Update the state based on the chosen action
        self.state = np.random.choice([0, 1], p=self.transition_matrix[self.state, action])
        
        # Generate observation based on the updated state
        observation = np.random.choice([0, 1], p=self.observation_matrix[self.state, action])
        
        # Define a simple reward function (you can modify this based on your task)
        reward = 1 if self.state == observation else 0
        
        # Check if the episode is done (not necessary for this example)
        done = False
        
        # Additional information (not necessary for this example)
        info = {}
        
        return observation, reward, done, info

    def reset(self):
        # Reset the environment to a random initial state
        self.state = self.state_space.sample()
        return self.state
