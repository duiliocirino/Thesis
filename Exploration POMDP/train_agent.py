import numpy as np
from tqdm import tqdm
import environments
import agents

def train_agent(args):
    run_idx, n_traj, n_episodes = args
    env = environments.GridworldPOMDPEnvGoalless(time_horizon=50, steepness=15, prob=0, randomize=1)  # Create a new environment for each run
    agent = agents.REINFORCEAgentEPOMDP(env, alpha=0.15)
    avg_entropies = []
    avg_true_entropies = []

    with tqdm(total=n_episodes, position=run_idx, ncols=80, desc=f'Run {run_idx}') as pbar:
        for i in range(n_episodes):
            episodes, true_entropies = agent.play(env=env, n_traj=n_traj)
            entropies = agent.update_multiple_sampling(episodes)
            avg_entropies.append(np.mean(entropies))
            avg_true_entropies.append(np.mean(true_entropies))
            pbar.update(1)

    return avg_entropies, avg_true_entropies