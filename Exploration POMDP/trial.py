import environments

env = environments.GridworldPOMDPEnvGoalless(4, 10, 0, 0, 0, False, None, False, True, 10, 0)

pos = (0,1)

print(env.state_to_index(pos))