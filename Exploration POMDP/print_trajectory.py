import os
import pickle
import utility

# Name of directories to print the trajectory for
exp_name = ['StochasticMatrix4RoomsCorridorG01T55Goggles']   
# Name of the objectives of which we want to print the trajectory for
sub_exp_name = ['BeliefReg']                           
# Seed number to pick the agent
agent_num = 0                  
                     
utility.print_trajectory_from_agent(exp_name, sub_exp_name, agent_num)