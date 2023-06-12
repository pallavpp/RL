import numpy as np
from environment import *
from simulations import *

M = 10						# number of players
n_exp = 5					# number of experiments to conduct
deterministic_env = False	# reward observed is true mu or sampled
randomization = False		# added randomization
skip_existing_sim = False	# skip if done already

# time steps for each K
dict_K_T = {
	15 : 100000
}

# key: number of arms (K), value: true mu values
dict_true_mu = {
	# 3 : [[0.1, 0.5, 0.9]], 
    15: [np.linspace(0.1, 0.2, 15), np.linspace(0.8, 0.9, 15), np.linspace(0.01, 0.99, 15)]
}

# list of all combinations of environment
env_list = compute_env_list(M, dict_K_T, dict_true_mu, deterministic_env)

# run all simulations
run_all_simulations(env_list, n_exp, randomization, skip_existing_sim)
