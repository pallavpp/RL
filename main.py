import numpy as np
from environment import *
from simulations import *

M = 2						# number of players
deterministic_env = False	# rewards observed is true mu or sampled
randomization = True		# added randomization
save_history = False		# save simulation history too
skip_existing_sim = True	# skip if done already

# time steps for each K
dict_K_T = {
	2 : 1000
}

# key: number of arms (K), value: true mu values
dict_true_mu = {
	2 : [[0.1, 0.9]], 
    # 5: [np.linspace(0.1, 0.9, 5)]
}

# metrics to plot
metrics_list = [
    "regret", 
    # "mu_hat", 
    # "collissions", 
    # "reward_hist", 
    # "arm_hist", 
    # "pulls_hist", 
    # "collision_hist", 
    # "mu_hat_hist"
]

# list of all combinations of environment
env_list = compute_env_list(M, dict_K_T, dict_true_mu, deterministic_env)

# run all simulations
run_all_simulations(env_list, randomization, save_history, metrics_list, skip_existing_sim)
