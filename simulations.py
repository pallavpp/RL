from algorithm import *

def run_all_simulations(env_list, randomization, save_history, skip_existing_sim):
	for env in env_list:
		algo = Selfish_KL_UCB(env, randomization, save_history)
		algo.run()
