import os
from algorithm import *
import matplotlib.pyplot as plt

def run_all_simulations(env_list, randomization, save_history, metrics_list, skip_existing_sim):
	for env in env_list:
		# result save path
		dir_name = f'M_{env.M}_K_{env.K}_T_{env.T}_mu_{env.true_mu[-1]}_{env.true_mu[0]}'
		if randomization:
			dir_path = os.path.abspath(os.path.join(os.getcwd(), f"randomized_selfish/{dir_name}"))
		else:
			dir_path = os.path.abspath(os.path.join(os.getcwd(), f"selfish/{dir_name}"))
		print(dir_path)
		if os.path.exists(dir_path):
			if skip_existing_sim:
				print(f"Simulation skipped for {dir_name}")
				continue
		else:
			os.makedirs(dir_path)

		algo = Selfish_KL_UCB(env, randomization, save_history)
		print(f"Running simulation for {dir_name}")
		algo.run()
		
		# plot metrics
		if "regret" in metrics_list:
			pass
			# plt.plot(np.linspace(0, ))
