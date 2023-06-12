import os
from algorithm import *
import matplotlib.pyplot as plt

def run_all_simulations(env_list, n_exp, randomization, skip_existing_sim):
	for exp_no in range(n_exp):
		for env in env_list:
			# result save path
			dir_name = f'M_{env.M}_K_{env.K}_T_{env.T}_mu_{env.true_mu[-1]}_{env.true_mu[0]}'
			if randomization:
				dir_path = os.path.abspath(os.path.join(os.getcwd(), f"results/randomized_selfish/{dir_name}"))
			else:
				dir_path = os.path.abspath(os.path.join(os.getcwd(), f"results/selfish/{dir_name}"))
			if os.path.exists(dir_path):
				if skip_existing_sim:
					print(f"Simulation skipped for {dir_name}")
					print()
					continue
			else:
				os.makedirs(dir_path)

			algo = Selfish_KL_UCB(env, randomization)
			if randomization:
				print(f"Running simulation for randomized {dir_name}")
			else:
				print(f"Running simulation for selfish {dir_name}")
			print()
			algo.run()

			time_axis = [i for i in range(1, env.T+1)]

			# plot cumulative regret
			plt.plot(time_axis, np.cumsum(algo.regret))
			plt.xlabel("T")
			plt.ylabel("Cumulative Regret")
			plt.title(f"Cumulative Regret for: {dir_name}")
			plt.savefig(f"{dir_path}/cumulative_regret_exp_{exp_no}.png", bbox_inches='tight')
			plt.clf()
			print(f"Cumulative regret saved")
			print()

			# plot chosen arm history
			for i in range(env.M):
				plt.plot(time_axis, algo.arm_history[i], label=f"P{i+1}")
			plt.xlabel("T")
			plt.ylabel("Arm Chosen")
			plt.title(f"Arm History for: {dir_name}")
			plt.legend(loc=(1.04, 0.5))
			plt.savefig(f"{dir_path}/arm_history_exp_{exp_no}.png", bbox_inches='tight')
			plt.clf()
			print(f"Arm history saved")
			print()

			# plot total collisions with time
			temp = np.cumsum(algo.collision_hist, axis=1)

			plt.plot(time_axis, np.sum(temp, axis=0))
			plt.xlabel("T")
			plt.ylabel("Total Collisions Upto t")
			plt.title(f"Total Collisions Across All Arms for: {dir_name}")
			plt.savefig(f"{dir_path}/total_collisions_exp_{exp_no}.png", bbox_inches='tight')
			plt.clf()
			print(f"Total collisions across all arms saved")
			print()

			# plot total collisions with time for each arm
			for i in range(env.K):
				plt.plot(time_axis, temp[i], label=f"A{i+1}")
			plt.xlabel("T")
			plt.ylabel("Total Collisions Upto t")
			plt.title(f"Total Collisions Per Arm for: {dir_name}")
			plt.legend(loc=(1.04, 0.5))
			plt.savefig(f"{dir_path}/total_collisions_per_arm_exp_{exp_no}.png", bbox_inches='tight')
			plt.clf()
			print(f"Total collisions per arm saved")
			print()

		print("Complete")
		print()
