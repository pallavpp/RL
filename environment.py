import numpy as np
import collections

class Environment:
	def __inti__(self, config, deterministic):
		self.M = config["M"]
		self.K = config["K"]
		self.T = config["T"]
		self.true_mu = np.array(sorted(config["true_mu"], reverse=True))
		self.top_true_mu_sum = np.sum(self.true_mu[:self.M])
		self.deterministic = deterministic

		def draw(self, arms):
			arm_counts = collections.Counter(arms)
			observed_rewards = np.zeros(self.M)
			regret_t = self.top_true_mu_sum
			arm_collisions_t = np.zeros((self.M, self.K))
			for player in range(self.M):
				if arm_counts[arms[player]] > 1:
					arm_collisions_t[player][arms[player]] = 1
				else:
					if self.deterministic:
						observed_rewards[player] = self.true_mu[arms[player]]
					else:
						observed_rewards[player] = np.random.binomial(n=1, p=self.true_mu[arms[player]])
					regret_t -= self.true_mu[arms[player]]
			
			return observed_rewards, regret_t, arm_collisions_t

def compute_env_list(M, dict_K_T, dict_true_mu, deterministic):
	env_ist = []
	for K in dict_K_T.keys():
		config = {}
		config["M"] = M
		config["K"] = K
		config["T"] = dict_K_T[K]
		list_true_mu = dict_true_mu[K]

		for true_mu in list_true_mu:
			config["true_mu"] = true_mu
			env = Environment(config, deterministic=deterministic)
			env_ist.append(env)
	return env_ist
