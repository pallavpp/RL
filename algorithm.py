import time
import numpy as np
from utils import *

class Selfish_KL_UCB:
	def __init__(self, environment, randomization):
		self.env = environment
		self.randomization = randomization
		self.klucb = np.vectorize(klucb)
		self.c = 3
		self.reset()
	
	# everything to be monitored
	def reset(self):
		self.t = 0
		self.regret = np.zeros(self.env.T)
		self.total_observed_reward = np.zeros((self.env.M, self.env.K))
		self.pulls = np.zeros((self.env.M, self.env.K), dtype=np.int32)
		self.mu_hat = np.zeros((self.env.M, self.env.K))
		self.arm_history = np.zeros((self.env.M, self.env.T))
		self.collisions = np.zeros((self.env.M, self.env.K)) # self.collisions[i,j] = nb of collisions of player i on arm j
		self.collision_hist = np.zeros((self.env.K, self.env.T))

	def compute_ucb_idx(self):
		ucb_idx = np.zeros((self.env.M, self.env.K))
		d = (np.log(self.t) + self.c*np.log(np.log(self.t))) / (self.pulls+1e-7)
		ucb_idx = self.klucb(self.mu_hat, d)
		ucb_idx[self.pulls < 1] = float('+inf')
		return ucb_idx
	
	def compute_mu_hat(self):
		mu_hat = np.zeros((self.env.M, self.env.K))
		mu_hat = self.total_observed_reward / (1e-7+self.pulls)
		mu_hat[self.pulls == 0] = 0 # if arm has never been pulled, mu_hat =0
		return mu_hat
	
	def update_stats(self, arms_t, rewards, regret_t, collisions_t):
		if self.t < self.env.T:
			self.regret[self.t] = regret_t
			self.total_observed_reward[np.arange(self.env.M), arms_t] += rewards
			self.pulls[np.arange(self.env.M), arms_t] += 1
			self.mu_hat = self.compute_mu_hat()
			self.arm_history[:, self.t] = arms_t
			self.collisions += collisions_t
			self.collision_hist[:, self.t] += np.sum(collisions_t, axis=0)
	
	def run(self):
		self.reset()
		tic = time.time()
		while self.t < self.env.T:
			if self.t == 1000:
				print(f"Estimated time remaining: {round((self.env.T-1000)*(time.time()-tic)/(1000*60), 1)} min")
				print()
			ucb_idx = self.compute_ucb_idx()
			if self.randomization:
				ucb_idx +=  np.random.normal(0, 1/(self.t+1), size=(self.env.M, self.env.K))
			arms_t = choose_best_arm(ucb_idx)
			rewards_t, regret_t, collisions_t = self.env.draw(arms_t,)
			self.update_stats(arms_t=arms_t, rewards=rewards_t, regret_t=regret_t, collisions_t=collisions_t)
			self.t += 1
