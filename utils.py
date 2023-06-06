import numpy as np

def choose_best_arm(indices, tie_breaking="random"):
	if tie_breaking == "random":
		max_idx = np.max(indices, axis=1)
		M = indices.shape[0]
		chosen_arms = np.zeros(M)
		for player in range(M):
			chosen_arms[player] = np.random.choice(np.where(indices[player] == max_idx[player])[0])
	elif "lexico" in tie_breaking:
		chosen_arms = np.argmax(indices, axis=1)
	return chosen_arms.astype(int)

def kl_div_bernoulli(x, y):
	"""Kullback-Leibler divergence for Bernoulli distributions."""
	eps = 1e-10
	x = min(max(x, eps), 1-eps)
	y = min(max(y, eps), 1-eps)
	return x*np.log(x/y) + (1-x)*np.log((1-x)/(1-y))

def klucb(x, d, precision=1e-2, max_iter=50):
	l = x
	u = 1
	count = 0
	while u-l>precision and count <= max_iter:
		count += 1
		m = (l+u)/2
		if kl_div_bernoulli(x, m) > d:
			u = m
		else:
			l = m
	return (l+u)/2
