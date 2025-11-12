import numpy as np
import torch
import time
from causalpfn import CATEEstimator, ATEEstimator

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# 1. Generate synthetic data
np.random.seed(42)
n, d = 20000, 5
X = np.random.normal(1, 1, size=(n, d)).astype(np.float32)

# Define true causal effects
def true_cate(x):
    return np.sin(x[:, 0]) + 0.5 * x[:, 1]

def true_ate():
    return np.mean(true_cate(X))

# Generate treatment and outcomes
tau = true_cate(X).astype(np.float32)
T = np.random.binomial(1, p=0.5, size=n).astype(np.float32)
Y0 = X[:, 0] - X[:, 1] + np.random.normal(0, 0.1, size=n).astype(np.float32)
Y1 = Y0 + tau
Y = Y0 * (1 - T) + Y1 * T


# 2. Train/test split
train_idx = np.random.choice(n, size=int(0.7 * n), replace=False)
test_idx = np.setdiff1d(np.arange(n), train_idx)
X_train, X_test = X[train_idx], X[test_idx]
T_train, Y_train = T[train_idx], Y[train_idx]
tau_test = tau[test_idx]


# 3. CATE Estimation
start_time = time.time()
causalpfn_cate = CATEEstimator(
    device=device,
    verbose=True,
)
causalpfn_cate.fit(X_train, T_train, Y_train)
cate_hat = causalpfn_cate.estimate_cate(X_test)
cate_time = time.time() - start_time

# 4. ATE Estimation
causalpfn_ate = ATEEstimator(
    device=device,
    verbose=True,
)
causalpfn_ate.fit(X, T, Y)
ate_hat = causalpfn_ate.estimate_ate()

# 5. Evaluation
pehe = np.sqrt(np.mean((cate_hat - tau_test) ** 2))
ate_rel_error = np.abs((ate_hat - true_ate()) / true_ate())

print(f"Results:")
print(f"ATE Relative Error: {ate_rel_error:.4f}")
print(f"PEHE: {pehe:.4f}")
print(f"CATE estimation time per 1000 samples: {cate_time / (len(X) / 1000):.4f}s")