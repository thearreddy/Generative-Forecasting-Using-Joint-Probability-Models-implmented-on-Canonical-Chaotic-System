import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr, wasserstein_distance
from sklearn.linear_model import LinearRegression


class DualLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


sys.stdout = DualLogger("UQ_Metrics.log")


dataset = np.load("/home/arreddy/generative_forecasting/data/ks_dataset_imex.npy").astype(np.float32)


class ConditionalNADE(nn.Module):
    def __init__(self, target_dim=400, history_dim=400, hidden_dim=1500):
        super().__init__()
        self.target_dim = target_dim
        self.history_dim = history_dim
        self.hidden_dim = hidden_dim

        self.W = nn.Parameter(torch.randn(hidden_dim, target_dim) * 0.01)
        self.c = nn.Parameter(torch.zeros(hidden_dim))
        self.U = nn.Linear(history_dim, hidden_dim, bias=False)

        self.V_mu = nn.Parameter(torch.randn(target_dim, hidden_dim) * 0.01)
        self.b_mu = nn.Parameter(torch.zeros(target_dim))

        self.V_log_sigma = nn.Parameter(torch.randn(target_dim, hidden_dim) * 0.01)
        self.b_log_sigma = nn.Parameter(torch.zeros(target_dim))

    def forward(self, x, history):
        B = x.shape[0]
        mu_out = torch.zeros(B, self.target_dim, device=x.device)
        log_sigma_out = torch.zeros(B, self.target_dim, device=x.device)
        a = self.c.unsqueeze(0).expand(B, -1) + self.U(history)

        for i in range(self.target_dim):
            h = torch.relu(a)
            mu_i = h @ self.V_mu[i] + self.b_mu[i]
            log_sigma_i = h @ self.V_log_sigma[i] + self.b_log_sigma[i]
            log_sigma_i = torch.clamp(log_sigma_i, min=-5, max=2)
            mu_out[:, i] = mu_i
            log_sigma_out[:, i] = log_sigma_i

            if i < self.target_dim - 1:
                v_i = x[:, i].unsqueeze(1)
                W_i = self.W[:, i].unsqueeze(0)
                a = a + v_i * W_i

        return mu_out, log_sigma_out

    @torch.no_grad()
    def sample(self, history):
        B = history.shape[0]
        samples = torch.zeros(B, self.target_dim, device=history.device)
        a = self.c.unsqueeze(0).expand(B, -1) + self.U(history)

        for i in range(self.target_dim):
            h = torch.relu(a)
            mu_i = h @ self.V_mu[i] + self.b_mu[i]
            log_sigma_i = h @ self.V_log_sigma[i] + self.b_log_sigma[i]
            log_sigma_i = torch.clamp(log_sigma_i, min=-5, max=2)
            sigma_i = torch.exp(log_sigma_i)
            x_i = mu_i + sigma_i * torch.randn(B, device=history.device)
            samples[:, i] = x_i

            if i < self.target_dim - 1:
                a = a + x_i.unsqueeze(1) * self.W[:, i].unsqueeze(0)

        return samples


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load("/home/arreddy/generative_forecasting/CN_100Epochs.pt", map_location=device)

nx = dataset.shape[1]
model = ConditionalNADE(target_dim=2 * nx, history_dim=2 * nx, hidden_dim=1500).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


# Statistics of the training data.
training_data = dataset[:1000003]
mean = training_data.mean(axis=0).astype(np.float32)
std = training_data.std(axis=0).astype(np.float32) + 1e-8
mean_window = torch.tensor(np.tile(mean, 2).astype(np.float32), device=device)
std_window = torch.tensor(np.tile(std, 2).astype(np.float32), device=device)


steps = 100
num_test_trajectories = 500
num_generated_samples = 1000
k = 50

start_idx = int(1e6)
end_idx = int(4e6)

base_points = np.linspace(start_idx, end_idx - 1000, num_test_trajectories).astype(int)
random_offsets = np.random.randint(0, 1000, size=num_test_trajectories)
test_start_indices = base_points + random_offsets
test_start_indices = test_start_indices[(test_start_indices > 0) & (test_start_indices + steps < len(dataset))]
num_test_trajectories = len(test_start_indices)

print(f"Running UQ metrics on {num_test_trajectories} test trajectories")
print(f"Device: {device}")


all_variance = np.zeros((num_test_trajectories, steps))
all_autocorr = np.zeros((num_test_trajectories, steps))
all_drift = np.zeros((num_test_trajectories, steps))
all_mae_pertraj = np.zeros((num_test_trajectories, steps))


with torch.no_grad():
    for traj_idx, idx in enumerate(test_start_indices):
        current_history = torch.tensor(dataset[idx - 1 : idx + 1].reshape(-1), device=device)
        truth = torch.tensor(dataset[idx + 1 : idx + steps + 1], device=device)

        prev_ensemble_heads = current_history[-nx:].unsqueeze(0).expand(k, -1).clone()
        prev_var = 0.0
        cumulative_wd = 0.0

        for step in range(steps):
            current_history_norm = (current_history - mean_window) / std_window
            history_batch = current_history_norm.unsqueeze(0).expand(num_generated_samples, -1)

            pred_norm = model.sample(history_batch)
            pred = pred_norm * std_window + mean_window

            candidate_tails = pred[:, :nx]
            candidate_heads = pred[:, nx:]

            current_state = current_history[-nx:]
            distances = torch.norm(candidate_tails - current_state, dim=1)
            _, topk_indices = torch.topk(distances, k=k, largest=False)

            ensemble_tails = candidate_tails[topk_indices]
            ensemble_heads = candidate_heads[topk_indices]

            # 1. Ensemble variance.
            current_var = torch.var(ensemble_heads, dim=0).mean().item()
            all_variance[traj_idx, step] = current_var

            # 2. Autocorrelation.
            t_mean = torch.mean(ensemble_tails, dim=0)
            h_mean = torch.mean(ensemble_heads, dim=0)
            t_std = torch.std(ensemble_tails, dim=0)
            h_std = torch.std(ensemble_heads, dim=0)
            numerator = torch.mean((ensemble_tails - t_mean) * (ensemble_heads - h_mean), dim=0)
            denom = t_std * h_std
            ac_dims = torch.where(denom > 1e-8, numerator / denom, torch.zeros_like(numerator))
            all_autocorr[traj_idx, step] = torch.mean(ac_dims).item()

            # 3. Wasserstein drift.
            tails_np = ensemble_tails.detach().cpu().numpy()
            prev_heads_np = prev_ensemble_heads.detach().cpu().numpy()
            wd_per_dim = []

            for d in range(nx):
                wd_per_dim.append(wasserstein_distance(tails_np[:, d], prev_heads_np[:, d]))

            wd = np.mean(wd_per_dim)

            if current_var < prev_var:
                signed_wd = -wd
            else:
                signed_wd = wd

            cumulative_wd += signed_wd
            all_drift[traj_idx, step] = cumulative_wd

            # 4. MAE per timestep.
            winning_head = ensemble_heads[0]
            all_mae_pertraj[traj_idx, step] = torch.mean(torch.abs(winning_head - truth[step])).item()

            current_history = torch.cat([current_history[-nx:], winning_head])
            prev_ensemble_heads = ensemble_heads
            prev_var = current_var

        print(f"Trajectory {traj_idx + 1}/{num_test_trajectories} has been completed")


rho_var = []
rho_ac = []
rho_wd = []
rho_all = []

for i in range(num_test_trajectories):
    y = all_mae_pertraj[i]

    X_var = all_variance[i].reshape(-1, 1)
    X_ac = all_autocorr[i].reshape(-1, 1)
    X_wd = all_drift[i].reshape(-1, 1)
    X_all = np.stack([all_variance[i], all_autocorr[i], all_drift[i]], axis=1)

    y_pred = LinearRegression().fit(X_var, y).predict(X_var)
    rho_var.append(pearsonr(y, y_pred)[0])

    y_pred = LinearRegression().fit(X_ac, y).predict(X_ac)
    rho_ac.append(pearsonr(y, y_pred)[0])

    y_pred = LinearRegression().fit(X_wd, y).predict(X_wd)
    rho_wd.append(pearsonr(y, y_pred)[0])

    y_pred = LinearRegression().fit(X_all, y).predict(X_all)
    rho_all.append(pearsonr(y, y_pred)[0])


rho_var = np.array(rho_var)
rho_ac = np.array(rho_ac)
rho_wd = np.array(rho_wd)
rho_all = np.array(rho_all)

rho_var = rho_var[np.isfinite(rho_var)]
rho_ac = rho_ac[np.isfinite(rho_ac)]
rho_wd = rho_wd[np.isfinite(rho_wd)]
rho_all = rho_all[np.isfinite(rho_all)]


plt.figure(figsize=(8, 4.5))
plt.plot(rho_var, label="Var", linewidth=1.5)
plt.plot(rho_ac, label="AC", linewidth=1.5)
plt.plot(rho_wd, label="WD", linewidth=1.5)
plt.plot(rho_all, label="All UQs", linewidth=1.5)
plt.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
plt.xlabel("Forecast trajectory index")
plt.ylabel("Pearson coefficient")
plt.title("Pearson Coefficients Per Forecast Trajectory")
plt.legend()
plt.tight_layout()
plt.savefig("Pearson_Coefficients_Per_Forecast.png", dpi=300, bbox_inches="tight")
plt.close()


fig, axes = plt.subplots(4, 1, figsize=(7, 10), sharex=True)

for ax, rho, label in zip(
    axes,
    [rho_var, rho_ac, rho_wd, rho_all],
    ["Var", "AC", "WD", "All UQs"],
):
    ax.hist(rho, bins=50)
    med = np.median(rho)
    ax.axvline(med, color="red", label=f"Median = {med:.2f}")
    ax.set_ylabel(label)
    ax.legend()

axes[-1].set_xlabel("Pearson coefficient")
fig.suptitle("Pearson Coefficient Histograms")
plt.tight_layout()
plt.savefig("Pearson_Coefficients_Histograms.png", dpi=300, bbox_inches="tight")
plt.close()

print("Saved Pearson_Coefficients_Per_Forecast.png")
print("Saved Pearson_Coefficients_Histograms.png")
print("Saved logs to UQ_Metrics.log")
