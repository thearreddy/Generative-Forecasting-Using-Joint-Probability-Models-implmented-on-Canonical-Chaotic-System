import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LOG_FILE = SCRIPT_DIR / "short_term.log"

def write_log(message):
    text = str(message)
    print(text, flush=True)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(text + "\n")

dataset = np.load("/home/arreddy/generative_forecasting/data/ks_dataset_imex.npy")

class JointWindowDataset(Dataset):
    def __init__(self, trajectory,history_len=2, target_len = 2, num_windows=1_000_000):
        self.trajectory = trajectory
        self.history_len = history_len
        self.target_len = target_len
        self.num_windows = min(num_windows, len(trajectory) - history_len - target_len + 1)

        data_for_stats = trajectory[:self.num_windows + history_len + target_len - 1]

        
        mean_state = data_for_stats.mean(axis=0).astype(np.float32)
        std_state = data_for_stats.std(axis=0).astype(np.float32) + 1e-8
        
        self.mean = np.tile(mean_state,history_len)
        self.std = np.tile(std_state,history_len)

    def __len__(self):
        return self.num_windows

    def __getitem__(self, idx):
        history = self.trajectory[idx:idx+self.history_len]
        target = self.trajectory[idx+1:idx+1+self.target_len]

        history = history.reshape(-1).astype(np.float32)
        target = target.reshape(-1).astype(np.float32)

        history = (history - self.mean) / self.std
        target = (target - self.mean) / self.std

        return torch.from_numpy(history), torch.from_numpy(target)

train_dataset = JointWindowDataset(dataset)
train_dataloader = DataLoader(train_dataset,batch_size=2000,shuffle=True,num_workers=4)
print(f'Number of time windows (samples) in the Training Dataset = {len(train_dataset)}')

class ConditionalNADE(nn.Module):
    def __init__(self,target_dim = 400,history_dim = 400, hidden_dim = 1500):
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

    def forward(self,x,history):
        B = x.shape[0]
        mu_out = torch.zeros(B,self.target_dim,device = x.device)
        log_sigma_out = torch.zeros(B,self.target_dim,device = x.device)
        a = self.c.unsqueeze(0).expand(B,-1) + self.U(history)
        
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

def gaussian_nll_loss(mu, log_sigma, target):
    variance = torch.exp(log_sigma) ** 2
    loss = 0.5 * ((target - mu) ** 2 / variance) + log_sigma + 0.5 * np.log(2 * np.pi)
    #Sum across the 6 dimensions, then get the mean for the batch
    return loss.sum(dim=1).mean()

test_idx = int(1e6)
available_ground_truth = dataset[test_idx:test_idx+2].reshape(-1)
available_ground_truth = available_ground_truth.astype(np.float32)
print(available_ground_truth.shape)

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
checkpoint=torch.load("/home/arreddy/generative_forecasting/CN_100Epochs.pt",map_location=device)
model=ConditionalNADE(target_dim=400,history_dim=400,hidden_dim=1500).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Calculate normalization stats from training data
data_for_stats = dataset[:1000000+2+2-1]
mean_state = data_for_stats.mean(axis=0).astype(np.float32)
std_state = data_for_stats.std(axis=0).astype(np.float32) + 1e-8
mean_window = np.tile(mean_state, 2).astype(np.float32)
std_window = np.tile(std_state, 2).astype(np.float32)

# For each test trajectory, forecast sequentially using conditional history
steps = 100
start_idx = 1000000
end_idx = 4000000
num_samples = 500
base_points = np.linspace(start_idx, end_idx - 1000, num_samples).astype(int)
random_offsets = np.random.randint(0, 1000, size=num_samples)
test_start_indices = [int(b+o) for b,o in zip(base_points, random_offsets) if int(b+o)+steps+1 <= len(dataset)]
num_test = len(test_start_indices)
nx = dataset.shape[1]
all_preds_t = torch.empty((num_test, steps, nx), device=device)
all_ground_truth_t = torch.empty((num_test, steps, nx), device=device)
dataset_t = torch.tensor(dataset, device=device, dtype=torch.float32)

for traj_idx, idx in enumerate(test_start_indices):
    # Get initial history from dataset
    initial_history = dataset[idx:idx+2].reshape(-1).astype(np.float32)
    current_history = torch.from_numpy(initial_history).float().to(device)
    
    all_ground_truth_t[traj_idx] = torch.from_numpy(dataset[idx + 1 : idx + steps + 1])
    
    for step in range(steps):
        with torch.no_grad():
            # Normalize history
            current_history_norm = (current_history - torch.tensor(mean_window, device=device)) / torch.tensor(std_window, device=device)
            # Sample candidates conditioned on current history
            batch_history = current_history_norm.unsqueeze(0).expand(1000, -1)  # Sample 1000 candidates
            candidates_norm = model.sample(batch_history)
            candidates = (candidates_norm * torch.tensor(std_window, device=device)) + torch.tensor(mean_window, device=device)
            # Extract tails and heads
            candidate_tails = candidates[:, :200]
            candidate_heads = candidates[:, 200:]
            latest_state = current_history[200:]            
            # Find best matching candidate
            distances = torch.sum((candidate_tails - latest_state) ** 2, dim=1)
            best_index = torch.argmin(distances)
            winning_forecast = candidate_heads[best_index]
        
        all_preds_t[traj_idx, step] = winning_forecast
        # Roll window: keep new history + new forecast
        current_history = torch.cat([current_history[200:], winning_forecast])
    
    if (traj_idx + 1) % 50 == 0:
        print(f"Processed {traj_idx + 1}/{num_test}")
all_preds = all_preds_t.cpu().numpy()
all_ground_truth = all_ground_truth_t.cpu().numpy()


all_preds = np.array(all_preds)              # Shape: (500, 100, 200)
all_ground_truth = np.array(all_ground_truth) # Shape: (500, 100, 200)


# Averaging across the 500 initial conditions (axis=0)
mae_per_step = np.mean(np.abs(all_preds - all_ground_truth), axis=0)  # Shape: (100, 200)


mae = mae_per_step.T   # shape: (100, 200)
plt.figure(figsize=(7, 4.5))
im = plt.imshow(mae,aspect="auto",origin="lower",cmap="magma",extent=[0, mae.shape[1]-1, 1, mae.shape[0]])
plt.xlabel("Forecast Step (t)")
plt.ylabel("Spatial Grid Point, x")
plt.title("MAE across 500 different intial conditions for KS | Forecasted for 200 steps")
cbar = plt.colorbar(im)
cbar.set_label("MAE")
plt.tight_layout()
plt.savefig(SCRIPT_DIR / "short_term_mae_heatmap.png", dpi=300, bbox_inches="tight")
plt.close()



steps = np.arange(1, mae_per_step.shape[0] + 1)
mae_curve = mae_per_step.mean(axis=1)
plt.figure(figsize=(5, 4))
plt.plot(steps, mae_curve, label="Uncond.", linewidth=2)
plt.xlabel("Forecast step, t")
plt.ylabel("MAE(x̄)")
plt.title("Mean MAE over spatial nodes")
plt.legend()
plt.tight_layout()
plt.savefig(SCRIPT_DIR / "short_term_mae_curve.png", dpi=300, bbox_inches="tight")
plt.close()