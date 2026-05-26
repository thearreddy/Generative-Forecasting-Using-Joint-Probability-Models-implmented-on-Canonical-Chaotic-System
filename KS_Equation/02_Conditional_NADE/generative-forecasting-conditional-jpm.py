# %% [code] Cell 1
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import sys
from datetime import datetime
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, message):
        for stream in self.streams:
            stream.write(message)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


OUTPUT_DIR = Path("/home/arreddy/generative_forecasting")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = OUTPUT_DIR / f"conditional_nade_training_{datetime.now():%Y%m%d_%H%M%S}.log"
_log_file = LOG_PATH.open("w", encoding="utf-8")
sys.stdout = Tee(sys.__stdout__, _log_file)
sys.stderr = Tee(sys.__stderr__, _log_file)
print(f"Logging output to {LOG_PATH}")

# %% [code] Cell 2
dataset = np.load("/home/arreddy/generative_forecasting/data/ks_dataset_imex.npy")

# %% [code] Cell 3
data_slice=dataset[int(1e6):int(1e6)+1000:]
plot_data=data_slice.T

dt=0.1
x_min,x_max=-25,25

t_start = 0
t_end = data_slice.shape[0] * dt 

plt.figure(figsize=(10,6))
plt.imshow(plot_data,
           aspect='auto',
           origin='lower',
           extent=[t_start,t_end,x_min,x_max],
           cmap='viridis',
           vmin=-2.88,
           vmax=2.88)

plt.title("Ground Truth")
plt.xlabel("t (time units)")
plt.ylabel("x (distance along the domain)")
plt.xticks([0, 20, 40, 60, 80, 100])
plt.tight_layout()
plt.colorbar()
plt.show()

# %% [code] Cell 4
dataset.shape

# %% [code] Cell 5
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

# %% [code] Cell 6
# We need to createthe joint windows of length of 3
train_dataset = JointWindowDataset(dataset)

# %% [code] Cell 7
train_dataset[0][1].shape

# %% [code] Cell 8
train_dataloader = DataLoader(train_dataset,batch_size=2000,shuffle=True,num_workers=4)

# %% [code] Cell 9
print(f'Number of time windows (samples) in the Training Dataset = {len(train_dataset)}')

# %% [code] Cell 10
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

# %% [code] Cell 11
def gaussian_nll_loss(mu, log_sigma, target):
    variance = torch.exp(log_sigma) ** 2
    loss = 0.5 * ((target - mu) ** 2 / variance) + log_sigma + 0.5 * np.log(2 * np.pi)
    #Sum across the 6 dimensions, then get the mean for the batch
    return loss.sum(dim=1).mean()

# %% [code] Cell 12
## Epochs
epochs = 100

# %% [code] Cell 13
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.set_device(device)

model = ConditionalNADE().to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4) 
train_loss_history = []
best_loss = float("inf")
save_path = OUTPUT_DIR / "Conditional_Nade.pt"

# Simple Training Loop
for epoch in range(epochs):
    train_loss = 0.0
    model.train()
    
    for history,target in train_dataloader:
        history = history.float().to(device)
        target = target.float().to(device)
        
        optimizer.zero_grad()
        # ConditionalNADE.forward expects (target x, conditioning history).
        mu_outputs, log_sigma_outputs = model(target, history)
        loss = gaussian_nll_loss(mu_outputs, log_sigma_outputs, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item()
        
    avg_loss = train_loss / len(train_dataloader)
    train_loss_history.append(avg_loss)

    if avg_loss < best_loss:
        best_loss = avg_loss
            
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_loss": best_loss,
            "train_loss_history": train_loss_history
        }, save_path)
        print(f"Saved best model at epoch {epoch+1} | Loss: {best_loss:.4f}")
    if (epoch+1)%5 == 0:
        print(f'Epoch {epoch+1}/{epochs} | Avg Train Loss: {avg_loss:.4f}')

# %% [code] Cell 14
plt.figure(figsize=(10,8))
plt.plot(range(epochs),train_loss_history)
plt.xlabel('Epochs')
plt.ylabel('Gausian NLL (Loss per Datapoint)')
plt.title('Training Loss')
plt.grid(True)
loss_curve_path = OUTPUT_DIR / "Conditional_Nade_loss_curve.png"
plt.savefig(loss_curve_path, dpi=300, bbox_inches='tight')
print(f"Saved loss curve to {loss_curve_path}")

# %% [code] Cell 15

