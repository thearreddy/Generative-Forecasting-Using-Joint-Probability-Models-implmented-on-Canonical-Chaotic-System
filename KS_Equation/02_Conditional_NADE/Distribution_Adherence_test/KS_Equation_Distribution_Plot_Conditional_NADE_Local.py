from pathlib import Path
import gc

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


DATASET_PATH = Path("/home/arreddy/generative_forecasting/data/ks_dataset_imex.npy")
CHECKPOINT_PATH = Path("/home/arreddy/generative_forecasting/Conditional_Nade.pt")

TEST_INDEX = int(1e6)
FORECAST_HORIZON = int(1e5)
N_CANDIDATES = int(1e4)
CANDIDATE_BATCH_SIZE = 5000

HISTORY_LEN = 2
TARGET_LEN = 2
STATE_DIM = 200
HIDDEN_DIM = 1500
TRAIN_WINDOWS_FOR_STATS = 1_000_000

SCRIPT_DIR = Path(__file__).resolve().parent
LOG_FILE = SCRIPT_DIR / "conditional_nade_local_outputs.txt"


def write_log(message):
    print(message, flush=True)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(str(message) + "\n")


def load_dataset():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    dataset = np.load(DATASET_PATH, mmap_mode="r")
    write_log(f"Loaded dataset from: {DATASET_PATH}")
    write_log(f"Dataset shape: {dataset.shape}")
    write_log(f"Dataset dtype: {dataset.dtype}")
    return dataset


def calculate_normalization_stats(dataset):
    num_windows = min(
        TRAIN_WINDOWS_FOR_STATS,
        len(dataset) - HISTORY_LEN - TARGET_LEN + 1,
    )

    stats_end = num_windows + HISTORY_LEN + TARGET_LEN - 1
    data_for_stats = dataset[:stats_end]

    mean_state = data_for_stats.mean(axis=0).astype(np.float32)
    std_state = data_for_stats.std(axis=0).astype(np.float32) + 1e-8

    mean_window = np.tile(mean_state, HISTORY_LEN).astype(np.float32)
    std_window = np.tile(std_state, HISTORY_LEN).astype(np.float32)

    write_log(f"Number of time windows used for stats: {num_windows}")
    return mean_window, std_window


def get_effective_forecast_horizon(dataset):
    available_horizon = len(dataset) - TEST_INDEX - HISTORY_LEN
    if available_horizon <= 0:
        raise ValueError(
            "TEST_INDEX leaves no truth data for comparison. "
            f"len(dataset)={len(dataset)}, TEST_INDEX={TEST_INDEX}, "
            f"HISTORY_LEN={HISTORY_LEN}"
        )

    effective_horizon = min(FORECAST_HORIZON, available_horizon)
    if effective_horizon < FORECAST_HORIZON:
        write_log(
            "Requested forecast horizon exceeds available truth data. "
            f"Using {effective_horizon} instead of {FORECAST_HORIZON}."
        )

    write_log(f"Effective forecast horizon: {effective_horizon}")
    return effective_horizon


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
        batch_size = x.shape[0]
        mu_out = torch.zeros(batch_size, self.target_dim, device=x.device)
        log_sigma_out = torch.zeros(batch_size, self.target_dim, device=x.device)
        a = self.c.unsqueeze(0).expand(batch_size, -1) + self.U(history)

        for i in range(self.target_dim):
            h = torch.relu(a)
            mu_i = h @ self.V_mu[i] + self.b_mu[i]
            log_sigma_i = h @ self.V_log_sigma[i] + self.b_log_sigma[i]
            log_sigma_i = torch.clamp(log_sigma_i, min=-5, max=2)

            mu_out[:, i] = mu_i
            log_sigma_out[:, i] = log_sigma_i

            if i < self.target_dim - 1:
                a = a + x[:, i].unsqueeze(1) * self.W[:, i].unsqueeze(0)

        return mu_out, log_sigma_out

    @torch.no_grad()
    def sample(self, history):
        batch_size = history.shape[0]
        samples = torch.zeros(batch_size, self.target_dim, device=history.device)
        a = self.c.unsqueeze(0).expand(batch_size, -1) + self.U(history)

        for i in range(self.target_dim):
            h = torch.relu(a)
            mu_i = h @ self.V_mu[i] + self.b_mu[i]
            log_sigma_i = h @ self.V_log_sigma[i] + self.b_log_sigma[i]
            log_sigma_i = torch.clamp(log_sigma_i, min=-5, max=2)

            sigma_i = torch.exp(log_sigma_i)
            sampled_i = mu_i + sigma_i * torch.randn(batch_size, device=history.device)
            samples[:, i] = sampled_i

            if i < self.target_dim - 1:
                a = a + sampled_i.unsqueeze(1) * self.W[:, i].unsqueeze(0)

        return samples


def load_model(device):
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only = True)
    model = ConditionalNADE(
        target_dim=TARGET_LEN * STATE_DIM,
        history_dim=HISTORY_LEN * STATE_DIM,
        hidden_dim=HIDDEN_DIM,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    write_log(f"Loaded model from: {CHECKPOINT_PATH}")
    write_log(f"Checkpoint epoch: {checkpoint.get('epoch')}")
    write_log(f"Best loss: {checkpoint.get('best_loss')}")
    return model


def forecast_trajectory(dataset, mean_window, std_window, model, device, forecast_horizon):
    initial_history = dataset[TEST_INDEX:TEST_INDEX + HISTORY_LEN].astype(np.float32)
    if len(initial_history) != HISTORY_LEN:
        raise ValueError(
            "Initial history is incomplete. "
            f"Got {len(initial_history)} rows, expected {HISTORY_LEN}."
        )

    current_history = torch.tensor(
        initial_history.reshape(-1),
        dtype=torch.float32,
        device=device,
    )

    mean_tensor = torch.tensor(mean_window, dtype=torch.float32, device=device)
    std_tensor = torch.tensor(std_window, dtype=torch.float32, device=device)
    forecasted_trajectory = []

    with torch.no_grad():
        for step in range(forecast_horizon):
            current_history_norm = (current_history - mean_tensor) / (std_tensor + 1e-8)

            best_distance = None
            best_next_state = None
            candidates_done = 0

            while candidates_done < N_CANDIDATES:
                batch_size = min(CANDIDATE_BATCH_SIZE, N_CANDIDATES - candidates_done)
                batch_history = current_history_norm.unsqueeze(0).expand(batch_size, -1)

                candidates_norm = model.sample(batch_history)
                candidates = (candidates_norm * std_tensor) + mean_tensor

                candidate_tails = candidates[:, :STATE_DIM]
                candidate_heads = candidates[:, STATE_DIM:]
                latest_state = current_history[STATE_DIM:]

                distances = torch.norm(candidate_tails - latest_state, dim=1)
                best_index = torch.argmin(distances)
                best_distance_in_batch = distances[best_index]

                if best_distance is None or best_distance_in_batch < best_distance:
                    best_distance = best_distance_in_batch
                    best_next_state = candidate_heads[best_index].clone()

                candidates_done += batch_size

            forecasted_trajectory.append(best_next_state.cpu().numpy())
            current_history = torch.cat([current_history[STATE_DIM:], best_next_state])

            if (step + 1) % 100 == 0:
                write_log(f"Forecasted {step + 1}/{forecast_horizon} steps")

    forecasted_trajectory = np.asarray(forecasted_trajectory, dtype=np.float32)
    write_log(f"Forecast shape: {forecasted_trajectory.shape}")
    return forecasted_trajectory


def add_to_histogram(histogram_counts, data_rows, bins):
    values = np.asarray(data_rows).reshape(-1)
    counts, _ = np.histogram(values, bins=bins)
    histogram_counts += counts


def convert_counts_to_density(counts, bins):
    bin_widths = np.diff(bins)
    total_count = counts.sum()
    if total_count == 0:
        raise ValueError("Cannot convert empty histogram counts to density.")
    return counts / (total_count * bin_widths)


def percentile_from_histogram(counts, bins, percentile):
    target_count = counts.sum() * percentile / 100.0
    cumulative_counts = np.cumsum(counts)
    bin_index = np.searchsorted(cumulative_counts, target_count)
    bin_index = min(bin_index, len(bins) - 2)
    return bins[bin_index]


def plot_distribution_adherence(dataset, forecasted_trajectory, forecast_horizon):
    truth_data = dataset[
        TEST_INDEX + HISTORY_LEN:TEST_INDEX + HISTORY_LEN + forecast_horizon
    ].astype(np.float32)
    pred_data = forecasted_trajectory[:len(truth_data)]

    write_log(f"Truth data shape for plotting: {truth_data.shape}")
    write_log(f"Prediction data shape for plotting: {pred_data.shape}")

    if truth_data.size == 0 or pred_data.size == 0:
        raise ValueError(
            "No data available for distribution plot. "
            f"truth_data.size={truth_data.size}, pred_data.size={pred_data.size}"
        )

    all_min = min(dataset.min(), truth_data.min(), pred_data.min())
    all_max = max(dataset.max(), truth_data.max(), pred_data.max())
    bins = np.linspace(all_min, all_max, 80)
    centers = 0.5 * (bins[:-1] + bins[1:])

    all_counts = np.zeros(len(bins) - 1, dtype=np.float64)
    train_counts = np.zeros(len(bins) - 1, dtype=np.float64)
    truth_counts = np.zeros(len(bins) - 1, dtype=np.float64)
    pred_counts = np.zeros(len(bins) - 1, dtype=np.float64)

    chunk_size = 50000
    for start in range(0, len(dataset), chunk_size):
        end = min(start + chunk_size, len(dataset))
        add_to_histogram(all_counts, dataset[start:end], bins)

    train_end = min(TEST_INDEX, len(dataset))
    for start in range(0, train_end, chunk_size):
        end = min(start + chunk_size, train_end)
        add_to_histogram(train_counts, dataset[start:end], bins)

    add_to_histogram(truth_counts, truth_data, bins)
    add_to_histogram(pred_counts, pred_data, bins)

    eps = 1e-12
    all_hist = convert_counts_to_density(all_counts, bins) + eps
    train_hist = convert_counts_to_density(train_counts, bins) + eps
    truth_hist = convert_counts_to_density(truth_counts, bins) + eps
    pred_hist = convert_counts_to_density(pred_counts, bins) + eps

    hist_path = SCRIPT_DIR / "conditional_nade_local_distribution_histograms.txt"
    np.savetxt(
        hist_path,
        np.column_stack([centers, all_hist, train_hist, truth_hist, pred_hist]),
        header="center all_density train_density truth_density prediction_density",
    )

    left_limit = percentile_from_histogram(train_counts, bins, 5)
    right_limit = percentile_from_histogram(train_counts, bins, 95)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.fill_between(centers, all_hist, alpha=0.35, label="All data")
    ax.plot(centers, train_hist, color="black", linewidth=1.5, label="Training set")
    ax.plot(centers, truth_hist, color="steelblue", linewidth=1.5, label="Truth")
    ax.plot(centers, pred_hist, color="crimson", linestyle="--", linewidth=1.5, label="Prediction")
    ax.set_yscale("log")
    ax.set_xlabel("State value")
    ax.set_ylabel("Probability density")
    ax.set_title("Main Distribution")
    ax.legend()

    ax = axes[1]
    ax.fill_between(centers, all_hist, alpha=0.35)
    ax.plot(centers, train_hist, color="black", linewidth=1.5)
    ax.plot(centers, truth_hist, color="steelblue", linewidth=1.5)
    ax.plot(centers, pred_hist, color="crimson", linestyle="--", linewidth=1.5)
    ax.set_yscale("log")
    ax.set_xlim(centers.min(), left_limit)
    ax.set_xlabel("State value")
    ax.set_ylabel("Probability density")
    ax.set_title("Left Tail View")

    ax = axes[2]
    ax.fill_between(centers, all_hist, alpha=0.35)
    ax.plot(centers, train_hist, color="black", linewidth=1.5)
    ax.plot(centers, truth_hist, color="steelblue", linewidth=1.5)
    ax.plot(centers, pred_hist, color="crimson", linestyle="--", linewidth=1.5)
    ax.set_yscale("log")
    ax.set_xlim(right_limit, centers.max())
    ax.set_xlabel("State value")
    ax.set_ylabel("Probability density")
    ax.set_title("Right Tail View")

    plt.tight_layout()
    figure_path = SCRIPT_DIR / "conditional_nade_local_distribution_adherence.png"
    plt.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close()

    write_log(f"Saved histogram values: {hist_path}")
    write_log(f"Saved distribution plot: {figure_path}")


def clear_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        write_log("Cleared CUDA cache.")


def main():
    LOG_FILE.write_text("Conditional NADE local distribution adherence test\n", encoding="utf-8")

    dataset = None
    mean_window = None
    std_window = None
    model = None
    forecasted_trajectory = None

    try:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        write_log(f"Using device: {device}")
        write_log(f"Dataset path: {DATASET_PATH}")
        write_log(f"Checkpoint path: {CHECKPOINT_PATH}")

        dataset = load_dataset()
        forecast_horizon = get_effective_forecast_horizon(dataset)
        mean_window, std_window = calculate_normalization_stats(dataset)
        model = load_model(device)
        forecasted_trajectory = forecast_trajectory(
            dataset,
            mean_window,
            std_window,
            model,
            device,
            forecast_horizon,
        )
        plot_distribution_adherence(dataset, forecasted_trajectory, forecast_horizon)

        write_log("Done.")
    finally:
        forecasted_trajectory = None
        model = None
        std_window = None
        mean_window = None
        dataset = None
        clear_gpu_memory()


if __name__ == "__main__":
    main()
