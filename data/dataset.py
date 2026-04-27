"""
dataset.py
──────────
Generate sliding-window measurement sequences and wrap them in
a PyTorch Dataset / DataLoader.

Label: 0 = normal, 1 = attack

Each sample is a tensor of shape (window_size, m) where m is the
number of measurements for the given test system.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from typing import Tuple, Optional, List

from data.grid_topology import GridSystem, get_system
from utils.state_estimator import WLSEstimator
from utils.attack_generator import AttackGenerator, AttackType


# ─────────────────────────────────────────────────────────────
#  Low-level data generation
# ─────────────────────────────────────────────────────────────

def generate_timeseries(
    estimator: WLSEstimator,
    n_steps: int,
    rng: np.random.Generator,
    load_variation: float = 0.05,
) -> np.ndarray:
    """
    Generate n_steps consecutive measurement vectors under normal
    operation, with slow load variation.

    Returns
    -------
    Z : (n_steps, m) measurement matrix
    """
    x = estimator.random_state(rng)
    m = estimator.m
    Z = np.zeros((n_steps, m))
    for t in range(n_steps):
        # Slow drift in operating point
        x = x + rng.normal(0, load_variation / n_steps, x.shape)
        z = estimator.generate_measurements(x)
        Z[t] = z
    return Z


def generate_dataset(
    system_name: str,
    n_samples: int,
    window_size: int = 10,
    noise_std: float = 0.01,
    attack_intensity_min: float = 0.10,
    attack_intensity_max: float = 0.50,
    attack_types: Optional[List[str]] = None,
    sparse_k: int = 5,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate (X, r, y) dataset.

    Returns
    -------
    X : (n_samples, window_size, m)   measurement windows
    R : (n_samples, window_size, m)   residual windows
    y : (n_samples,)                  labels (0/1)
    """
    if attack_types is None:
        attack_types = ["targeted", "random", "sparse"]

    rng      = np.random.default_rng(seed)
    system   = get_system(system_name)
    est      = WLSEstimator(system, noise_std=noise_std)
    attacker = AttackGenerator(est.H, sparse_k=sparse_k, seed=seed)

    m = est.m
    n_normal = n_samples // 2
    n_attack = n_samples - n_normal

    X_list, R_list, y_list = [], [], []

    # ── Normal samples ──────────────────────────────────────
    # Generate a long timeseries and extract sliding windows
    total_steps = n_normal + window_size + 100
    Z_normal    = generate_timeseries(est, total_steps, rng)
    residuals_n = np.zeros_like(Z_normal)
    for t in range(total_steps):
        x_hat           = est.estimate(Z_normal[t])
        residuals_n[t]  = est.residual(Z_normal[t], x_hat)

    for i in range(n_normal):
        start     = rng.integers(0, total_steps - window_size)
        X_list.append(Z_normal[start: start + window_size])
        R_list.append(residuals_n[start: start + window_size])
        y_list.append(0)

    # ── Attack samples ──────────────────────────────────────
    per_type = n_attack // len(attack_types)
    for atype in attack_types:
        n_this = per_type if atype != attack_types[-1] else (n_attack - per_type * (len(attack_types) - 1))
        total_steps_a = n_this + window_size + 100
        Z_base        = generate_timeseries(est, total_steps_a, rng)

        for i in range(n_this):
            intensity = rng.uniform(attack_intensity_min, attack_intensity_max)
            attacker.intensity = intensity

            start = rng.integers(0, total_steps_a - window_size)
            window_z = Z_base[start: start + window_size].copy()
            window_r = np.zeros_like(window_z)

            # Inject attack into a random subset of time steps in the window
            attack_start = rng.integers(0, window_size // 2)
            for t in range(window_size):
                if t >= attack_start:
                    za, _ = attacker.generate(window_z[t], attack_type=atype)
                    window_z[t] = za
                x_hat        = est.estimate(window_z[t])
                window_r[t]  = est.residual(window_z[t], x_hat)

            X_list.append(window_z)
            R_list.append(window_r)
            y_list.append(1)

    X = np.array(X_list, dtype=np.float32)   # (N, w, m)
    R = np.array(R_list, dtype=np.float32)   # (N, w, m)
    y = np.array(y_list, dtype=np.float32)   # (N,)

    # Shuffle
    perm = rng.permutation(len(y))
    return X[perm], R[perm], y[perm]


# ─────────────────────────────────────────────────────────────
#  PyTorch Dataset
# ─────────────────────────────────────────────────────────────

class FDIADataset(Dataset):
    """
    Wraps numpy arrays into a PyTorch Dataset.

    Each item: (x_window, r_window, label)
      x_window : (window_size, m) float tensor
      r_window : (window_size, m) float tensor
      label    : scalar float tensor {0., 1.}
    """

    def __init__(
        self,
        X: np.ndarray,
        R: np.ndarray,
        y: np.ndarray,
        normalise: bool = True,
        stats: Optional[Tuple] = None,
    ):
        self.X = torch.from_numpy(X)   # (N, w, m)
        self.R = torch.from_numpy(R)
        self.y = torch.from_numpy(y)

        if normalise:
            if stats is None:
                self.x_mean = self.X.mean(dim=(0, 1), keepdim=True)
                self.x_std  = self.X.std(dim=(0, 1), keepdim=True).clamp(min=1e-8)
                self.r_mean = self.R.mean(dim=(0, 1), keepdim=True)
                self.r_std  = self.R.std(dim=(0, 1), keepdim=True).clamp(min=1e-8)
            else:
                self.x_mean, self.x_std, self.r_mean, self.r_std = stats

            self.X = (self.X - self.x_mean) / self.x_std
            self.R = (self.R - self.r_mean) / self.r_std

        self.stats = (self.x_mean, self.x_std, self.r_mean, self.r_std) \
                     if normalise else None

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.R[idx], self.y[idx]


# ─────────────────────────────────────────────────────────────
#  Convenience builder
# ─────────────────────────────────────────────────────────────

def build_dataloaders(cfg: dict) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and test DataLoaders from a config dict.

    Parameters
    ----------
    cfg : dict with keys matching configs/config.yaml

    Returns
    -------
    train_loader, test_loader
    """
    sys_name   = cfg["system"]["name"]
    noise_std  = cfg["system"]["noise_std"]
    data_cfg   = cfg["data"]
    train_cfg  = cfg["training"]

    print(f"[Dataset] Generating training data ({data_cfg['n_train']} samples) …")
    X_tr, R_tr, y_tr = generate_dataset(
        system_name          = sys_name,
        n_samples            = data_cfg["n_train"],
        window_size          = data_cfg["window_size"],
        noise_std            = noise_std,
        attack_intensity_min = data_cfg["attack_intensity_min"],
        attack_intensity_max = data_cfg["attack_intensity_max"],
        attack_types         = data_cfg["attack_types"],
        sparse_k             = data_cfg["sparse_k"],
        seed                 = data_cfg["seed"],
    )

    print(f"[Dataset] Generating test data ({data_cfg['n_test']} samples) …")
    X_te, R_te, y_te = generate_dataset(
        system_name          = sys_name,
        n_samples            = data_cfg["n_test"],
        window_size          = data_cfg["window_size"],
        noise_std            = noise_std,
        attack_intensity_min = data_cfg["attack_intensity_min"],
        attack_intensity_max = data_cfg["attack_intensity_max"],
        attack_types         = data_cfg["attack_types"],
        sparse_k             = data_cfg["sparse_k"],
        seed                 = data_cfg["seed"] + 1000,
    )

    train_ds = FDIADataset(X_tr, R_tr, y_tr, normalise=True)
    test_ds  = FDIADataset(X_te, R_te, y_te, normalise=True, stats=train_ds.stats)

    # Balanced sampler so model sees equal pos/neg
    labels  = y_tr.astype(int)
    counts  = np.bincount(labels)
    weights = 1.0 / counts[labels]
    sampler = WeightedRandomSampler(weights, len(weights))

    train_loader = DataLoader(
        train_ds,
        batch_size  = train_cfg["batch_size"],
        sampler     = sampler,
        num_workers = 0,
        pin_memory  = False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size  = train_cfg["batch_size"] * 2,
        shuffle     = False,
        num_workers = 0,
    )
    print(f"[Dataset] m={X_tr.shape[2]} measurements, w={X_tr.shape[1]} window")
    return train_loader, test_loader
