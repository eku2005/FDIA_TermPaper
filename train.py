"""
train.py
────────
Train the CNN+Attention FDIA detector.

Usage
-----
  python train.py                              # uses configs/config.yaml defaults
  python train.py --system ieee30 --epochs 100
  python train.py --system ieee14 --device cuda --lr 0.0005
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import yaml

# ── Local imports ────────────────────────────────────────────
from data.grid_topology import get_system
from data.dataset import build_dataloaders
from models.cnn_attention import CNNAttentionDetector
from models.detector import FDIADetector
from utils.metrics import compute_metrics, print_metrics
from utils.visualize import plot_fig4, save_training_log


# ─────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────

def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_device(cfg: dict) -> torch.device:
    pref = cfg["training"].get("device", "auto")
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


def build_model(cfg: dict, m_measurements: int) -> CNNAttentionDetector:
    mc = cfg["model"]
    return CNNAttentionDetector(
        m_measurements = m_measurements,
        window_size    = cfg["data"]["window_size"],
        conv_filters   = mc["conv_filters"],
        kernel_size    = mc["kernel_size"],
        attention_dim  = mc["attention_dim"],
        dropout        = mc["dropout"],
    )


def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for x, r, y in loader:
        x, r, y = x.to(device), r.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                pred = model(x, r)
                loss = criterion(pred, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(x, r)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct    += ((pred > 0.5).float() == y).sum().item()
        total      += x.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for x, r, y in loader:
        x, r, y = x.to(device), r.to(device), y.to(device)
        pred       = model(x, r)
        loss       = criterion(pred, y)
        total_loss += loss.item() * x.size(0)
        all_preds.append(pred.cpu().numpy())
        all_labels.append(y.cpu().numpy())

    preds  = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    metrics = compute_metrics(labels, (preds > 0.5).astype(int), preds)
    return total_loss / len(labels), metrics


# ─────────────────────────────────────────────────────────────
#  Main training loop
# ─────────────────────────────────────────────────────────────

def train(cfg: dict):
    device = get_device(cfg)
    print(f"\n{'═'*55}")
    print(f"  FDIA Detector Training")
    print(f"  System  : {cfg['system']['name'].upper()}")
    print(f"  Device  : {device}")
    print(f"{'═'*55}\n")

    # ── Data ────────────────────────────────────────────────
    train_loader, test_loader = build_dataloaders(cfg)

    # Infer m from first batch
    x_sample, _, _ = next(iter(train_loader))
    m_measurements = x_sample.shape[2]

    # ── Model ───────────────────────────────────────────────
    model = build_model(cfg, m_measurements)
    model = model.to(device)
    n_params = model.count_parameters()
    print(f"[Model] Parameters: {n_params:,}")

    # ── Optimiser & scheduler ───────────────────────────────
    tc         = cfg["training"]
    optimizer  = optim.Adam(model.parameters(),
                            lr=tc["lr"],
                            weight_decay=tc["weight_decay"])
    criterion  = nn.BCELoss()

    if tc["scheduler"] == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=tc["epochs"])
    elif tc["scheduler"] == "step":
        scheduler = StepLR(optimizer, step_size=30, gamma=0.5)
    else:
        scheduler = None

    # Mixed precision if CUDA
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    # ── Training loop ────────────────────────────────────────
    epochs    = tc["epochs"]
    patience  = tc["early_stopping_patience"]
    best_f1   = 0.0
    no_improve = 0
    history   = {
        "train_loss": [], "val_loss": [],
        "train_acc": [],  "val_dr": [], "val_far": [], "val_f1": []
    }

    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = ckpt_dir / f"best_model_{cfg['system']['name']}.pt"

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc  = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss, vmetrics = eval_one_epoch(model, test_loader, criterion, device)

        if scheduler:
            scheduler.step()

        dr  = vmetrics["detection_rate"]
        far = vmetrics["false_alarm_rate"]
        f1  = vmetrics["f1"]

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc)
        history["val_dr"].append(dr)
        history["val_far"].append(far)
        history["val_f1"].append(f1)

        elapsed = time.time() - t0
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"Loss {tr_loss:.4f}/{val_loss:.4f} | "
                  f"DR {dr*100:.1f}% F1 {f1:.4f} | "
                  f"{elapsed:.1f}s")

        # Early stopping
        if f1 > best_f1:
            best_f1 = f1
            no_improve = 0
            torch.save(model.state_dict(), best_ckpt)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\n[Train] Early stopping at epoch {epoch}")
                break

    print(f"\n[Train] Best val F1: {best_f1:.4f}")
    print(f"[Train] Checkpoint: {best_ckpt}")

    # ── Reload best model & calibrate detector ────────────────
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    detector = FDIADetector(model, beta=cfg["model"]["beta"], device=str(device))
    detector.calibrate(test_loader, target_fpr=cfg["detection"]["threshold_fpr"])

    # Save full detector
    final_ckpt = ckpt_dir / f"detector_{cfg['system']['name']}.pt"
    detector.save(str(final_ckpt))

    # ── Final evaluation ─────────────────────────────────────
    print("\n[Eval] Final test evaluation:")
    results = detector.evaluate_loader(test_loader)
    print_metrics(results, prefix=f"Test ({cfg['system']['name'].upper()})")

    # ── Save logs & plots ─────────────────────────────────────
    save_training_log(history)
    plot_fig4(
        history["train_loss"],
        history["val_loss"],
        {cfg["system"]["name"].upper(): results["accuracy"]},
    )
    print("\n[Done] Training complete. Figures saved to results/figures/")
    return detector, results


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train FDIA detector")
    parser.add_argument("--config",  default="configs/config.yaml")
    parser.add_argument("--system",  default=None, help="ieee14 | ieee30") #| ieee57")
    parser.add_argument("--epochs",  type=int,   default=None)
    parser.add_argument("--lr",      type=float, default=None)
    parser.add_argument("--device",  default=None)
    parser.add_argument("--seed",    type=int,   default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg  = load_config(args.config)

    # CLI overrides
    if args.system: cfg["system"]["name"]        = args.system
    if args.epochs: cfg["training"]["epochs"]    = args.epochs
    if args.lr:     cfg["training"]["lr"]        = args.lr
    if args.device: cfg["training"]["device"]    = args.device
    if args.seed:   cfg["data"]["seed"]          = args.seed

    train(cfg)
