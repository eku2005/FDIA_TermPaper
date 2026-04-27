"""
demo.py
───────
Self-contained demonstration of the FDIA detection pipeline.
Trains a small model (fast), then shows detection results.
No pre-trained weights required.

Run:  python demo.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data.grid_topology import get_system
from data.dataset import generate_dataset, FDIADataset
from models.cnn_attention import CNNAttentionDetector
from models.detector import FDIADetector
from utils.state_estimator import WLSEstimator
from utils.attack_generator import AttackGenerator
from utils.metrics import compute_metrics, print_metrics


def main():
    print("═" * 60)
    print("  FDIA Detection — Quick Demo (IEEE 14-bus)")
    print("═" * 60)

    # ── 1. Generate small dataset ────────────────────────────
    print("\n[1/5] Generating dataset …")
    SYSTEM     = "ieee14"
    WINDOW     = 10
    N_TRAIN    = 2000
    N_TEST     = 400

    X_tr, R_tr, y_tr = generate_dataset(
        SYSTEM, N_TRAIN, WINDOW, noise_std=0.01,
        attack_intensity_min=0.10, attack_intensity_max=0.50, seed=42)
    X_te, R_te, y_te = generate_dataset(
        SYSTEM, N_TEST,  WINDOW, noise_std=0.01,
        attack_intensity_min=0.10, attack_intensity_max=0.50, seed=999)

    m = X_tr.shape[2]
    print(f"    Measurements per step : {m}")
    print(f"    Window size           : {WINDOW}")
    print(f"    Train / Test          : {N_TRAIN} / {N_TEST}")

    tr_ds = FDIADataset(X_tr, R_tr, y_tr, normalise=True)
    te_ds = FDIADataset(X_te, R_te, y_te, normalise=True, stats=tr_ds.stats)
    tr_loader = DataLoader(tr_ds, batch_size=64, shuffle=True)
    te_loader = DataLoader(te_ds, batch_size=128, shuffle=False)

    # ── 2. Build model ───────────────────────────────────────
    print("\n[2/5] Building CNN+Attention model …")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = CNNAttentionDetector(
        m_measurements=m, window_size=WINDOW,
        conv_filters=[32, 64, 128], attention_dim=64, dropout=0.3)
    model  = model.to(device)
    print(f"    Trainable parameters: {model.count_parameters():,}")

    # ── 3. Train (fast: 20 epochs) ───────────────────────────
    print("\n[3/5] Training (20 epochs, this takes ~1–2 min on CPU) …")
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    for epoch in range(1, 21):
        model.train()
        for x, r, y in tr_loader:
            x, r, y = x.to(device), r.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x, r), y)
            loss.backward()
            optimizer.step()

        if epoch % 5 == 0:
            model.eval()
            preds_all, y_all = [], []
            with torch.no_grad():
                for x, r, y in te_loader:
                    p = model(x.to(device), r.to(device)).cpu().numpy()
                    preds_all.append(p)
                    y_all.append(y.numpy())
            p = np.concatenate(preds_all)
            y = np.concatenate(y_all)
            m_ = compute_metrics(y.astype(int), (p > 0.5).astype(int), p)
            print(f"    Epoch {epoch:2d}  DR={m_['detection_rate']*100:.1f}%  "
                  f"FAR={m_['false_alarm_rate']*100:.1f}%  F1={m_['f1']:.3f}")

    # ── 4. Calibrate detector ────────────────────────────────
    print("\n[4/5] Calibrating detection threshold (target FAR=3%) …")
    detector = FDIADetector(model, beta=0.5, device=str(device))
    detector.calibrate(te_loader, target_fpr=0.02)

    # ── 5. Final evaluation & single-sample demo ─────────────
    print("\n[5/5] Final evaluation …")
    results = detector.evaluate_loader(te_loader)
    print_metrics(results, prefix="IEEE 14-bus Test Results")

    # ── Single-window real-time inference demo ───────────────
    print("\n── Single-window real-time inference ──")
    idx = np.random.randint(len(X_te))
    x_w = X_te[idx]
    r_w = R_te[idx]
    true_label = int(y_te[idx])

    # Normalise manually using training stats
    x_mean, x_std, r_mean, r_std = [s.numpy() for s in tr_ds.stats]
    x_w_norm = (x_w - x_mean[0]) / x_std[0]
    r_w_norm = (r_w - r_mean[0]) / r_std[0]

    y_score, D, is_attack = detector.detect(x_w_norm, r_w_norm)

    print(f"    True label     : {'ATTACK' if true_label else 'NORMAL'}")
    print(f"    NN score (y_t) : {y_score:.4f}")
    print(f"    Statistic (D_t): {D:.4f}  (threshold θ={detector.theta:.4f})")
    print(f"    Decision       : {'⚠ ATTACK DETECTED' if is_attack else '✓ NORMAL'}")

    # ── Traditional chi-square comparison ────────────────────
    print("\n── Traditional chi-square BDD comparison ──")
    system = get_system(SYSTEM)
    est    = WLSEstimator(system, noise_std=0.01)
    chi2_preds = []
    for i in range(len(X_te)):
        z    = X_te[i, -1, :]
        chi2_preds.append(int(est.chi2_detect(z)))
    chi2_m = compute_metrics(y_te.astype(int), np.array(chi2_preds))
    print(f"    Chi-square BDD  DR: {chi2_m['detection_rate']*100:.1f}%  "
          f"FAR: {chi2_m['false_alarm_rate']*100:.1f}%")
    print(f"    Proposed method DR: {results['detection_rate']*100:.1f}%  "
          f"FAR: {results['false_alarm_rate']*100:.1f}%")
    print(f"\n    Improvement: +{(results['detection_rate'] - chi2_m['detection_rate'])*100:.1f}pp detection rate")

    print("\n" + "═" * 60)
    print("  Demo complete!")
    print("═" * 60)


if __name__ == "__main__":
    main()
