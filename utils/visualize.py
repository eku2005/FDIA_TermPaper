"""
visualize.py
────────────
Reproduces the four key figures from the paper plus training curves.

Fig 1: Detection rate vs attack intensity, FAR by attack type
Fig 2: Method comparison bar chart
Fig 3: Temporal detection statistic (normal vs attack)
Fig 4: Training convergence + generalisation
Fig 5: Sensitivity — attack magnitude & noise level
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend safe for scripts
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Dict, List, Optional


COLORS = {
    "targeted": "#1f77b4",
    "random":   "#ff7f0e",
    "sparse":   "#2ca02c",
    "proposed": "#d62728",
    "normal":   "#2ca02c",
    "attack":   "#d62728",
}

FIG_DIR = Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────
#  Fig 1: Detection rate vs intensity + FAR by type
# ─────────────────────────────────────────────────────────────

def plot_fig1(results_by_type, save=True):
    fig, axes = plt.subplots(1,2,figsize=(11,4.5))

    # (a) Detection Rate vs intensity
    ax = axes[0]

    for atype,data in results_by_type.items():
        intensities = sorted(data.keys())
        drs = [data[i]["detection_rate"]*100 for i in intensities]

        ax.plot(
            [i*100 for i in intensities],
            drs,
            marker="o",
            linewidth=2,
            label=atype.capitalize(),
            color=COLORS.get(atype,"gray")
        )

    ax.set_xlabel("Attack Intensity (%)")
    ax.set_ylabel("Detection Rate (%)")
    ax.set_title("(a) Detection Rate vs Attack Intensity")
    ax.set_ylim(0,105)
    ax.legend()
    ax.grid(True,alpha=.3)


    # (b) FAR vs intensity  (better than fake FAR-by-type bars)
    ax = axes[1]

    for atype,data in results_by_type.items():
        intensities = sorted(data.keys())
        fars = [data[i]["false_alarm_rate"]*100 for i in intensities]

        ax.plot(
            [i*100 for i in intensities],
            fars,
            marker="s",
            linewidth=2,
            label=atype.capitalize(),
            color=COLORS.get(atype,"gray")
        )

    ax.axhline(
        3,
        linestyle="--",
        color="red",
        alpha=.7,
        label="3% target"
    )

    ax.set_xlabel("Attack Intensity (%)")
    ax.set_ylabel("False Alarm Rate (%)")
    ax.set_title("(b) FAR vs Attack Intensity")
    ax.set_ylim(0,10)
    ax.legend()
    ax.grid(True,alpha=.3)

    plt.tight_layout()

    if save:
        fig.savefig(
            FIG_DIR/"fig1_detection_vs_intensity.png",
            dpi=150
        )

    return fig

# ─────────────────────────────────────────────────────────────
#  Fig 2: Method comparison
# ─────────────────────────────────────────────────────────────

def plot_fig2(method_results: Dict, save: bool = True):
    """
    method_results : {method_name: {detection_rate, f1, specificity}, ...}
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    methods  = list(method_results.keys())
    metrics  = ["detection_rate", "f1", "specificity"]
    labels   = ["Detection Rate", "F1-Score", "Specificity"]
    n_m      = len(metrics)
    x        = np.arange(len(methods))
    width    = 0.25
    offsets  = np.linspace(-(n_m - 1) * width / 2, (n_m - 1) * width / 2, n_m)

    for j, (metric, label) in enumerate(zip(metrics, labels)):
        vals = [method_results[m].get(metric, 0) * 100 for m in methods]
        bars = ax.bar(x + offsets[j], vals, width=width, label=label, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.set_ylabel("Performance (%)", fontsize=11)
    ax.set_title("Comparison of Detection Methods")
    ax.set_ylim([60,102])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    if save:
        fig.savefig(FIG_DIR / "fig2_method_comparison.png", dpi=150)
    return fig


# ─────────────────────────────────────────────────────────────
#  Fig 3: Temporal detection statistic
# ─────────────────────────────────────────────────────────────

def plot_fig3(
    D_normal: np.ndarray,
    D_attack: np.ndarray,
    threshold: float,
    attack_start: int = 40,
    save: bool = True,
):
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    T = len(D_normal)
    ts = np.arange(T)

    # (a) Normal
    ax = axes[0]
    ax.plot(ts, D_normal, color=COLORS["normal"], linewidth=1.2)
    ax.axhline(threshold, ls="--", color="red", label=f"Threshold θ={threshold:.2f}")
    ax.set_ylabel("Detection Statistic $D_t$")
    ax.set_title("(a) Normal Operation")
    ax.legend(fontsize=9)
    ax.set_ylim([-0.05, max(1.1, D_normal.max() * 1.2)])
    ax.grid(True, alpha=0.3)

    # (b) Under attack
    ax = axes[1]
    ax.plot(ts, D_attack, color=COLORS["attack"], linewidth=1.2, label="Detection Statistic")
    ax.axhline(threshold, ls="--", color="gray", label=f"Threshold θ={threshold:.2f}")
    ax.axvspan(attack_start, T, alpha=0.15, color="red", label="Attack Period")
    ax.set_ylabel("Detection Statistic $D_t$")
    ax.set_xlabel("Time Step")
    ax.set_title("(b) Under Attack")
    ax.legend(fontsize=9)
    ax.set_ylim([-0.05, max(1.1, D_attack.max() * 1.1)])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        fig.savefig(FIG_DIR / "fig3_temporal_statistic.png", dpi=150)
    return fig


# ─────────────────────────────────────────────────────────────
#  Fig 4: Training convergence + generalisation
# ─────────────────────────────────────────────────────────────

def plot_fig4(
    train_losses: List[float],
    val_losses: List[float],
    generalization: Dict,   # {system_name: accuracy}
    save: bool = True,
):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # (a) Loss curves
    ax = axes[0]
    epochs = np.arange(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Training Loss",   color="#1f77b4")
    ax.plot(epochs, val_losses,   label="Validation Loss", color="#ff7f0e", ls="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("(a) Training Convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (b) Generalisation accuracy
    ax = axes[1]
    systems  = list(generalization.keys())
    accs     = [generalization[s] * 100 for s in systems]
    ax.bar(systems, accs, color="#2ca02c", alpha=0.8)
    for i, (s, a) in enumerate(zip(systems, accs)):
        ax.text(i, a + 0.3, f"{a:.1f}%", ha="center", va="bottom")
    ax.set_ylabel("Detection Accuracy (%)")
    ax.set_title("(b) Generalisation Performance")
    ax.set_ylim([0, 100])
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    if save:
        fig.savefig(FIG_DIR / "fig4_convergence_generalization.png", dpi=150)
    return fig


# ─────────────────────────────────────────────────────────────
#  Fig 5: Sensitivity analysis
# ─────────────────────────────────────────────────────────────

def plot_fig5(
    magnitudes,
    acc_proposed,
    acc_ml,
    acc_traditional,
    noise_levels,
    rob_proposed,
    rob_ml,
    rob_traditional,
    save=True
):
    fig,axes=plt.subplots(1,2,figsize=(12,5))

    all_vals=np.concatenate([
        acc_proposed,
        acc_ml,
        acc_traditional,
        rob_proposed,
        rob_ml,
        rob_traditional
    ])*100

    ymin=max(0,np.floor(all_vals.min()-5))
    ymax=min(100,np.ceil(all_vals.max()+5))


    ax=axes[0]
    ax.plot(magnitudes*100,acc_proposed*100,'o-',label="Proposed")
    ax.plot(magnitudes*100,acc_ml*100,'s--',label="ML Baseline")
    ax.plot(magnitudes*100,acc_traditional*100,'^:',label="Traditional")
    ax.set_ylim(ymin,ymax)
    ax.set_title("(a) Detection vs Attack Magnitude")
    ax.set_xlabel("Attack Magnitude (%)")
    ax.set_ylabel("Detection Accuracy (%)")
    ax.legend()
    ax.grid(True,alpha=.3)


    ax=axes[1]
    ax.plot(noise_levels*100,rob_proposed*100,'o-',label="Proposed")
    ax.plot(noise_levels*100,rob_ml*100,'s--',label="ML Baseline")
    ax.plot(noise_levels*100,rob_traditional*100,'^:',label="Traditional")
    ax.set_ylim(ymin,ymax)
    ax.set_title("(b) Robustness to Noise")
    ax.set_xlabel("Measurement Noise (%)")
    ax.set_ylabel("Detection Accuracy (%)")
    ax.legend()
    ax.grid(True,alpha=.3)

    plt.tight_layout()

    if save:
        fig.savefig(
            FIG_DIR/"fig5_sensitivity.png",
            dpi=150
        )

    return fig


# ─────────────────────────────────────────────────────────────
#  Utility: save training history CSV
# ─────────────────────────────────────────────────────────────

def save_training_log(history: Dict, path: str = "results/training_log.csv"):
    import csv
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    keys = list(history.keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        n = len(history[keys[0]])
        for i in range(n):
            w.writerow({k: history[k][i] for k in keys})
    print(f"[Log] Training history saved to {path}")
