"""
detector.py
───────────
Full detection pipeline combining:
  1. CNNAttentionDetector (neural network score y_t)
  2. Residual-based statistic (eq. 12-14)

Detection statistic:
  Δr_t = r_t - r_{t-1}                              (12)
  D_t  = y_t · (1 + β ‖Δr_t‖₂)                     (13)
  attack iff D_t > θ                                 (14)

θ is calibrated on a held-out calibration set to achieve
the target false alarm rate (default: 3%).
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

from models.cnn_attention import CNNAttentionDetector


class FDIADetector:
    """
    End-to-end FDIA detection system.

    Usage
    -----
    detector = FDIADetector(model, beta=0.5, device='cpu')
    detector.calibrate(calib_loader, target_fpr=0.03)

    # Inference on a single window
    y, D, is_attack = detector.detect(x_window, r_window)
    """

    def __init__(
        self,
        model: CNNAttentionDetector,
        beta: float = 0.05,
        device: str = "cpu",
    ):
        self.model  = model.to(device)
        self.beta   = beta
        self.device = device
        self.theta  = 0.5   # default; update with calibrate()

    # ----------------------------------------------------------
    #  Detection statistic (eqs. 12-13)
    # ----------------------------------------------------------

    @staticmethod
    def residual_delta(r_window: torch.Tensor) -> torch.Tensor:
        """
        Compute ‖Δr_t‖₂ for the last two time steps in the window.
        r_window : (batch, w, m)
        Returns  : (batch,)
        """
        delta = r_window[:, -1, :] - r_window[:, -2, :]   # (batch, m)
        return torch.norm(delta, dim=-1)                    # (batch,)

    def compute_statistic(
        self,
        x: torch.Tensor,
        r: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute (y_t, D_t) for a batch.

        Returns
        -------
        y : (batch,)  neural network attack probability
        D : (batch,)  combined detection statistic
        """
        self.model.eval()
        with torch.no_grad():
            x = x.to(self.device)
            r = r.to(self.device)
            y      = self.model(x, r)                   # (batch,)
            delta  = self.residual_delta(r)              # (batch,)
            D      = y * (1.0 + self.beta * delta)       # (batch,)
        return y.cpu(), D.cpu()
    

    # ----------------------------------------------------------
    #  Threshold calibration
    # ----------------------------------------------------------

    def calibrate(
        self,
        calib_loader,
        target_fpr: float = 0.03,
    ) -> float:
        """
        Set detection threshold θ from calibration data.
        Uses the (1-target_fpr) quantile of D on NORMAL samples.

        Returns the chosen θ.
        """
        D_normal = []
        self.model.eval()
        with torch.no_grad():
            for x, r, y in calib_loader:
                mask = (y == 0)
                if mask.sum() == 0:
                    continue
                _, D = self.compute_statistic(x[mask], r[mask])
                D_normal.append(D.numpy())

        if not D_normal:
            return self.theta

        D_all     = np.concatenate(D_normal)
        self.theta = float(np.quantile(D_all, 1.0 - target_fpr))
        print(f"[Detector] Calibrated θ = {self.theta:.4f} "
              f"(target FPR={target_fpr:.1%})")
        return self.theta

    # ----------------------------------------------------------
    #  Batch evaluation
    # ----------------------------------------------------------

    def predict_batch(
        self, x: torch.Tensor, r: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (predictions, D_values).
        predictions : (batch,) int array — 0 normal, 1 attack
        """
        _, D   = self.compute_statistic(x, r)
        preds  = (D.numpy() > self.theta).astype(int)
        return preds, D.numpy()

    def evaluate_loader(self, loader) -> dict:
        """
        Full evaluation on a DataLoader.

        Returns dict with: detection_rate, false_alarm_rate,
                           f1, accuracy, D_values, labels
        """
        all_preds, all_labels, all_D = [], [], []

        for x, r, y in loader:
            preds, D = self.predict_batch(x, r)
            all_preds.append(preds)
            all_labels.append(y.numpy().astype(int))
            all_D.append(D)

        preds  = np.concatenate(all_preds)
        labels = np.concatenate(all_labels)
        D_vals = np.concatenate(all_D)

        # Metrics
        tp = np.sum((preds == 1) & (labels == 1))
        fp = np.sum((preds == 1) & (labels == 0))
        tn = np.sum((preds == 0) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))

        dr  = tp / (tp + fn + 1e-9)   # detection rate (recall)
        far = fp / (fp + tn + 1e-9)   # false alarm rate
        pre = tp / (tp + fp + 1e-9)   # precision
        f1  = 2 * pre * dr / (pre + dr + 1e-9)
        acc = (tp + tn) / len(labels)

        return {
            "detection_rate":  dr,
            "false_alarm_rate": far,
            "f1":              f1,
            "accuracy":        acc,
            "tp": int(tp), "fp": int(fp),
            "tn": int(tn), "fn": int(fn),
            "D_values": D_vals,
            "labels":   labels,
        }

    # ----------------------------------------------------------
    #  Single-window real-time inference
    # ----------------------------------------------------------

    def detect(
        self,
        x_window: np.ndarray,
        r_window: np.ndarray,
    ) -> Tuple[float, float, bool]:
        """
        Detect attack on a single measurement window.

        Parameters
        ----------
        x_window : (window_size, m) numpy array
        r_window : (window_size, m) numpy array

        Returns
        -------
        y_t      : neural network score
        D_t      : combined statistic
        is_attack: bool
        """
        x = torch.from_numpy(x_window.astype(np.float32)).unsqueeze(0)
        r = torch.from_numpy(r_window.astype(np.float32)).unsqueeze(0)
        y_t, D_t = self.compute_statistic(x, r)
        return float(y_t[0]), float(D_t[0]), bool(D_t[0] > self.theta)

    # ----------------------------------------------------------
    #  Save / Load
    # ----------------------------------------------------------

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "theta":       self.theta,
            "beta":        self.beta,
        }, path)
        print(f"[Detector] Saved to {path}")

    @classmethod
    def load(
        cls,
        path: str,
        model: CNNAttentionDetector,
        device: str = "cpu",
    ) -> "FDIADetector":
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        det       = cls(model, beta=ckpt["beta"], device=device)
        det.theta = ckpt["theta"]
        print(f"[Detector] Loaded from {path}, θ={det.theta:.4f}")
        return det
