"""
state_estimator.py
──────────────────
Weighted Least Squares (WLS) state estimator and
traditional chi-square Bad Data Detection (BDD).

Paper equations:
  z = h(x) + e                          (1)
  x̂ = argmin [z-h(x)]ᵀ R⁻¹ [z-h(x)]   (2)
  r = z - h(x̂)                          (3)
  ‖r‖²_R = rᵀ R⁻¹ r > τ                (4)  ← traditional detector
"""

import numpy as np
from data.grid_topology import GridSystem


class WLSEstimator:
    """
    Linearised WLS estimator using the DC approximation.

    States  x : [θ₁…θ_{n-1}, |V|₀…|V|_{n-1}]   (n-1 + n = 2n-1 values)
    Measurements z : [P_flow (n_br), P_inj (n_bus), |V| (n_bus)]
    """

    def __init__(self, system: GridSystem, noise_std: float = 0.01):
        self.system   = system
        self.H        = system.H_matrix()          # (m × n_state)
        self.m        = self.H.shape[0]
        self.n_state  = self.H.shape[1]
        self.noise_std = noise_std

        # Measurement covariance R = diag(σᵢ²)
        self._build_R()

        # Pre-compute gain matrix G = HᵀR⁻¹H and its inverse
        self.R_inv   = np.diag(1.0 / np.diag(self.R))
        self.G       = self.H.T @ self.R_inv @ self.H
        self.G_inv   = np.linalg.pinv(self.G)

    # ----------------------------------------------------------
    #  Internal helpers
    # ----------------------------------------------------------

    def _build_R(self):
        """Diagonal covariance with 1% noise on all measurements."""
        # Nominal measurement values are ~1 pu, so σ = noise_std * 1.0
        sigmas    = np.full(self.m, self.noise_std)
        self.R    = np.diag(sigmas ** 2)

    def _true_measurements(self, x_true: np.ndarray) -> np.ndarray:
        """z_true = H x_true  (DC linearisation around flat start)."""
        return self.H @ x_true

    def _add_noise(self, z_true: np.ndarray) -> np.ndarray:
        """Add Gaussian measurement noise."""
        noise = np.random.normal(0, self.noise_std, size=z_true.shape)
        return z_true + noise

    # ----------------------------------------------------------
    #  Public API
    # ----------------------------------------------------------

    def generate_measurements(self, x_true: np.ndarray) -> np.ndarray:
        """Return noisy measurements for a given state vector."""
        return self._add_noise(self._true_measurements(x_true))

    def estimate(self, z: np.ndarray) -> np.ndarray:
        """
        WLS estimate: x̂ = (HᵀR⁻¹H)⁻¹ HᵀR⁻¹ z
        Returns estimated state vector.
        """
        rhs   = self.H.T @ self.R_inv @ z
        x_hat = self.G_inv @ rhs
        return x_hat

    def residual(self, z: np.ndarray, x_hat: np.ndarray) -> np.ndarray:
        """Measurement residual r = z - Hx̂."""
        return z - self.H @ x_hat

    def residual_norm(self, r: np.ndarray) -> float:
        """Weighted residual norm ‖r‖²_R = rᵀ R⁻¹ r."""
        return float(r @ self.R_inv @ r)

    def chi2_detect(self, z: np.ndarray, threshold: float = None) -> bool:
        """
        Traditional chi-square bad data detection.
        Default threshold = chi2 critical value at 99% CL with m - n_state DOF.
        Returns True if attack detected.
        """
        from scipy.stats import chi2
        x_hat = self.estimate(z)
        r     = self.residual(z, x_hat)
        stat  = self.residual_norm(r)
        if threshold is None:
            dof       = self.m - self.n_state
            threshold = chi2.ppf(0.99, df=max(dof, 1))
        return stat > threshold

    def default_state(self) -> np.ndarray:
        """Flat-start state vector: all angles 0, all voltages 1 pu."""
        n   = self.system.n_bus
        x   = np.zeros(self.n_state)
        # voltage part (indices n-1 to 2n-2)
        x[n - 1:] = 1.0
        return x

    def random_state(self, rng: np.random.Generator = None) -> np.ndarray:
        """Random operating point close to flat start."""
        if rng is None:
            rng = np.random.default_rng()
        n     = self.system.n_bus
        x     = np.zeros(self.n_state)
        # Angles: ±15°
        x[:n - 1] = rng.uniform(-0.26, 0.26, n - 1)
        # Voltages: 0.95 – 1.05 pu
        x[n - 1:] = rng.uniform(0.95, 1.05, n)
        return x
