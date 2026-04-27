"""
attack_generator.py
────────────────────
Generate False Data Injection Attack (FDIA) vectors.

Paper §II-B — three attack strategies:

  Targeted : a = H Δx  where Δx targets specific state variables
  Random   : aᵢ ~ N(0, σ_a²)
  Sparse   : only k ≤ m measurements manipulated, using a = H c projected
             to sparsity ‖a‖₀ ≤ k

All attacks are scaled to a user-specified intensity (fraction of
nominal measurement magnitude).
"""

import numpy as np
from typing import Literal, Optional, Tuple


AttackType = Literal["targeted", "random", "sparse"]


class AttackGenerator:
    """
    Generates stealthy FDIA vectors for a given measurement system.

    Parameters
    ----------
    H          : Jacobian matrix (m × n_state), from GridSystem.H_matrix()
    intensity  : attack magnitude as fraction of nominal measurements (e.g. 0.3)
    sparse_k   : number of measurements manipulated in sparse attacks
    seed       : random seed
    """

    def __init__(
        self,
        H: np.ndarray,
        intensity: float = 0.30,
        sparse_k: int = 5,
        seed: Optional[int] = None,
    ):
        self.H         = H
        self.m, self.n = H.shape
        self.intensity = intensity
        self.sparse_k  = sparse_k
        self.rng       = np.random.default_rng(seed)

    # ----------------------------------------------------------
    #  Core stealthy attack: a = H c   →  bypasses chi-square
    # ----------------------------------------------------------

    def _stealthy_vector(self, c: np.ndarray) -> np.ndarray:
        """Construct a = Hc (eq. 6 in paper)."""
        return self.H @ c

    def _scale_to_intensity(
        self, a: np.ndarray, z_nominal: np.ndarray
    ) -> np.ndarray:
        """Scale attack vector so ‖a‖ ≈ intensity × ‖z_nominal‖."""
        norm_a = np.linalg.norm(a)
        norm_z = np.linalg.norm(z_nominal)
        if norm_a < 1e-10:
            return a
        target_norm = self.intensity * norm_z
        return a * (target_norm / norm_a)

    # ----------------------------------------------------------
    #  Attack types
    # ----------------------------------------------------------

    def targeted_attack(
        self,
        z_nominal: np.ndarray,
        target_states: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Targeted attack: inject Δx into specific state variables.
        c = Δx with only the targeted state indices non-zero.
        """
        c = np.zeros(self.n)
        if target_states is None:
            # Randomly choose 1-3 state variables to target
            n_target = self.rng.integers(1, min(4, self.n))
            target_states = self.rng.choice(self.n, n_target, replace=False)

        # Δx magnitudes: intensity × 1.0 (normalised state)
        c[target_states] = self.rng.uniform(
            -self.intensity, self.intensity, len(target_states)
        )
        a = self._stealthy_vector(c)
        return self._scale_to_intensity(a, z_nominal)

    def random_attack(self, z_nominal):
        sigma_a = self.intensity * np.std(z_nominal)
        a = self.rng.normal(0, sigma_a, self.m)
        return a

    def sparse_attack(self, z_nominal: np.ndarray) -> np.ndarray:
        """
        Sparse stealthy attack: a = Hc but retain only top-k entries.
        Projects onto sparsity by zeroing out all but k largest |aᵢ|.
        """
        # Random c → stealthy a
        c = self.rng.normal(0, 1, self.n)
        a = self._stealthy_vector(c)
        a = self._scale_to_intensity(a, z_nominal)

        # Hard-threshold to k measurements
        k = min(self.sparse_k, self.m)
        mask = np.zeros(self.m, dtype=bool)
        top_k = np.argsort(np.abs(a))[-k:]
        mask[top_k] = True
        a_sparse = np.where(mask, a, 0.0)
        return a_sparse

    # ----------------------------------------------------------
    #  Unified interface
    # ----------------------------------------------------------

    def generate(
        self,
        z_nominal: np.ndarray,
        attack_type: AttackType = "targeted",
        intensity: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate corrupted measurements za = z + a.

        Returns
        -------
        za : corrupted measurement vector (m,)
        a  : attack vector              (m,)
        """
        if intensity is not None:
            old = self.intensity
            self.intensity = intensity

        if attack_type == "targeted":
            a = self.targeted_attack(z_nominal)
        elif attack_type == "random":
            a = self.random_attack(z_nominal)
        elif attack_type == "sparse":
            a = self.sparse_attack(z_nominal)
        else:
            raise ValueError(f"Unknown attack type: {attack_type}")

        if intensity is not None:
            self.intensity = old

        za = z_nominal + a
        return za, a

    def batch_generate(
        self,
        z_batch: np.ndarray,
        attack_type: AttackType = "targeted",
        intensity: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate attacks for a batch of measurement vectors.

        Parameters
        ----------
        z_batch : (N, m) array of nominal measurements

        Returns
        -------
        za_batch : (N, m)
        a_batch  : (N, m)
        """
        N = z_batch.shape[0]
        za_batch = np.zeros_like(z_batch)
        a_batch  = np.zeros_like(z_batch)
        for i in range(N):
            za_batch[i], a_batch[i] = self.generate(
                z_batch[i], attack_type, intensity
            )
        return za_batch, a_batch
