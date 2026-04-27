"""
grid_topology.py
────────────────
IEEE 14-bus, 30-bus, and 57-bus test system data.

Each system provides:
  - bus_data   : [bus_id, type, Pd, Qd, Gs, Bs, Vm, Va, base_kv]
  - branch_data: [from, to, r, x, b, rateA]
  - Ybus()     : admittance matrix (complex, n×n)
  - H_matrix() : measurement Jacobian linearised at flat start

References:
  Power Systems Test Case Archive — U. of Washington
  https://labs.ece.uw.edu/pstca/
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple


# ─────────────────────────────────────────────────────────────
#  Data containers
# ─────────────────────────────────────────────────────────────

@dataclass
class BusData:
    """Per-bus data."""
    bus_id: int
    bus_type: int   # 1=PQ, 2=PV, 3=slack
    Pd: float       # active load (pu)
    Qd: float       # reactive load (pu)
    Vm: float = 1.0
    Va: float = 0.0

@dataclass
class BranchData:
    """Transmission line / transformer."""
    fbus: int
    tbus: int
    r: float        # resistance (pu)
    x: float        # reactance (pu)
    b: float = 0.0  # total line charging susceptance

@dataclass
class GridSystem:
    name: str
    buses: List[BusData]
    branches: List[BranchData]

    @property
    def n_bus(self) -> int:
        return len(self.buses)

    @property
    def n_branch(self) -> int:
        return len(self.branches)

    # ----------------------------------------------------------
    #  Admittance matrix
    # ----------------------------------------------------------
    def Ybus(self) -> np.ndarray:
        n = self.n_bus
        Y = np.zeros((n, n), dtype=complex)
        # map bus_id → 0-indexed
        idx = {b.bus_id: i for i, b in enumerate(self.buses)}
        for br in self.branches:
            i, j = idx[br.fbus], idx[br.tbus]
            y_series = 1.0 / complex(br.r, br.x) if (br.r != 0 or br.x != 0) else 0
            y_shunt  = complex(0, br.b / 2)
            Y[i, i] += y_series + y_shunt
            Y[j, j] += y_series + y_shunt
            Y[i, j] -= y_series
            Y[j, i] -= y_series
        return Y

    # ----------------------------------------------------------
    #  Linearised DC measurement Jacobian H  (m × n_state)
    #  Measurements: [P_flows (n_branch), P_injections (n_bus),
    #                 |V| magnitudes   (n_bus)]
    #  States      : [theta (n_bus-1), |V| (n_bus)]
    # ----------------------------------------------------------
    def H_matrix(self) -> np.ndarray:
        n = self.n_bus
        n_br = self.n_branch
        m = n_br + 2 * n          # total measurements
        # n_state = (n-1) angles + n voltages
        n_state = (n - 1) + n
        H = np.zeros((m, n_state))

        idx = {b.bus_id: i for i, b in enumerate(self.buses)}
        # Slack bus is first bus (index 0), its angle is fixed → excluded from states
        # State layout: [theta_1 … theta_{n-1}, |V|_0 … |V|_{n-1}]
        angle_offset = 0
        volt_offset  = n - 1

        for k, br in enumerate(self.branches):
            i, j = idx[br.fbus], idx[br.tbus]
            b_ij = 1.0 / br.x if br.x != 0 else 1e6
            # dP_flow/d_theta_i
            if i > 0:
                H[k, angle_offset + i - 1] =  b_ij
            if j > 0:
                H[k, angle_offset + j - 1] = -b_ij

        for i in range(n):
            row = n_br + i
            # dP_inj/d_theta: sum over neighbours
            for br in self.branches:
                fi, ti = idx[br.fbus], idx[br.tbus]
                b_ij = 1.0 / br.x if br.x != 0 else 1e6
                if fi == i and ti > 0:
                    H[row, angle_offset + ti - 1] -= b_ij
                elif ti == i and fi > 0:
                    H[row, angle_offset + fi - 1] -= b_ij
            # dP_inj/d_theta_i (diagonal)
            if i > 0:
                H[row, angle_offset + i - 1] = -H[row, angle_offset + i - 1]

        # Voltage magnitude block: identity
        for i in range(n):
            H[n_br + n + i, volt_offset + i] = 1.0

        return H

    @property
    def n_measurements(self) -> int:
        return self.n_branch + 2 * self.n_bus

    @property
    def n_states(self) -> int:
        return (self.n_bus - 1) + self.n_bus


# ─────────────────────────────────────────────────────────────
#  IEEE 14-bus system
# ─────────────────────────────────────────────────────────────

def ieee14() -> GridSystem:
    buses = [
        BusData(1,  3, 0.000, 0.000, 1.060, 0.000),
        BusData(2,  2, 0.217, 0.127, 1.045, -4.98),
        BusData(3,  2, 0.942, 0.190, 1.010, -12.72),
        BusData(4,  1, 0.478, -0.039, 1.019, -10.33),
        BusData(5,  1, 0.076, 0.016, 1.020, -8.78),
        BusData(6,  2, 0.112, 0.075, 1.070, -14.22),
        BusData(7,  1, 0.000, 0.000, 1.062, -13.37),
        BusData(8,  2, 0.000, 0.000, 1.090, -13.36),
        BusData(9,  1, 0.295, 0.166, 1.056, -14.94),
        BusData(10, 1, 0.090, 0.058, 1.051, -15.10),
        BusData(11, 1, 0.035, 0.018, 1.057, -14.79),
        BusData(12, 1, 0.061, 0.016, 1.055, -15.07),
        BusData(13, 1, 0.135, 0.058, 1.050, -15.16),
        BusData(14, 1, 0.149, 0.050, 1.036, -16.04),
    ]
    branches = [
        BranchData(1,  2,  0.01938, 0.05917, 0.0528),
        BranchData(1,  5,  0.05403, 0.22304, 0.0492),
        BranchData(2,  3,  0.04699, 0.19797, 0.0438),
        BranchData(2,  4,  0.05811, 0.17632, 0.0374),
        BranchData(2,  5,  0.05695, 0.17388, 0.0340),
        BranchData(3,  4,  0.06701, 0.17103, 0.0346),
        BranchData(4,  5,  0.01335, 0.04211, 0.0128),
        BranchData(4,  7,  0.00000, 0.20912, 0.0000),
        BranchData(4,  9,  0.00000, 0.55618, 0.0000),
        BranchData(5,  6,  0.00000, 0.25202, 0.0000),
        BranchData(6,  11, 0.09498, 0.19890, 0.0000),
        BranchData(6,  12, 0.12291, 0.25581, 0.0000),
        BranchData(6,  13, 0.06615, 0.13027, 0.0000),
        BranchData(7,  8,  0.00000, 0.17615, 0.0000),
        BranchData(7,  9,  0.00000, 0.11001, 0.0000),
        BranchData(9,  10, 0.03181, 0.08450, 0.0000),
        BranchData(9,  14, 0.12711, 0.27038, 0.0000),
        BranchData(10, 11, 0.08205, 0.19207, 0.0000),
        BranchData(12, 13, 0.22092, 0.19988, 0.0000),
        BranchData(13, 14, 0.17093, 0.34802, 0.0000),
    ]
    return GridSystem("IEEE 14-bus", buses, branches)


# ─────────────────────────────────────────────────────────────
#  IEEE 30-bus system
# ─────────────────────────────────────────────────────────────

def ieee30() -> GridSystem:
    buses = [
        BusData(1,  3, 0.000, 0.000),
        BusData(2,  2, 0.217, 0.127),
        BusData(3,  1, 0.024, 0.012),
        BusData(4,  1, 0.076, 0.016),
        BusData(5,  2, 0.942, 0.190),
        BusData(6,  1, 0.000, 0.000),
        BusData(7,  1, 0.228, 0.109),
        BusData(8,  2, 0.300, 0.300),
        BusData(9,  1, 0.000, 0.000),
        BusData(10, 1, 0.058, 0.020),
        BusData(11, 2, 0.000, 0.000),
        BusData(12, 1, 0.112, 0.075),
        BusData(13, 2, 0.000, 0.000),
        BusData(14, 1, 0.062, 0.016),
        BusData(15, 1, 0.082, 0.025),
        BusData(16, 1, 0.035, 0.018),
        BusData(17, 1, 0.090, 0.058),
        BusData(18, 1, 0.032, 0.009),
        BusData(19, 1, 0.095, 0.034),
        BusData(20, 1, 0.022, 0.007),
        BusData(21, 1, 0.175, 0.112),
        BusData(22, 1, 0.000, 0.000),
        BusData(23, 1, 0.032, 0.016),
        BusData(24, 1, 0.087, 0.067),
        BusData(25, 1, 0.000, 0.000),
        BusData(26, 1, 0.035, 0.023),
        BusData(27, 1, 0.000, 0.000),
        BusData(28, 1, 0.000, 0.000),
        BusData(29, 1, 0.024, 0.009),
        BusData(30, 1, 0.106, 0.019),
    ]
    branches = [
        BranchData(1,  2,  0.0192, 0.0575),
        BranchData(1,  3,  0.0452, 0.1852),
        BranchData(2,  4,  0.0570, 0.1737),
        BranchData(3,  4,  0.0132, 0.0379),
        BranchData(2,  5,  0.0472, 0.1983),
        BranchData(2,  6,  0.0581, 0.1763),
        BranchData(4,  6,  0.0119, 0.0414),
        BranchData(5,  7,  0.0460, 0.1160),
        BranchData(6,  7,  0.0267, 0.0820),
        BranchData(6,  8,  0.0120, 0.0420),
        BranchData(6,  9,  0.0000, 0.2080),
        BranchData(6,  10, 0.0000, 0.5560),
        BranchData(9,  11, 0.0000, 0.2080),
        BranchData(9,  10, 0.0000, 0.1100),
        BranchData(4,  12, 0.0000, 0.2560),
        BranchData(12, 13, 0.0000, 0.1400),
        BranchData(12, 14, 0.1231, 0.2559),
        BranchData(12, 15, 0.0662, 0.1304),
        BranchData(12, 16, 0.0945, 0.1987),
        BranchData(14, 15, 0.2210, 0.1997),
        BranchData(16, 17, 0.0524, 0.1923),
        BranchData(15, 18, 0.1073, 0.2185),
        BranchData(18, 19, 0.0639, 0.1292),
        BranchData(19, 20, 0.0340, 0.0680),
        BranchData(10, 20, 0.0936, 0.2090),
        BranchData(10, 17, 0.0324, 0.0845),
        BranchData(10, 21, 0.0348, 0.0749),
        BranchData(10, 22, 0.0727, 0.1499),
        BranchData(21, 22, 0.0116, 0.0236),
        BranchData(15, 23, 0.1000, 0.2020),
        BranchData(22, 24, 0.1150, 0.1790),
        BranchData(23, 24, 0.1320, 0.2700),
        BranchData(24, 25, 0.1885, 0.3292),
        BranchData(25, 26, 0.2544, 0.3800),
        BranchData(25, 27, 0.1093, 0.2087),
        BranchData(28, 27, 0.0000, 0.3960),
        BranchData(27, 29, 0.2198, 0.4153),
        BranchData(27, 30, 0.3202, 0.6027),
        BranchData(29, 30, 0.2399, 0.4533),
        BranchData(8,  28, 0.0636, 0.2000),
        BranchData(6,  28, 0.0169, 0.0599),
    ]
    return GridSystem("IEEE 30-bus", buses, branches)


# ─────────────────────────────────────────────────────────────
#  IEEE 57-bus — compact representation (key branches only)
#  (full 80-branch version kept for realism)
# ─────────────────────────────────────────────────────────────

# def ieee57() -> GridSystem:
#     """Simplified IEEE 57-bus for generalization tests."""
#     # Using 57 buses with representative branch subset
#     buses = [BusData(i, 1 if i > 7 else (3 if i == 1 else 2), 0.05*i % 0.5, 0.02*i % 0.2)
#              for i in range(1, 58)]
#     buses[0].bus_type = 3  # slack
#     branches = [
#         BranchData(1, 2,  0.0083, 0.0280), BranchData(2, 3,  0.0298, 0.0850),
#         BranchData(3, 4,  0.0112, 0.0366), BranchData(4, 5,  0.0625, 0.1320),
#         BranchData(4, 6,  0.0430, 0.1480), BranchData(6, 7,  0.0200, 0.1020),
#         BranchData(6, 8,  0.0339, 0.1730), BranchData(8, 9,  0.0099, 0.0505),
#         BranchData(9, 10, 0.0369, 0.1679), BranchData(9, 11, 0.0258, 0.0848),
#         BranchData(9, 12, 0.0648, 0.2950), BranchData(9, 13, 0.0481, 0.1580),
#         BranchData(13,14, 0.0132, 0.0434), BranchData(13,15, 0.0269, 0.0869),
#         BranchData(1, 15, 0.0178, 0.0910), BranchData(1, 16, 0.0454, 0.2060),
#         BranchData(1, 17, 0.0238, 0.1080), BranchData(3, 15, 0.0162, 0.0530),
#         BranchData(4, 18, 0.0000, 0.5550), BranchData(4, 18, 0.0000, 0.4300),
#         BranchData(5, 6,  0.0302, 0.0641), BranchData(7, 8,  0.0139, 0.0712),
#         BranchData(10,12, 0.0277, 0.1262), BranchData(11,13, 0.0223, 0.0732),
#         BranchData(14,15, 0.0171, 0.0547), BranchData(18,19, 0.4610, 0.6850),
#         BranchData(19,20, 0.2830, 0.4340), BranchData(21,20, 0.0000, 0.7767),
#         BranchData(21,22, 0.0736, 0.1170), BranchData(22,23, 0.0099, 0.0152),
#         BranchData(23,24, 0.1660, 0.2560), BranchData(24,25, 0.0000, 1.1820),
#         BranchData(24,25, 0.0000, 1.2300), BranchData(24,26, 0.0000, 0.0473),
#         BranchData(26,27, 0.1650, 0.2540), BranchData(27,28, 0.0618, 0.0954),
#         BranchData(28,29, 0.0418, 0.0587), BranchData(7, 29, 0.0000, 0.0648),
#         BranchData(25,30, 0.1350, 0.2020), BranchData(30,31, 0.3260, 0.4970),
#         BranchData(23,32, 0.0507, 0.0755), BranchData(31,32, 0.0392, 0.0360),
#         BranchData(32,33, 0.0369, 0.0399), BranchData(34,32, 0.0000, 0.9530),
#         BranchData(34,35, 0.0086, 0.0860), BranchData(35,36, 0.0000, 0.0174),
#         BranchData(36,37, 0.0580, 0.0427), BranchData(37,38, 0.0289, 0.1420),
#         BranchData(37,39, 0.0671, 0.2000), BranchData(36,40, 0.0712, 0.2000),
#     ]
#     return GridSystem("IEEE 57-bus", buses, branches)


# ─────────────────────────────────────────────────────────────
#  Factory
# ─────────────────────────────────────────────────────────────

def get_system(name: str) -> GridSystem:
    mapping = {"ieee14": ieee14, "ieee30": ieee30} #, "ieee57": ieee57}
    if name not in mapping:
        raise ValueError(f"Unknown system '{name}'. Choose from {list(mapping)}")
    return mapping[name]()
