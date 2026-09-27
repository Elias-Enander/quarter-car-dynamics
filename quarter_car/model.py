"""
Dynamic state-space model for a 2-DOF Quarter-Car suspension system.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class SuspensionParameters:
    """Quarter-car physical and geometric parameters."""
    m1: float = 465.0       # Sprung mass (chassis) [kg]
    m2: float = 55.0        # Unsprung mass (wheel assembly) [kg]
    c1: float = 310.0       # Suspension damping coefficient [Ns/m]
    c2: float = 1250.0      # Tire damping coefficient [Ns/m]
    k1: float = 5350.0      # Suspension spring stiffness (baseline) [N/m]
    k2: float = 136100.0    # Tire spring stiffness (baseline) [N/m]
    H: float = 0.27         # Road bump amplitude [m]
    L: float = 1.1          # Road bump length [m]
    v_car: float = 63 / 3.6 # Longitudinal vehicle velocity [m/s]


class QuarterCarModel:
    """
    Formulates equations of motion into state-space representation:
        dv/dt = H * v(t) + g(t)
    where state vector v = [z1, z2, dz1/dt, dz2/dt]^T.
    """

    def __init__(self, params: SuspensionParameters):
        self.p = params
        self.H_mat = self._assemble_system_matrix()

    def _assemble_system_matrix(self) -> np.ndarray:
        """Assembles the 4x4 state-space transition matrix H."""
        p = self.p
        return np.array([
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [-p.k1 / p.m1, p.k1 / p.m1, -p.c1 / p.m1, p.c1 / p.m1],
            [p.k1 / p.m2, -(p.k1 + p.k2) / p.m2, p.c1 / p.m2, -(p.c1 + p.c2) / p.m2]
        ])

    def road_excitation(self, t: float) -> tuple[float, float]:
        """
        Computes road elevation h(t) and vertical road velocity h_dot(t).
        Models a single sinus like bump over time duration L / v_car.
        """
        p = self.p
        bump_duration = p.L / p.v_car

        if t <= bump_duration:
            omega = 2.0 * np.pi * p.v_car / p.L
            h = (p.H / 2.0) * (1.0 - np.cos(omega * t))
            h_dot = (p.H / 2.0) * omega * np.sin(omega * t)
            return h, h_dot

        return 0.0, 0.0

    def rhs(self, t: float, v: np.ndarray) -> np.ndarray:
        """Right-hand side evaluator for the ODE system: dv/dt = H * v + g(t)."""
        h, h_dot = self.road_excitation(t)
        g_vec = np.array([0.0, 0.0, 0.0, (self.p.k2 * h + self.p.c2 * h_dot) / self.p.m2])
        return self.H_mat @ v + g_vec

    def compute_forward_euler_stability_limit(self) -> float:
        """
        Computes the theoretical maximum time-step dt_max for Forward Euler stability
        based on the eigenvalues of the system matrix:
            dt_max = min_k (-2 * Re(lambda_k) / |lambda_k|^2)
        """
        eigenvalues = np.linalg.eigvals(self.H_mat)
        dt_candidates = []
        for lam in eigenvalues:
            alpha = np.real(lam)
            beta = np.imag(lam)
            if alpha < 0.0:
                dt_candidates.append(-2.0 * alpha / (alpha**2 + beta**2))

        return min(dt_candidates) if dt_candidates else float("inf")