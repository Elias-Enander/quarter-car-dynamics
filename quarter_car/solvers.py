"""
Numerical time-integration solvers for first-order ODE systems.
"""

import numpy as np


def forward_euler(model, t_span: tuple[float, float], dt: float, v0: np.ndarray):
    """
    Integrates system using the explicit Forward Euler scheme:
        v_{n+1} = v_n + dt * f(t_n, v_n)
    """
    t0, t_end = t_span
    num_steps = int(np.ceil((t_end - t0) / dt))
    t_vec = np.linspace(t0, t_end, num_steps + 1)
    v_mat = np.zeros((num_steps + 1, len(v0)))
    v_mat[0] = v0

    for n in range(num_steps):
        t_curr = t_vec[n]
        v_curr = v_mat[n]
        dv = model.rhs(t_curr, v_curr)
        v_mat[n + 1] = v_curr + dt * dv

    return t_vec, v_mat


def implicit_trapezoidal(model, t_span: tuple[float, float], dt: float, v0: np.ndarray):
    """
    Integrates system using the unconditionally A-stable Implicit Trapezoidal rule:
        (I - 0.5 * dt * H) * v_{n+1} = (I + 0.5 * dt * H) * v_n + 0.5 * dt * (g_n + g_{n+1})
    """
    t0, t_end = t_span
    num_steps = int(np.ceil((t_end - t0) / dt))
    t_vec = np.linspace(t0, t_end, num_steps + 1)
    v_mat = np.zeros((num_steps + 1, len(v0)))
    v_mat[0] = v0

    dim = len(v0)
    identity_mat = np.eye(dim)
    H = model.H_mat
    p = model.p

    # Pre-factor LHS and RHS linear transformation matrices
    lhs_inv = np.linalg.inv(identity_mat - 0.5 * dt * H)
    rhs_mat = identity_mat + 0.5 * dt * H

    for n in range(num_steps):
        t_curr = t_vec[n]
        t_next = t_vec[n + 1]
        v_curr = v_mat[n]

        h_n, dh_n = model.road_excitation(t_curr)
        h_np1, dh_np1 = model.road_excitation(t_next)

        g_n = np.array([0.0, 0.0, 0.0, (p.k2 * h_n + p.c2 * dh_n) / p.m2])
        g_np1 = np.array([0.0, 0.0, 0.0, (p.k2 * h_np1 + p.c2 * dh_np1) / p.m2])

        g_trap = 0.5 * dt * (g_n + g_np1)
        v_mat[n + 1] = lhs_inv @ (rhs_mat @ v_curr + g_trap)

    return t_vec, v_mat