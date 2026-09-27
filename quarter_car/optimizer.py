"""
Frequency-domain transfer function evaluation and multivariate Newton-Raphson optimization.
"""

import numpy as np


def transfer_functions_and_jacobian(k1: float, k2: float, ck: float = 0.7, cs: float = 1.0):
    """
    Evaluates residual vector F = [F_comfort; F_safety]^T and its 2x2 Jacobian matrix J
    with respect to parameters [k1, k2].
    """
    # Nominal system constants
    M, m = 465.0, 55.0
    B, C = 310.0, 1250.0
    k1_ref, k2_ref = 5350.0, 136100.0
    omega = 63.0 / 3.6

    def compute_spectral_magnitudes(K_val: float, k_val: float):
        real_D = (M * m) * omega**4 - (m * K_val + C * B + M * k_val + M * K_val) * omega**2 + K_val * k_val
        imag_D = -omega**3 * (B * m + C * M + M * B) + omega * (C * K_val + B * k_val)
        D_sq = real_D**2 + imag_D**2

        real_F1 = M * m * k_val * K_val * omega**4 - M * m * C * B * omega**6 + \
                  (m * K_val + C * B + M * k_val + M * K_val) * C * B * omega**4 - \
                  (m * K_val + C * B + M * k_val + M * K_val) * k_val * K_val * omega**2 + \
                  (-omega**2 * C * B + k_val * K_val) * k_val * K_val + omega**2 * (C * K_val + B * k_val)**2 - \
                  omega**4 * (C * K_val + B * k_val) * (B * m + C * M + M * B)
        imag_F1 = (C * K_val + B * k_val) * M * m * omega**5 - \
                  (C * K_val + B * k_val) * (m * K_val + M * k_val + M * K_val) * omega**3 + \
                  (-omega**2 * C * B + k_val * K_val) * (B * m + C * M + M * B) * omega**3
        F1_norm = np.sqrt(real_F1**2 + imag_F1**2) / D_sq

        real_F2 = -m**2 * M**2 * omega**8 + omega**6 * M * m * (M + m) * K_val + \
                  (m * K_val + C * B + M * k_val + M * K_val) * omega**6 * m * M - \
                  (m * K_val + C * B + M * k_val + M * K_val) * omega**4 * (M + m) * K_val - \
                  m * M * omega**4 * K_val * k_val + omega**2 * (M + m) * K_val**2 * k_val - \
                  omega**6 * (M + m) * B * (B * m + C * M + M * B) + omega**4 * (C * K_val + B * k_val) * (M + m) * B
        imag_F2 = omega**7 * (M + m) * M * m * B - omega**5 * (M + m) * B * ((M + m) * K_val + C * B + M * k_val) + \
                  omega**3 * (M + m) * B * K_val * k_val - omega**7 * m * M * (B * m + C * M + M * B) + \
                  omega**5 * m * M * (C * K_val + B * k_val) - omega**3 * (m + M) * K_val * (C * K_val + B * k_val) + \
                  omega**5 * (B * m + C * M + M * B) * (M + m) * K_val
        F2_norm = np.sqrt(real_F2**2 + imag_F2**2) / D_sq

        return F1_norm, F2_norm, real_D, imag_D, D_sq, real_F1, imag_F1, real_F2, imag_F2

    # Baseline reference evaluations
    F1_ref, F2_ref, _, _, _, _, _, _, _ = compute_spectral_magnitudes(k1_ref, k2_ref)
    F1_val, F2_val, rD, iD, D_sq, rF1, iF1, rF2, iF2 = compute_spectral_magnitudes(k1, k2)

    residuals = np.array([F1_val / F1_ref - ck, F2_val / F2_ref - cs])

    # Exact partial derivatives for Jacobian matrix assembly
    d_rD_dk = -M * omega**2 + k1
    d_rD_dK = -(M + m) * omega**2 + k2
    d_iD_dk = omega * B
    d_iD_dK = omega * C

    d_Dsq_dk = 2 * rD * d_rD_dk + 2 * iD * d_iD_dk
    d_Dsq_dK = 2 * rD * d_rD_dK + 2 * iD * d_iD_dK

    d_rF1_dk = M * m * k1 * omega**4 - (M + m) * k1**2 * omega**2 - 2 * M * k2 * k1 * omega**2 + \
               2 * k1**2 * k2 + 2 * omega**2 * k2 * B**2 - B**2 * omega**4 * (M + m)
    d_rF1_dK = M * m * k2 * omega**4 - omega**2 * M * k2**2 - 2 * omega**2 * (M + m) * k1 * k2 + \
               2 * k1 * k2**2 + 2 * omega**2 * C**2 * k1 - omega**4 * M * C**2
    d_iF1_dk = omega**5 * B * M * m - 2 * omega**3 * B * M * k2
    d_iF1_dK = omega**5 * C * M * m - 2 * omega**3 * C * (M + m) * k1

    F1_sq = rF1**2 + iF1**2
    d_F1_dk = (0.5 * (F1_sq)**(-0.5) * (2 * rF1 * d_rF1_dk + 2 * iF1 * d_iF1_dk) * D_sq - np.sqrt(F1_sq) * d_Dsq_dk) / (D_sq**2)
    d_F1_dK = (0.5 * (F1_sq)**(-0.5) * (2 * rF1 * d_rF1_dK + 2 * iF1 * d_iF1_dK) * D_sq - np.sqrt(F1_sq) * d_Dsq_dK) / (D_sq**2)

    d_rF2_dk = m * M**2 * omega**6 - m * M * omega**4 * k1 - M * (M + m) * omega**4 * k1 + omega**2 * (M + m) * k1**2 + omega**4 * B**2 * (M + m)
    d_rF2_dK = 2 * omega**6 * M * m * (M + m) - omega**4 * m * M * k2 - omega**4 * (M + m) * (2 * (M + m) * k1 + M * k2) + 2 * (M + m) * k1 * k2 * omega**2
    d_iF2_dk = -omega**5 * B * M**2
    d_iF2_dK = -omega**5 * (M + m)**2 * B + omega**5 * m * M * C - 2 * omega**3 * (M + m) * C * k1 + (M + m) * (B * m + C * M + M * B) * omega**5

    F2_sq = rF2**2 + iF2**2
    d_F2_dk = (0.5 * (F2_sq)**(-0.5) * (2 * rF2 * d_rF2_dk + 2 * iF2 * d_iF2_dk) * D_sq - np.sqrt(F2_sq) * d_Dsq_dk) / (D_sq**2)
    d_F2_dK = (0.5 * (F2_sq)**(-0.5) * (2 * rF2 * d_rF2_dK + 2 * iF2 * d_iF2_dK) * D_sq - np.sqrt(F2_sq) * d_Dsq_dK) / (D_sq**2)

    jacobian = np.array([
        [d_F1_dK / F1_ref, d_F1_dk / F1_ref],
        [d_F2_dK / F2_ref, d_F2_dk / F2_ref]
    ])

    return residuals, jacobian


def optimize_suspension(x0: np.ndarray = np.array([5350.0, 136100.0]), tol: float = 1e-6, max_iter: int = 50):
    """
    Solves non-linear parameter targeting problem using multivariate Newton-Raphson:
        x_{k+1} = x_k - J(x_k)^{-1} * F(x_k)
    """
    x = x0.astype(float)
    iteration_history = [x.copy()]

    for iteration in range(max_iter):
        residuals, jacobian = transfer_functions_and_jacobian(x[0], x[1])
        residual_norm = np.linalg.norm(residuals)

        if residual_norm < tol:
            return x, iteration_history, True

        delta_step = np.linalg.solve(jacobian, -residuals)
        x += delta_step
        iteration_history.append(x.copy())

    return x, iteration_history, False