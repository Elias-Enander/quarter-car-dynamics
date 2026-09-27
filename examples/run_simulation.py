"""
Main simulation and validation script comparing baseline vs optimized suspension dynamics.
"""

import sys
from pathlib import Path

# Lägger till projektets rotmapp i Pythons sökväg
sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from quarter_car.model import QuarterCarModel, SuspensionParameters
from quarter_car.optimizer import optimize_suspension
from quarter_car.solvers import forward_euler


def main():
    print("=" * 70)
    print("QUARTER-CAR SUSPENSION DYNAMICS & NUMERICAL OPTIMIZATION")
    print("=" * 70)

    # newton-Raphson
    print("\n[1/3] Running Multivariate Newton-Raphson Optimization...")
    initial_guess = np.array([5350.0, 136100.0])
    solution, history, converged = optimize_suspension(x0=initial_guess)
    k1_opt, k2_opt = solution

    print(f"      Convergence Status : {converged}")
    print(f"      Iterations Required: {len(history) - 1}")
    print(f"      Optimized k1       : {k1_opt:.2f} N/m (Baseline: 5350.0 N/m)")
    print(f"      Optimized k2       : {k2_opt:.2f} N/m (Baseline: 136100.0 N/m)")

    
    print("\n[2/3] Solving Transient Response for Road Bump Disturbance...")
    params_baseline = SuspensionParameters(k1=5350.0, k2=136100.0)
    params_optimized = SuspensionParameters(k1=k1_opt, k2=k2_opt)

    model_baseline = QuarterCarModel(params_baseline)
    model_optimized = QuarterCarModel(params_optimized)

    t_span = (0.0, 2.0)
    v0 = np.zeros(4)

    # Runge-Kutta 4-5
    res_base = solve_ivp(model_baseline.rhs, t_span, v0, method="RK45", rtol=1e-6, atol=1e-9)
    res_opt = solve_ivp(model_optimized.rhs, t_span, v0, method="RK45", rtol=1e-6, atol=1e-9)

    max_z1_base = np.max(np.abs(res_base.y[0]))
    max_z1_opt = np.max(np.abs(res_opt.y[0]))
    reduction_pct = (1.0 - max_z1_opt / max_z1_base) * 100.0
    print(f"      Chassis Peak Displacement (Baseline) : {max_z1_base * 1000:.2f} mm")
    print(f"      Chassis Peak Displacement (Optimized): {max_z1_opt * 1000:.2f} mm")
    print(f"      Chassis Peak Vibration Attenuation   : {reduction_pct:.1f}%")

   
    print("\n[3/3] Exporting Publication-Ready Figures...")

    # Figure 1: Chassis Vibration
    plt.figure(figsize=(9, 5))
    plt.plot(res_base.t, res_base.y[0], "r--", linewidth=1.8, label=r"Baseline ($k_1=5350\ \mathrm{{N/m}}$)")
    plt.plot(res_opt.t, res_opt.y[0], "b-", linewidth=2.2, label=rf"Optimized ($k_1={k1_opt:.0f}\ \mathrm{{N/m}}$)")
    plt.xlabel("Time [s]", fontsize=11)
    plt.ylabel("Chassis Vertical Displacement $z_1$ [m]", fontsize=11)
    plt.title("Transient Chassis Response: Baseline vs Optimized Parameters", fontsize=12, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(frameon=True, facecolor="white", framealpha=0.9)
    plt.tight_layout()
    plt.savefig("chassis_vibration_comparison.png", dpi=300)
    print("      Saved: 'chassis_vibration_comparison.png'")

    # Figure 2: Numerical Solver Verification
    dt_limit = model_baseline.compute_forward_euler_stability_limit()
    dt_stable = 0.95 * dt_limit
    t_fe, v_fe = forward_euler(model_baseline, t_span, dt=dt_stable, v0=v0)

    plt.figure(figsize=(9, 5))
    plt.plot(res_base.t, res_base.y[1], "k-", linewidth=2.0, label="Benchmark (Adaptive RK45)")
    plt.plot(t_fe, v_fe[:, 1], "r-.", linewidth=1.4, label=rf"Forward Euler ($\Delta t = 0.95 \Delta t_{{\max}} = {dt_stable*1000:.2f}\ \mathrm{{ms}}$)")
    plt.xlabel("Time [s]", fontsize=11)
    plt.ylabel("Wheel Vertical Displacement $z_2$ [m]", fontsize=11)
    plt.title("Numerical Scheme Verification: Adaptive RK45 vs Forward Euler", fontsize=12, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(frameon=True, facecolor="white", framealpha=0.9)
    plt.tight_layout()
    plt.savefig("solver_benchmark.png", dpi=300)
    print("      Saved: 'solver_benchmark.png'")

    print("\nExecution completed successfully.\n")


if __name__ == "__main__":
    main()