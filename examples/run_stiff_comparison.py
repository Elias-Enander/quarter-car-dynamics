"""
Numerical stiffness benchmarking: Evaluates convergence order of the Implicit Trapezoidal scheme.
"""
import sys
from pathlib import Path

# Lägger till projektets rotmapp i Pythons sökväg
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.integrate import solve_ivp
from quarter_car.model import QuarterCarModel, SuspensionParameters
from quarter_car.solvers import implicit_trapezoidal


def main():
    print("=" * 70)
    print("NUMERICAL STIFFNESS & CONVERGENCE ORDER ANALYSIS")
    print("=" * 70)

    # Define stiff regime (stiff tire spring k2 = 100 * k2_ref
    stiff_params = SuspensionParameters(k1=5350.0, k2=100.0 * 136100.0)
    model = QuarterCarModel(stiff_params)

    t_span = (0.0, 0.05)
    v0 = np.zeros(4)

    # tight error tolerances: this is the reference soulutuin
    ref_sol = solve_ivp(model.rhs, t_span, v0, method="RK45", rtol=1e-9, atol=1e-9)
    z2_reference = ref_sol.y[1, -1]

    # Step-size reduction 
    alpha_scales = [1.0, 0.5, 0.25, 0.125]
    base_dt = 0.05 / 100.0
    errors = []

    print("\n[1/2] Computing step-size error progression for Implicit Trapezoidal...")
    for alpha in alpha_scales:
        dt = alpha * base_dt
        t_vec, v_mat = implicit_trapezoidal(model, t_span, dt=dt, v0=v0)
        z2_final = v_mat[-1, 1]
        err = abs(z2_reference - z2_final)
        errors.append(err)
        print(f"      dt = {dt*1e4:.2f}e-4 s | Wheel Displacement Error: {err:.6e} m")

    #convergence orders p = log2(e_k / e_{k+1})
    errors = np.array(errors)
    empirical_orders = np.log2(errors[:-1] / errors[1:])

    print("\n[2/2] Empirical Convergence Order (Theoretical: 2.0):")
    for idx, p in enumerate(empirical_orders):
        print(f"      Refinement step {idx + 1} -> {idx + 2}: p = {p:.3f}")


if __name__ == "__main__":
    main()