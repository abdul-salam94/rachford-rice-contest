from typing import Tuple

import numpy as np
from check_convergence import EPS_T

MAX_ITR = 100

def rachford_rice_solver(
    Nc: int, zi: np.ndarray, Ki: np.ndarray
) -> Tuple[int, np.ndarray, np.ndarray, float, float]:
    """
    Robust Rachford–Rice solver using the b-transformation to avoid catastrophic round-off error.
    Returns number of iterations, vapor composition, liquid composition, vapor fraction, liquid fraction.
    """

    # 1. Compute ci for original K
    ci = 1.0 / (1.0 - Ki)

    # 2. Compute bounds
    Kmin, Kmax = np.min(Ki), np.max(Ki)
    Vmin = 1.0 / (1.0 - Kmax)
    Vmax = 1.0 / (1.0 - Kmin)

    # 3. Function for h(V)
    def h(V):
        return np.sum(zi * (Ki - 1.0) / (1.0 + V * (Ki - 1.0)))

    # 4. Decide phase variable (V or L)
    h05 = h(0.5)
    solve_liquid = (h05 > 0.0)

    # 5. Generalized K, bounds, ci for transformed variable
    if solve_liquid:
        Khat = 1.0 / Ki
        phimin = 1.0 / (1.0 - np.max(Khat))
        phimax = 1.0 / (1.0 - np.min(Khat))
    else:
        Khat = Ki
        phimin = Vmin
        phimax = Vmax

    ci_hat = 1.0 / (1.0 - Khat)

    # 6. Newton-Raphson for b
    def func_b(b):
        return np.sum(zi * b / (1.0 + b * (phimin - ci_hat)))
    def dfunc_b(b):
        return np.sum(zi / (1.0 + b * (phimin - ci_hat)) ** 2)

    phi0 = 0.5
    b0 = 1.0 / (phi0 - phimin)
    b = b0 if b0 > 0 else 1.0
    bmin = 1.0 / (phimax - phimin) + 1e-12

    Niter = 0
    converged = False
    for Niter in range(1, MAX_ITR + 1):
        f = func_b(b)
        df = dfunc_b(b)
        if df == 0 or not np.isfinite(df):
            b_new = b * 1.1
        else:
            b_new = b - f / df
        if b_new < bmin or not np.isfinite(b_new):
            b_new = 0.5 * (b + max(b, 10 * bmin))
        if abs(f) < EPS_T:
            converged = True
            break
        b = b_new

    phi = phimin + 1.0 / b
    V = 1.0 - phi if solve_liquid else phi
    L = phi if solve_liquid else 1.0 - phi

    # 7. Compositions using b-transformed equations
    denom = 1.0 + b * (phimin - ci_hat)
    u = -zi * ci_hat * b / denom
    v = u * Khat

    if solve_liquid:
        yi = u
        xi = v
    else:
        xi = u
        yi = v

    # Normalize for safety
    xi = xi / np.sum(xi)
    yi = yi / np.sum(yi)

    # If non-converged, print warning
    if not converged:
        print("******************************************************")
        print("*** The maximum number of iterations was exceeded! ***")
        print("******************************************************")

    return Niter, yi, xi, V, L