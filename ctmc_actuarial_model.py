"""
Continuous-Time Multi-State Actuarial Valuation
Using Gompertz-Calibrated Mortality and Age-Dependent Markov Models

Linear Algebra Group Project
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm, eigvals
from sklearn.linear_model import LinearRegression
def load_mortality_data(filepath=r"C:\Users\KIETLINH\Downloads\mortality_data_female.csv", age_min=30, age_max=90, year=2023):
    import pandas as pd
    df = pd.read_csv(filepath)
    df = df[df["Year"] == year]

    ages = df["Age (x)"].values.astype(float)
    mx = df["Central death rate m(x,n)"].values.astype(float)

    mask = (ages >= age_min) & (ages <= age_max)
    ages = ages[mask]
    mx = mx[mask]

    # For Gompertz fitting, treat mx as Dx/Ex
    Ex = np.ones_like(mx)
    Dx = mx

    print(f"Loaded {len(ages)} age groups from year {year}")
    return ages, Dx, Ex

# ============================================================
# 2. GOMPERTZ CALIBRATION
# ============================================================
# The Gompertz hazard is: mu(x) = A * exp(B * x)
# Taking logs: log(mu(x)) = log(A) + B*x
# We fit this via OLS on the log crude rates.
# Note: this is a log-linear least-squares fit, not strict MLE.

def fit_gompertz(ages, Dx, Ex):
    """Fit Gompertz parameters via log-linear regression on crude rates."""
    lambda_hat = np.clip(Dx / Ex, 1e-10, None)

    X = ages.reshape(-1, 1)
    y = np.log(lambda_hat)

    reg = LinearRegression().fit(X, y)
    B = reg.coef_[0]
    A = np.exp(reg.intercept_)

    return A, B, lambda_hat


def gompertz_hazard(age, A, B):
    """Gompertz force of mortality at a given age."""
    return A * np.exp(B * age)


# ============================================================
# 3. GENERATOR MATRIX Q(x) — THE CORE LINEAR ALGEBRA
# ============================================================
# States: 0 = Healthy, 1 = Disabled, 2 = Dead (absorbing)
#
# The generator matrix Q satisfies:
#   - Off-diagonal entries Q[i,j] >= 0 (transition rates)
#   - Row sums = 0 (probability conservation)
#   - Dead is absorbing: row 2 is all zeros
#
# The transition matrix over interval dt is P(dt) = exp(Q * dt),
# computed via matrix exponentiation (eigendecomposition or Padé).

def build_generator(age, A, B, lambda_HD, lambda_DH, DD_factor):
    """
    Construct the 3x3 generator matrix Q(age).

    Parameters
    ----------
    age        : current age
    A, B       : Gompertz mortality parameters
    lambda_HD  : healthy -> disabled transition rate
    lambda_DH  : disabled -> healthy recovery rate
    DD_factor  : disabled mortality multiplier (relative to healthy)
    """
    mu_H = gompertz_hazard(age, A, B)       # healthy mortality
    mu_D = DD_factor * mu_H                  # disabled mortality

    Q = np.array([
        [-(lambda_HD + mu_H),   lambda_HD,            mu_H  ],
        [  lambda_DH,          -(lambda_DH + mu_D),   mu_D  ],
        [  0,                    0,                    0     ]
    ])

    return Q


# ============================================================
# 4. EIGENVALUE ANALYSIS OF Q (LINEAR ALGEBRA SHOWCASE)
# ============================================================

def analyze_generator(age, A, B, params):
    """
    Display the eigenstructure of Q(age).

    For a CTMC generator matrix:
      - One eigenvalue is always 0 (corresponding to the absorbing state)
      - All other eigenvalues have negative real parts (ensuring convergence)
      - exp(Qt) = P * exp(Dt) * P^{-1} where D = diag(eigenvalues)
    """
    Q = build_generator(age, A, B, **params)
    evals = eigvals(Q)

    print(f"\n--- Eigenvalue Analysis at age {age} ---")
    print(f"Q({age}) =")
    print(np.array2string(Q, precision=6, suppress_small=True))
    print(f"\nEigenvalues: {np.round(evals, 6)}")
    print(f"All non-zero eigenvalues have negative real part: "
          f"{all(e.real < 1e-10 for e in evals if abs(e) > 1e-10)}")
    print(f"Zero eigenvalue present (absorbing state): "
          f"{any(abs(e) < 1e-10 for e in evals)}")

    return Q, evals


# ============================================================
# 5. STATE PROBABILITY EVOLUTION
# ============================================================

def evolve_probabilities(age_start, A, B, params, T=50, dt=0.1):
    """
    Compute state probabilities over time using P(dt) = exp(Q*dt).

    This is deterministic probability evolution, not stochastic simulation:
    at each step, pi(t+dt) = pi(t) @ exp(Q(age) * dt).
    """
    n_steps = int(T / dt)
    times = np.arange(n_steps) * dt
    trajectory = np.zeros((n_steps, 3))

    state = np.array([1.0, 0.0, 0.0])  # start in Healthy

    for i, t in enumerate(times):
        trajectory[i] = state
        age = age_start + t
        Q = build_generator(age, A, B, **params)
        P_mat = expm(Q * dt)
        state = state @ P_mat

    return times, trajectory


# ============================================================
# 6. EXPECTED PRESENT VALUE (EPV) OF DEATH BENEFIT
# ============================================================

def compute_EPV(age_start, A, B, params, r=0.03, benefit=100_000,
                T=50, dt=0.1):
    """
    EPV = sum over t of: benefit * Pr(death in [t, t+dt]) * discount(t)

    The incremental death probability is P(Dead, t+dt) - P(Dead, t).
    """
    state = np.array([1.0, 0.0, 0.0])
    EPV = 0.0

    for t in np.arange(0, T, dt):
        age = age_start + t
        Q = build_generator(age, A, B, **params)
        P_mat = expm(Q * dt)

        next_state = state @ P_mat
        death_increment = next_state[2] - state[2]
        discount = np.exp(-r * t)
        EPV += benefit * death_increment * discount

        state = next_state

    return EPV


# ============================================================
# 7. PLOTTING
# ============================================================

def plot_gompertz_fit(ages, lambda_hat, A, B):
    """Plot observed vs fitted Gompertz hazard rates."""
    fitted = gompertz_hazard(ages, A, B)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.scatter(ages, lambda_hat, s=10, alpha=0.7, label="Observed rates")
    ax1.plot(ages, fitted, "r-", linewidth=2, label="Gompertz fit")
    ax1.set_xlabel("Age")
    ax1.set_ylabel("Force of mortality μ(x)")
    ax1.set_title("Gompertz Fit (Natural Scale)")
    ax1.legend()

    ax2.scatter(ages, np.log(lambda_hat), s=10, alpha=0.7, label="log(observed)")
    ax2.plot(ages, np.log(fitted), "r-", linewidth=2, label="log(Gompertz)")
    ax2.set_xlabel("Age")
    ax2.set_ylabel("log μ(x)")
    ax2.set_title("Gompertz Fit (Log Scale — should be linear)")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("gompertz_fit.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_state_evolution(times, traj, age_start):
    """Plot state probabilities with age on secondary x-axis."""
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(times, traj[:, 0], label="Healthy", linewidth=2, color="#2ecc71")
    ax1.plot(times, traj[:, 1], label="Disabled", linewidth=2, color="#e67e22")
    ax1.plot(times, traj[:, 2], label="Dead", linewidth=2, color="#e74c3c")

    ax1.set_xlabel("Time (years from start)")
    ax1.set_ylabel("Probability")
    ax1.set_title(f"State Probability Evolution (starting age {age_start})")
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Secondary x-axis showing attained age
    ax2 = ax1.twiny()
    ax2.set_xlim(age_start, age_start + times[-1])
    ax2.set_xlabel("Attained Age")

    plt.savefig("state_evolution.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_sensitivity(age_start, A, B, base_params):
    """Sensitivity analysis: EPV across varying disability inception rates."""
    rates = np.linspace(0.005, 0.10, 20)
    epvs = []

    for lam in rates:
        test_params = {**base_params, "lambda_HD": lam}
        epv = compute_EPV(age_start, A, B, test_params)
        epvs.append(epv)

    plt.figure(figsize=(8, 5))
    plt.plot(rates, epvs, "b-o", markersize=4)
    plt.xlabel("Disability Inception Rate λ_HD")
    plt.ylabel("EPV of Death Benefit ($)")
    plt.title("Sensitivity: EPV vs. Disability Rate")
    plt.grid(True, alpha=0.3)
    plt.savefig("sensitivity.png", dpi=150, bbox_inches="tight")
    plt.show()


# ============================================================
# 8. MAIN
# ============================================================

def main():
    # --- Data and Gompertz fit ---
    ages, Dx, Ex = load_mortality_data()
    A, B, lambda_hat = fit_gompertz(ages, Dx, Ex)

    print("=== Gompertz Parameters ===")
    print(f"  A = {A:.8f}")
    print(f"  B = {B:.6f}")
    print(f"  Interpretation: mortality doubles every {np.log(2)/B:.1f} years")

    plot_gompertz_fit(ages, lambda_hat, A, B)

    # --- Model parameters (passed explicitly, never as globals) ---
    params = {
        "lambda_HD": 0.02,     # healthy -> disabled (per year)
        "lambda_DH": 0.10,     # disabled -> healthy (per year)
        "DD_factor": 2.0,      # disabled mortality multiplier
    }

    # --- Eigenvalue analysis at selected ages ---
    for age in [40, 60, 80]:
        analyze_generator(age, A, B, params)

    # --- State evolution ---
    age_start = 40
    times, traj = evolve_probabilities(age_start, A, B, params)
    plot_state_evolution(times, traj, age_start)

    # --- Verify probabilities sum to 1 (sanity check) ---
    row_sums = traj.sum(axis=1)
    print(f"\nProbability conservation check:")
    print(f"  Max deviation from 1: {np.max(np.abs(row_sums - 1)):.2e}")

    # --- EPV ---
    epv = compute_EPV(age_start, A, B, params)
    print(f"\n=== Expected Present Value ===")
    print(f"  Starting age: {age_start}")
    print(f"  Benefit: $100,000")
    print(f"  Discount rate: 3%")
    print(f"  EPV = ${epv:,.2f}")

    # --- Sensitivity ---
    print("\n=== Sensitivity: Disability Inception Rate ===")
    for lam in [0.01, 0.02, 0.05, 0.10]:
        test_params = {**params, "lambda_HD": lam}
        epv_test = compute_EPV(age_start, A, B, test_params)
        print(f"  λ_HD = {lam:.2f}: EPV = ${epv_test:,.2f}")

    plot_sensitivity(age_start, A, B, params)


if __name__ == "__main__":
    main()
