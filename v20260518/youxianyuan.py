import numpy as np
import sys

# --------------------------------------------------
def WynnEpsilon(sn, k):
    """
    Perform Wynn Epsilon Algorithm (original implementation)
    """
    n = 2 * k + 1
    e = np.zeros((n + 1, n + 1))

    for i in range(1, n + 1):
        e[i, 1] = sn[i - 1]

    for i in range(3, n + 2):
        for j in range(3, i + 1):
            e[i - 1, j - 1] = e[i - 2, j - 3] + 1.0 / (e[i - 1, j - 2] - e[i - 2, j - 2])

    ek = e[:, 1:n + 1:2]
    return ek


# --------------------------------------------------
def compute_slope(x, y):
    """Compute slope of log(y) vs log(x) using linear regression."""
    logx = np.log(x)
    logy = np.log(y)
    # Remove possible inf/nan
    mask = np.isfinite(logx) & np.isfinite(logy)
    if np.sum(mask) < 2:
        return np.nan
    slope, _ = np.polyfit(logx[mask], logy[mask], 1)
    return slope


# --------------------------------------------------
def main():
    # --- original computation ---
    n = np.logspace(0, 8, 9, base=2).astype(int)   # [1, 2, 4, 8, 16, 32, 64, 128, 256]
    pn = n * np.sin(np.pi / n)                     # approximation of pi by inscribed polygon

    pw = np.zeros(4)
    for i in range(1, 5):
        en = WynnEpsilon(pn, i)
        pw[i - 1] = en[-1, -1]

    # --- error analysis ---
    true_pi = np.pi
    error_pn = np.abs(pn - true_pi)               # errors of polygon approximations
    h = 1.0 / n                                   # step size h = 1/n

    # errors of Wynn‑extrapolated values (only for n = 4, 16, 64, 256)
    n_wynn = n[2::2]                              # indices 2,4,6,8 -> [4,16,64,256]
    error_pw = np.abs(pw - true_pi)

    # --- compute convergence slopes ---
    slope_pn = compute_slope(h, error_pn)
    slope_pw = compute_slope(1.0 / n_wynn, error_pw)   # use h for Wynn points

    # --- print table like the image ---
    print("\n" + "=" * 75)
    print(f"{'n':<6} {'h = 1/n':<12} {'error (pn)':<20} {'error (Wynn)':<20}")
    print(f"{'':<6} {'':<12} {'(slope = {:.2f})'.format(slope_pn):<20} {'(slope = {:.2f})'.format(slope_pw):<20}".format())
    print("-" * 75)

    # Keep track of which Wynn value corresponds to which n
    wynn_idx = 0
    for i, (ni, hi, err_pn) in enumerate(zip(n, h, error_pn)):
        if i % 2 == 0 and i > 0:          # positions where Wynn result is available
            err_pw_val = error_pw[wynn_idx]
            print(f"{ni:<6} {hi:<12.3e} {err_pn:<20.6e} {err_pw_val:<20.6e}")
            wynn_idx += 1
        else:
            print(f"{ni:<6} {hi:<12.3e} {err_pn:<20.6e} {'---':<20}")
    print("=" * 75)

    # --- optional plot (requires matplotlib) ---
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        plt.loglog(h, error_pn, 'o-', label=f'Polygon (slope = {slope_pn:.2f})')
        plt.loglog(1.0 / n_wynn, error_pw, 's-', label=f'Wynn Epsilon (slope = {slope_pw:.2f})')
        plt.xlabel('h = 1/n')
        plt.ylabel('Absolute error')
        plt.title('Convergence of π approximations')
        plt.grid(True, which='both', linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig('pi_convergence.png', dpi=150)
        plt.show()
        print("\nPlot saved as 'pi_convergence.png'")
    except ImportError:
        print("\nMatplotlib not installed – skipping plot.")


# --------------------------------------------------
if __name__ == '__main__':
    main()