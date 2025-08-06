import numpy as np
import matplotlib.pyplot as plt
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram

def benchmark_exponential_distribution(board, lam, shots=1024, runs=20, plot=True):
    sim = AerSimulator()
    n_bins = board.n + 1
    
    all_counts = []
    all_distributions = []
    
    mse_list = []
    tvd_list = []
    empirical_means = []
    empirical_vars = []

    expected = board.compute_exponential_distribution(n_bins, lam)
    expected_mean = np.sum(np.arange(n_bins) * expected)
    expected_var = np.sum(((np.arange(n_bins) - expected_mean) ** 2) * expected)

    for r in range(runs):
        circ = board.exponential_gb(lam=lam)
        counts = board.run_circuit(circ, sim, shots=shots)
        dist = board.onehot_counts_to_distribution(counts)
        # dist = np.flip(dist)

        all_counts.append(counts)
        all_distributions.append(dist)

        empirical_mean = np.sum(np.arange(n_bins) * dist)
        empirical_var = np.sum(((np.arange(n_bins) - empirical_mean) ** 2) * dist)

        mse = np.mean((dist - expected) ** 2)
        tvd = 0.5 * np.sum(np.abs(dist - expected))

        mse_list.append(mse)
        tvd_list.append(tvd)
        empirical_means.append(empirical_mean)
        empirical_vars.append(empirical_var)

    # Aggregate results
    avg_dist = np.mean(all_distributions, axis=0)
    avg_mse = np.mean(mse_list)
    avg_tvd = np.mean(tvd_list)
    avg_mean = np.mean(empirical_means)
    avg_var = np.mean(empirical_vars)

    # Print diagnostics
    print(f"\n=== Benchmark for λ = {lam}, n = {board.n}, shots = {shots}, runs = {runs} ===")
    print(f"Expected Mean: {expected_mean:.4f}, Expected Variance: {expected_var:.4f}")
    print(f"Avg Empirical Mean: {avg_mean:.4f}, Variance: {avg_var:.4f}")
    print(f"Avg MSE: {avg_mse:.6f}")
    print(f"Avg TVD: {avg_tvd:.6f}")
    print(f"Final Averaged Distribution:")
    print(f"Observed: {avg_dist}")
    print(f"Expected: {expected}")

    if plot:
        x = np.arange(n_bins)
        plt.bar(x - 0.2, avg_dist, width=0.4, label='Observed (avg)', alpha=0.7)
        plt.bar(x + 0.2, expected, width=0.4, label='Expected (exp)', alpha=0.7)
        plt.title(f"Exponential Distribution Benchmark (λ = {lam})")
        plt.xlabel("Bucket Index")
        plt.ylabel("Probability")
        plt.legend()
        plt.grid(True)
        plt.show()

    return {
        "avg_dist": avg_dist,
        "expected": expected,
        "mse_list": mse_list,
        "tvd_list": tvd_list,
        "avg_mse": avg_mse,
        "avg_tvd": avg_tvd,
        "empirical_means": empirical_means,
        "empirical_vars": empirical_vars,
        "avg_mean": avg_mean,
        "avg_var": avg_var
    }
