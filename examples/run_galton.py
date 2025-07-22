import sys
import os
import numpy as np
from qiskit.visualization import plot_distribution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.galton import galton_board_circuit, run_circuit, mse

def main():
    n = 3
    qc = galton_board_circuit(n, bias=True, theta=0.83*np.pi)
    counts = run_circuit(qc)
    stats = mse(counts)

    print(f"Galton Box level: {n}; pegs: {(n**2+n)/2}")
    print(f"MSE: {stats['mse']:.6f}")
    print(f"Empirical mean: {stats['empirical_mean']:.3f}, variance: {stats['empirical_var']:.3f}")
    print(f"Expected mean: {stats['expected_mean']:.3f}, variance: {stats['expected_var']:.3f}")
    print(f"---------------------------------------------------------------------------------")
    print(f"Empirical data: {stats["empirical"]}")
    plot_distribution([counts])
    

if __name__ == "__main__":
    main()
