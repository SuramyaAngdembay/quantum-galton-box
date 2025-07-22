import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.galton import galton_board_circuit, run_circuit, mse

def main():
    qc = galton_board_circuit(4)
    counts = run_circuit(qc)
    stats = mse(counts)

    print(f"MSE: {stats['mse']:.6f}")
    print(f"Empirical mean: {stats['empirical_mean']:.3f}, variance: {stats['empirical_var']:.3f}")
    print(f"Expected mean: {stats['expected_mean']:.3f}, variance: {stats['expected_var']:.3f}")
    print(f"---------------------------------------------------------------------------------")
    print(f"Empirical data: {stats["empirical"]}")

if __name__ == "__main__":
    main()
