import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from scipy.stats import binom, geom

# ==============================================================================
# 0. CONFIGURATION
# ==============================================================================
SHOTS = 2048  # Number of times to run the circuit
# A single, reusable simulator instance is efficient
AER_SIMULATOR = AerSimulator()

# ==============================================================================
# 1. CORE & HELPER FUNCTIONS
# ==============================================================================

def num_of_qbits(n):
    """Calculates the number of qubits needed for n layers."""
    # We use n+1 qubits for position and n+1 for ancilla/helpers
    return 2 * (n + 1)

def one_peg(qc, start):
    """Applies a controlled swap for the first layer."""
    qc.cswap(0, start - 1, start)
    qc.cx(start, 0)
    qc.cswap(0, start, start + 1)
    return qc

def multi_swap(qc, start, curr, count):
    """Applies a series of controlled swaps for subsequent layers."""
    n = curr
    max_q = start + count
    while n != max_q:
        qc.cswap(0, n, n + 1)
        if n < max_q - 1:
            qc.cx(n + 1, 0)
        n += 1
    return qc

# def measure_gb(qc):
#     cb = 0
#     for i in range(1, qc.num_qubits, 2):
#         qc.measure(i, cb)
#         cb += 1
#     return qc

def measure_gb(qc, n):
    """
    Measures the n+1 position qubits to get a one-hot encoded result.
    This is the CORRECT measurement scheme.
    """
    # Position qubits are assumed to be in a contiguous block.
    # Let's say they are qubits 1 to n+1.
    position_qubits = range(1, n + 2)
    qc.measure(position_qubits, range(n + 1))
    return qc


def run_circuit(qc):
    """Executes a circuit on the noiseless simulator and returns counts."""
    job = AER_SIMULATOR.run(transpile(qc, AER_SIMULATOR), shots=SHOTS)
    return job.result().get_counts()


def calculate_quantum_walk_distribution(n_steps):
    """
    Calculates the theoretical probability distribution for a 1D Hadamard QW.
    CORRECTED VERSION.
    """
    # Position space goes from -n to +n, so 2*n+1 possible sites.
    num_positions = 2 * n_steps + 1
    # We map these positions to indices 0, 1, ..., 2*n
    initial_pos_idx = n_steps

    # State vector: [amp_coin0_pos0, amp_coin1_pos0, amp_coin0_pos1, amp_coin1_pos1, ...]
    # Size is 2 * num_positions
    psi = np.zeros(2 * num_positions, dtype=complex)

    # Initial state: |0> coin state at the center position
    psi[2 * initial_pos_idx] = 1.0

    # Operators
    H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]]) # Hadamard coin operator

    # Evolve the state for n_steps
    for _ in range(n_steps):
        new_psi = np.zeros_like(psi)
        
        # 1. Apply Coin Flip (Hadamard) to every position
        for i in range(num_positions):
            # Extract coin amplitudes at position i
            coin_state = np.array([psi[2 * i], psi[2 * i + 1]])
            # Apply Hadamard
            new_coin_state = H @ coin_state
            psi[2 * i], psi[2 * i + 1] = new_coin_state[0], new_coin_state[1]

        # 2. Apply Conditional Shift
        for i in range(num_positions):
            # If coin is |0> (at psi[2*i]), move LEFT to position i-1
            if i > 0:
                new_psi[2 * (i - 1)] += psi[2 * i]
            # If coin is |1> (at psi[2*i+1]), move RIGHT to position i+1
            if i < num_positions - 1:
                new_psi[2 * (i + 1)] += psi[2 * i + 1]
                
        psi = new_psi

    # Calculate final probabilities by summing |amp|^2 for both coin states at each position
    probabilities = np.zeros(num_positions)
    for i in range(num_positions):
        probabilities[i] = np.abs(psi[2 * i])**2 + np.abs(psi[2 * i + 1])**2
        
    # For a walk of n steps, there are n+1 possible final bins.
    # The final positions are separated by 2 sites.
    final_distribution = []
    for i in range(n_steps + 1):
        # This maps the n+1 bins to the correct indices in the full probability vector
        pos_index = (initial_pos_idx - n_steps) + 2*i
        final_distribution.append(probabilities[pos_index])
        
    return np.array(final_distribution)


# ==============================================================================
# 2. MAIN CIRCUIT CREATION FUNCTIONS (Your Three Deliverables)
# ==============================================================================

def create_gaussian_circuit(n):
    """Creates a Galton board circuit that produces a Binomial/Gaussian distribution."""
    qbits = num_of_qbits(n)
    start = qbits // 2
    qc = QuantumCircuit(qbits, n + 1)
    qc.x(start)
    
    curr = start
    count = 1
    while curr > 1 and count <= n:
        qc.h(0)  # Unbiased coin flip
        # This part could also be a helper if you wish, but it's clear here.
        if count < 2:
            one_peg(qc, curr)
        else:
            multi_swap(qc, start, curr - 1, count)
        qc.reset(0)  # Key for classical random walk -> binomial dist
        curr -= 1
        count += 1
        
    return measure_gb(qc, n)

def create_exponential_circuit(n, lambda_param):
    """Creates a Galton board circuit that produces a Geometric/Exponential distribution."""
    # Formula relating p to lambda for a geometric approximation of an exponential.
    # p is the probability of an "event" (moving one direction).
    p = 1 - np.exp(-lambda_param) # A more standard definition
    # The probability of measuring |1> after Rx(theta) on |0> is sin^2(theta/2).
    # So, we set sin^2(theta/2) = p
    theta = 2 * np.arcsin(np.sqrt(p))
    
    qbits = num_of_qbits(n)
    start = qbits // 2
    qc = QuantumCircuit(qbits, n + 1)
    qc.x(start)

    curr = start
    count = 1
    while curr > 1 and count <= n:
        qc.rx(theta, 0)  # Biased coin flip
        if count < 2:
            one_peg(qc, curr)
        else:
            multi_swap(qc, start, curr - 1, count)
        qc.reset(0)  # Still a classical-like walk
        curr -= 1
        count += 1
        
    return measure_gb(qc, n)

def create_quantum_walk_circuit(n):
    """Creates a circuit for a 1D Hadamard Quantum Walk."""
    qbits = num_of_qbits(n)
    start = qbits // 2
    qc = QuantumCircuit(qbits, n + 1)
    qc.x(start)

    curr = start
    count = 1
    while curr > 1 and count <= n:
        qc.h(0)  # Hadamard coin flip
        if count < 2:
            one_peg(qc, curr)
        else:
            multi_swap(qc, start, curr - 1, count)
        # NO qc.reset(0) HERE! This preserves coherence and makes it a quantum walk.
        curr -= 1
        count += 1
        
    return measure_gb(qc, n)


# ==============================================================================
# 3. ANALYSIS FUNCTIONS (Computing Distances/MSE)
# ==============================================================================

def analyze_results(counts, n, shots, expected_dist):
    """Generic analysis function to compute MSE from one-hot counts."""
    empirical_counts = np.zeros(n + 1)
    for bitstring, count in counts.items():
        try:
            position = bitstring.find('1')
            if position != -1:
                empirical_counts[position] += count
        except:
            pass # Ignore anomalous results like '00000'

    empirical_dist = empirical_counts / shots
    mse = np.sum((empirical_dist - expected_dist)**2)
    
    return {"mse": mse, "empirical": empirical_dist, "expected": expected_dist}

# ==============================================================================
# 4. MAIN EXECUTION (Your Experiment Runner)
# ==============================================================================

if __name__ == "__main__":
    N_LAYERS = 4
    LAMBDA = 0.5
    NUM_RUNS = 10  # Number of times to repeat the experiment for stochastic analysis

    print(f"Running experiments for n={N_LAYERS} layers and {SHOTS} shots.")
    print(f"Each experiment will be repeated {NUM_RUNS} times.\n")

    # --- Store all MSEs for statistical analysis ---
    all_gaussian_mses = []
    all_exponential_mses = []
    all_qwalk_mses = []

    # --- Pre-calculate theoretical distributions ---
    expected_gaussian = binom.pmf(range(N_LAYERS + 1), N_LAYERS, 0.5)
    p_geom = 1 - np.exp(-LAMBDA)
    expected_exponential = geom.pmf(range(1, N_LAYERS + 2), p_geom)
    expected_qwalk = calculate_quantum_walk_distribution(N_LAYERS)

    for i in range(NUM_RUNS):
        print(f"--- Iteration {i+1}/{NUM_RUNS} ---")

        # 1. Gaussian / Binomial
        qc_gauss = create_gaussian_circuit(N_LAYERS)
        counts_gauss = run_circuit(qc_gauss)
        results_gauss = analyze_results(counts_gauss, N_LAYERS, SHOTS, expected_gaussian)
        all_gaussian_mses.append(results_gauss['mse'])
        print(f"  Gaussian MSE: {results_gauss['mse']:.6f}")

        # 2. Exponential / Geometric
        qc_exp = create_exponential_circuit(N_LAYERS, LAMBDA)
        counts_exp = run_circuit(qc_exp)
        results_exp = analyze_results(counts_exp, N_LAYERS, SHOTS, expected_exponential)
        all_exponential_mses.append(results_exp['mse'])
        print(f"  Exponential MSE: {results_exp['mse']:.6f}")

        # 3. Hadamard Quantum Walk
        qc_qw = create_quantum_walk_circuit(N_LAYERS)
        counts_qw = run_circuit(qc_qw)
        results_qw = analyze_results(counts_qw, N_LAYERS, SHOTS, expected_qwalk)
        all_qwalk_mses.append(results_qw['mse'])
        print(f"  Quantum Walk MSE: {results_qw['mse']:.6f}")

    print("\n--- Final Stochastic Distance Analysis ---")
    print(f"Gaussian MSE      : Mean = {np.mean(all_gaussian_mses):.6f}, Std Dev = {np.std(all_gaussian_mses):.6f}")
    print(f"Exponential MSE   : Mean = {np.mean(all_exponential_mses):.6f}, Std Dev = {np.std(all_exponential_mses):.6f}")
    print(f"Quantum Walk MSE  : Mean = {np.mean(all_qwalk_mses):.6f}, Std Dev = {np.std(all_qwalk_mses):.6f}")