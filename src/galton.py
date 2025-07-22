from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np
from scipy.stats import binom

SHOTS = 2024

def num_of_qbits(n):
    return 2 * (n + 1)

def one_peg(qc, start):
    qc.cswap(0, start-1, start)
    qc.cx(start, 0)
    qc.cswap(0, start, start+1)
    return qc

def measure_gb(qc):
    cb = 0
    for i in range(1, qc.num_qubits, 2):
        qc.measure(i, cb)
        cb += 1
    return qc

def multi_swap(qc, start, curr, count):
    n = curr
    max_q = start + count
    while n != max_q:
        qc.cswap(0, n, n+1)
        if n < max_q - 1:
            qc.cx(n+1, 0)
        n += 1
    return qc

def galton_board_circuit(n):
    qbits = num_of_qbits(n)
    start = qbits // 2
    qc = QuantumCircuit(qbits, n+1)
    qc.x(start)
    curr = start
    count = 1
    while curr != 1:
        qc.h(0)
        if count < 2:
            qc = one_peg(qc, curr)
        else:
            qc = multi_swap(qc, start, curr-1, count)
        qc.reset(0)
        curr -= 1
        count += 1
    return measure_gb(qc)

def run_circuit(qc, shots=SHOTS):
    sim = AerSimulator()
    job = sim.run(transpile(qc, sim), shots=shots)
    result = job.result()
    return result.get_counts()

def mse(counts, shots=SHOTS):
    counts = {k: counts[k] for k in sorted(counts.keys())}
    empirical, expected, mse_total = [], [], 0
    keys = list(counts.keys())
    for i, key in enumerate(keys):
        e_val = counts[key] / shots
        b_val = binom.pmf(i, len(keys)-1, 0.5)
        empirical.append(e_val)
        expected.append(b_val)
        mse_total += (e_val - b_val)**2

    return {
        "mse": mse_total,
        "empirical": empirical,
        "empirical_mean": np.mean(empirical),
        "empirical_var": np.var(empirical),
        "expected": expected,
        "expected_mean": np.mean(expected),
        "expected_var": np.var(expected)
    }
