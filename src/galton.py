from qiskit import QuantumCircuit, transpile
from qiskit.visualization import plot_histogram
from qiskit_aer import AerSimulator
import numpy as np
from scipy.stats import binom
import matplotlib.pyplot as plt 
from benchmark import benchmark_exponential_distribution

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

def galton_board_circuit(n, bias=False, theta=np.pi/2):
    qbits = num_of_qbits(n)
    start = qbits // 2
    qc = QuantumCircuit(qbits, n+1)
    qc.x(start)
    curr = start
    count = 1
    while curr != 1:
        if bias == True :
            qc.rx(theta,0)
        else:
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

class GaltonBoard():
    def __init__(self, n):
        self.n = n
        print(f"galton circuit of size {n} has been created.")

    # initialize the required circuit the n-level board
    def build_circuit(self):
        q_num = 2*(self.n+1) 
        qc = QuantumCircuit(q_num, self.n+1)
        return q_num, qc
    
    # one peg
    def one_peg(self, qc, start):
        qc.cswap(0, start-1, start)
        qc.cx(start, 0)
        qc.cswap(0, start, start+1)
        return qc
    
    def multi_swap(self, qc, start, curr, count):
        n = curr
        max_q = start + count
        while n != max_q:
            qc.cswap(0, n, n+1)
            if n < max_q - 1:
                qc.cx(n+1, 0)
            n += 1
        return qc
    
    def multi_swap_fine_tuned(self, qc, curr, curr_level, t):
            final_cnot = []
            for i in range(curr_level):
                qc = one_peg(qc, curr)
                if (i != curr_level-1):
                    final_cnot.append([curr+1,curr+2])
                    qc.reset(0)
                    qc.rx(t[i], 0)
                curr += 2
            
            qc.barrier()
            
            for cnot in final_cnot:
                qc.cx(cnot[1], cnot[0])
                qc.reset(cnot[1])
            return qc 

    def measure_gb(self,  qc):
        cb = 0
        for i in range(1, qc.num_qubits, 2):
            qc.measure(i, cb)
            cb += 1
        return qc
    
    # build the classical gb circuit that approximates to normal
    def classical_gb(self):
        qbits, qc = self.build_circuit()
        start = qbits // 2
        qc.x(start)
        curr = start
        count = 1
        while curr != 1:
            qc.h(0)
            if count < 2:
                qc = self.one_peg(qc, curr)
            else:
                qc = self.multi_swap(qc, start, curr-1, count)
            qc.reset(0)
            curr -= 1
            count += 1
        return self.measure_gb(qc)
    
    def theta_schedule_exp(self, lam):
        thetas = []
        for level in range(1, self.n + 1):  # Levels 1 to n
            theta_l = 2 * np.arcsin(np.exp(-lam * level / 2))  # θ for this level
            thetas.extend([theta_l] * level)  # Repeat θ 'level' times (for that many pegs)
        return thetas

    def exponential_gb(self, lam):
        qbits, qc = self.build_circuit()
        start = qbits // 2
        qc.x(start)
        curr = start
        curr_level = 1
        theta = self.theta_schedule_exp(lam)
        theta_idx = 0
        while curr_level < self.n+1:
            qc.rx(theta[theta_idx],0)
            theta_idx += 1
            if curr_level < 2:
                qc = self.one_peg(qc, curr)
            else:
                thetas_for_layer = theta[theta_idx: theta_idx + curr_level -1 ]
                qc = self.multi_swap_fine_tuned(qc, curr-1, curr_level, thetas_for_layer)
                curr -= 1
                theta_idx += curr_level -1 
            qc.reset(0)
            curr_level += 1
        return self.measure_gb(qc)
    
    def quantum_walk_gb(self):
        qbits, qc = self.build_circuit()
        start = qbits // 2
        qc.x(start)
        curr = start
        count = 1
        while curr != 1:
            qc.h(0)
            if count < 2:
                qc = self.one_peg(qc, curr)
            else:
                qc = self.multi_swap(qc, start, curr-1, count)
            curr -= 1
            count += 1
        return self.measure_gb(qc)   

    def run_circuit(self, qc, sim, shots=SHOTS):
        job = sim.run(transpile(qc, sim), shots=shots)
        result = job.result()
        return result.get_counts()
    
    def mse(self, counts, shots=SHOTS):
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

    def compute_exponential_distribution(self, n_bins, lam):
        """Compute target exponential distribution over discrete bins"""
        x = np.arange(n_bins)
        unnorm = np.exp(-lam * x)
        return unnorm / np.sum(unnorm)
    
    def compute_tvd(self, p, q):
        """Total Variation Distance between two discrete distributions"""
        return 0.5 * np.sum(np.abs(p - q))
    
    def exponential_error(self, ):
        pass 

    def onehot_counts_to_distribution(self, counts_dict):
        n_bins = len(next(iter(counts_dict)))  # number of bits
        counts = np.zeros(n_bins)
        for bitstring, count in counts_dict.items():
            index = bitstring.index('1')  # index of '1' gives bucket
            counts[index] += count
        return counts / np.sum(counts)
    
    def view_circuit(self, qc):
        qc.draw('mpl')
        return plt.show()
    
# SHOTS = 1024
# n = 3
# LAM = 0.5
# board = GaltonBoard(n)
# board_circ = board.build_circuit()
# quantum = board.exponential_gb(lam=LAM)
# # board.view_circuit(quantum)
# sim = AerSimulator()
# counts = board.run_circuit(quantum, sim, shots=SHOTS)
# counts_dist = board.onehot_counts_to_distribution(counts_dict=counts)
# counts_dist = np.flip(counts_dist)
# print(f"observed: {counts_dist}")
# standard_exp = board.compute_exponential_distribution(n+1, LAM)
# print(f"expected: {standard_exp}")
# exp_diff = board.compute_tvd(counts_dist, standard_exp)
# print(f"exponential distribution error: {exp_diff}")

# plot_histogram(counts)
# plt.show()

SHOTS = 1024
N = 3
LAM = 0.2
RUNS = 30

board = GaltonBoard(N)
results = benchmark_exponential_distribution(board, lam=LAM, shots=SHOTS, runs=RUNS)
