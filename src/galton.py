from qiskit import QuantumCircuit, transpile
from qiskit.visualization import plot_histogram
from qiskit_aer import AerSimulator
import numpy as np
from scipy.stats import binom
import matplotlib.pyplot as plt 

SHOTS = 2024

class GaltonBoard():
    def __init__(self, n):
        self.n = n
        self.theta_scheduler = None
        print(f"galton circuit of size {n} has been created.")

    # initialize the required circuit the n-level board
    def build_circuit(self):
        q_num = 2*(self.n+1) 
        qc = QuantumCircuit(q_num, q_num)
        return q_num, qc
    
    # one peg
    def one_peg(self, qc, start):
        qc.cswap(0, start-1, start)
        qc.cx(start, 0)
        qc.cswap(0, start, start+1)
        return qc
    
    def one_peg_h(self, qc, start):
        qc.cswap(0, start-1, start)
        qc.x(0)
        qc.cswap(0, start, start+1)
        qc.x(0)
        return qc
    
    def multi_swap_u(self, qc, curr, curr_level):
        for i in range(curr_level):
            qc = self.one_peg(qc, curr)
            if (i != curr_level-1):
                qc.cx(curr+1, 0)
            curr += 2
        return qc 

    def multi_swap_h(self, qc, curr, curr_level):
        for _ in range(curr_level):
            qc = self.one_peg_h(qc, curr)
            curr += 2
        return qc 

    def multi_swap_fine_tuned(self, qc, curr, curr_level, t):
        final_cnot = []
        for i in range(curr_level):
            qc = self.one_peg(qc, curr)
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

    def theta_schedule_exp(self, lam):
        thetas = []
        alpha = 0.3269
        for level in range(1, self.n + 1):
            n = (self.n**2 + self.n) / 2
            depth = level / self.n
            theta_l = np.arcsin(np.sqrt(1/(lam*(n)*depth)))
            thetas.extend([theta_l] * level)
        return thetas
    
    def set_theta_scheduler(self, scheduler_fn):
        self.theta_scheduler = scheduler_fn


    def measure_gb(self,  qc):
        cb = 0
        for i in range(1, qc.num_qubits):
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
                qc = self.multi_swap_u(qc, curr=curr, curr_level=count)
            qc.reset(0)
            curr -= 1
            count += 1
        return self.measure_gb(qc)

    def exponential_gb(self, lam):
        qbits, qc = self.build_circuit()
        start = qbits // 2
        qc.x(start)
        curr = start
        curr_level = 1
        self.set_theta_scheduler(self.theta_schedule_exp)
        theta = self.theta_scheduler(lam)
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
        print(theta)
        return self.measure_gb(qc)

    def quantum_walk_gb(self):
        qbits, qc = self.build_circuit()
        start = qbits // 2
        qc.x(start)
        qc.h(0)
        qc.s(0)
        curr = start
        count = 1
        while curr != 1:
            qc.h(0)
            if count < 2:
                qc = self.one_peg_h(qc, curr)
            else:
                qc = self.multi_swap_h(qc, curr=curr, curr_level=count)
            curr -= 1
            count += 1
        return self.measure_gb(qc)   

    def run_circuit(self, qc, sim, shots=SHOTS):
        job = sim.run(transpile(qc, sim), shots=shots)
        result = job.result()
        return result.get_counts()

    def compute_exponential_distribution(self, n_bins, lam):
        """Compute target exponential distribution over discrete bins"""
        x = np.arange(n_bins)
        unnorm = np.exp(-lam * x)
        return unnorm / np.sum(unnorm)
    
    def compute_tvd(self, p, q):
        """Total Variation Distance between two discrete distributions"""
        return 0.5 * np.sum(np.abs(p - q))
    
    def onehot_counts_to_distribution(self, counts_dict, endianness='as_is', return_positions=False):
        if not counts_dict:
            return (np.array([]), np.array([])) if return_positions else np.array([])

        n_bins = len(next(iter(counts_dict)))
        counts = np.zeros(n_bins, dtype=float)

        for bitstring, c in counts_dict.items():
            # Flip so that a Qiskit string like '0001' maps to leftmost '1000'
            b = bitstring[::-1] if endianness == 'qiskit' else bitstring
            if b.count('1') != 1:
                continue  # post-select: skip non one-hot
            idx = b.index('1')           # 0..n_bins-1, left->right
            counts[idx] += c

        total = counts.sum()
        probs = counts / total if total > 0 else counts

        if return_positions:
            # centered integer coordinates for post-processing/metrics
            positions = np.array([2*i - (n_bins - 1) for i in range(n_bins)], dtype=int)
            return positions, probs
        return probs

    def bitstring_labels_left_to_right(self, n_bins):
        # '1000..0', '0100..0', ..., '000..01'
        labels = []
        for i in range(n_bins):
            s = ['0']*n_bins
            s[i] = '1'
            labels.append(''.join(s))
        return labels

    def plot_qgb(self,positions, probs, title="QGB distribution",
                target=None, target_name="Target", show_bitstrings=True, save=None):
        plt.figure(figsize=(8,5))

        xs = positions
        plt.bar(xs, probs, width=0.6, edgecolor='black', label='Observed')

        if target is not None:
            # side-by-side overlay
            width = 0.28
            plt.cla()  # clear and re-plot side-by-side if target is given
            plt.bar(xs - width/2, probs, width=width, edgecolor='black', label='Observed', alpha=0.9)
            plt.bar(xs + width/2, target, width=width, edgecolor='black', label=target_name, alpha=0.8)

        plt.title(title)
        plt.ylabel('Probability')
        if show_bitstrings:
            # use bitstrings as xtick labels in left->right order
            labels = self.bitstring_labels_left_to_right(len(positions))
            plt.xticks(xs, labels)
            plt.xlabel('One-hot bitstrings (left→right)')
        else:
            plt.xticks(xs)
            plt.xlabel('Position')

        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.legend()
        plt.tight_layout()
        if save:
            plt.savefig(save, dpi=200)
        plt.show()
    
    def view_circuit(self, qc):
        qc.draw('mpl')
        return plt.show()
 