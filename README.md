## 📁 Project Structure

```
galton_box_monte_carlo/
├── src/
│   └── galton.py          # Core circuit builder and statistical tools
├── examples/
│   └── run_galton.py      # Main script to build, run, and evaluate the circuit
├── notebooks/
│   └── peg.ipynb          # Exploratory notebook
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Background

This project is an implementation of the concepts from:

> **Universal Statistical Simulator**  
> *by Mark Carney, and Ben Varcoe*
[arXiv:2202.01735](https://arxiv.org/pdf/2202.01735)

The paper introduces a universal framework to simulate classical probability distributions using quantum circuits. The method is based on a Galton board analogy, where a quantum particle undergoes a discrete-time walk across multiple layers of beam splitters, simulated using Hadamard gates and controlled swaps. By manipulating quantum interference and measurement, the circuit can reproduce various statistical distributions including binomial, exponential, and more.

---

## Usage

### Running the Simulation

```bash
cd galton_box_monte_carlo
python examples/run_galton.py
```

This builds and executes a multi-layer Galton board and prints out:

- Measurement counts
- Empirical vs expected statistics
- Mean squared error against binomial model

For a circuit diagram display, check out peg.ipynb . 

---

## Example Output

```bash
Galton Box level: 3; pegs: 6
MSE: 0.000879
Empirical mean: 0.250, variance: 0.013
Expected mean: 0.250, variance: 0.016
---------------------------------------------------------------------------------
Empirical data: {0.11956521739130435, 0.3705533596837945, 0.3601778656126482, 0.14970355731225296]
```

---

## Next Steps (as per challenge brief)

- [x] Generalized n-layer Galton board circuit
- [ ] Implement exponential distribution shaping
- [ ] Simulate quantum Hadamard walk
- [ ] Add noise model optimization
- [ ] Compute Wasserstein / MSE distances under stochastic noise

---

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

Recommended Python version: **3.10+**

---

## References

- [Quantum Galton Board Paper (PDF)](https://arxiv.org/pdf/2202.01735)
- Qiskit Documentation: https://qiskit.org/documentation/
- [Scipy binomial](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binom.html)

---

## Acknowledgments

Developed as part of the **Womanium Quantum Challenge 2025** under the Quantum Walks and Monte Carlo simulation track.

---

## 🔗 License

MIT License.
