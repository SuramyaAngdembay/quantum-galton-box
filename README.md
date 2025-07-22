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

> **Quantum Galton Boards and Monte Carlo Sampling**  
> *by Aram W. Harrow, Ashley Montanaro, and others*

The paper explores how quantum walks can be used to simulate statistical distributions (e.g., binomial, exponential) using quantum circuits — with applications to high-dimensional sampling and Monte Carlo methods.

We recreate the quantum Galton board structure and demonstrate how different gate patterns result in different output distributions, leveraging quantum interference and measurement collapse.

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
MSE: 0.000107
Empirical mean: 0.200, variance: 0.015
Expected mean: 0.200, variance: 0.015
---------------------------------------------------------------------------------
Empirical data: {0.06620553359683795, 0.2554347826086957, 0.3759881422924901, 0.2425889328063241, 0.059782608695652176]
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

- [Quantum Galton Board Paper (PDF)](./galton_board.pdf)
- Qiskit Documentation: https://qiskit.org/documentation/
- [Scipy binomial](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binom.html)

---

## Acknowledgments

Developed as part of the **Womanium Quantum Challenge 2025** under the Quantum Walks and Monte Carlo simulation track.

---

## 🔗 License

MIT License.
