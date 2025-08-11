Project: Quantum Galton Board as a Universal Statistical Simulator
Team: Gorkhalis
Members (WISER IDs): <Prajwal – gst-9ZHH1Kj14wFrIIO> · <Suramya – gst-wYBMQBj2KrRSykZ> · <Name – ID>
Program: WISER 2025 Quantum Projects

Summary (≈500 words)
We implement a Quantum Galton Board (QGB) that maps a Galton box to a quantum circuit so a single circuit pass creates a superposition of all trajectories and samples a target distribution. This is a compact, intuitive playground for Monte-Carlo-style simulation on quantum hardware and a clean testbed for noise and mitigation in the NISQ era—relevant to transport, finance, and uncertainty quantification. (Project scope per brief: general n-layer circuit; additional targets—exponential and Hadamard; noisy simulations; and distance metrics.) Project details

Objectives. (1) Generalize the QGB to n layers; (2) reproduce three targets—Hadamard walk, classical/binomial, and exponential; (3) build a robust benchmarking harness to compare noiseless vs. noisy runs with standardized post-processing and metrics (TVD, KL, Hellinger), plus leakage (mass off support) and integrity (fraction of one-hot strings).

Circuit design. We use a modular peg/shift construction closely following the reference QASM: prepare the coin, then apply controlled-SWAPs to realize left/right motion, repeated level-by-level. Our first pass showed a subtle right-bias; we fixed it by removing any CX(position→coin) and using proper control-on-0 branching, then consolidated into a symmetric “H-then-shift” step each level. We standardized endianness so '…001' is the rightmost bar and always measure only the position register. 

Targets.
• Hadamard walk (n=4): a fixed 5-bin analytic distribution embedded on the 10-slot full grid (zeros in between bins).
• Classical/binomial: verification target using ${n \choose k} / 2^n$

• Exponential: per-peg Rx(θ) biasing per as depth = level / total layers
 theta = arcsin(sqrt(1/(lam*n*depth))), n=total pegs given by triangle number 


Benchmarking. Our benchmark.py always post-processes to the full grid (raw bitstring length 
𝐿
L), embeds the analytic target onto that grid with zeros on “forbidden” slots, and chooses offset parity by best match to the clean run. We report: TVD(noisy,clean), TVD/KL vs target, Hellinger(noisy,clean), plus leakage and integrity. This makes apples-to-apples comparisons across targets and seeds straightforward.

Representative baseline (n=4, seeds=8).

Hadamard: clean vs analytic TVD ≈ 0.0086; noisy vs clean TVD ≈ 0.2525; leakage_noisy ≈ 0.1657; integrity_noisy ≈ 0.3845.

Exponential (λ≈0.7): clean vs analytic TVD ≈ 0.1910; noisy vs clean TVD ≈ 0.2314; leakage_noisy ≈ 0.0263; integrity_noisy ≈ 0.4202.

Binomial: clean vs analytic TVD ≈ 0.0046; noisy vs clean TVD ≈ 0.1943; leakage_noisy ≈ 0.1864; integrity_noisy ≈ 0.3590.
(We include JSON + PNG overlays for each run in results_fullgrid/.)

Impact.

A reference-quality, symmetric QGB step that others can reuse; 2) a target-agnostic benchmarking harness with leakage/integrity, enabling measurable progress on noise optimization; 3) a clear path from circuits → distributions for Monte-Carlo tasks.

Future scope. Optimize noisy accuracy and depth: smarter CSWAP decompositions and layout, dynamical decoupling, readout mitigation, transpiler tuning; per-layer θ-calibration for exponential; and (time permitting) MCMR/RUS to discard corrupted shots. 2202.01735v1

Deck: slides/<final_presentation>.pdf (includes overlays + key metrics).
How to run: python src/benchmark.py --target hadamard --steps 4 --shots 4096 --seeds 8 --out_dir results_fullgrid --save_json results_fullgrid/hadamard_n4.json