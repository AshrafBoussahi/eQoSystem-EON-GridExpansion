# genpce: discrete circuit synthesis for Pauli-correlation-encoded optimisation

Research code behind the circuit-synthesis results in team eQoSystem's E.ON Phase 1 proposal. The library combines

* **Pauli correlation encoding (PCE)**: `m <= 3 * C(n, k)` binary variables read off the signs of `k`-body Pauli correlators of an `n`-qubit state, with only three measurement settings (Sciorilli et al., *Nature Communications* 16, 476, 2025), and
* **discrete, compile-free circuits**: the state is prepared by a sequence of native gate tokens (single-qubit rotations on a fixed angle grid plus a CZ brickwork matching device connectivity) chosen by a classical search, so the circuit carries no trainable parameter and no gradient is ever taken on the device.

Everything is Qiskit-native. Each circuit is a `QuantumCircuit`; the same three X, Y, Z measurement circuits feed the exact statevector evaluator, the finite-shot evaluator, Aer noise-model simulation, and the Qiskit Runtime submission path. Read-out is a Walsh-Hadamard transform over the three count tables that returns every correlator at once in `O(n 2^n)` per basis, independent of `m`, and it is tested against `Statevector.expectation_value(Pauli)` to 1e-10.

## Results this code produced

| Result | Numbers | Script | Output |
|---|---|---|---|
| Discrete search against variational PCE at matched quantum budget | leads by 4 to 9 points of approximation ratio below 5,000 circuit executions at m = 60, 252 and 858 (n = 6, 9, 13); variational overtakes above 1e4 to 1e5 executions | `experiments/e1_budget_curves.py` | `results/e1/final_all.csv`, `results/e1/curves_all.csv`, `results/figs/e1_quality_vs_budget.png` |
| The smoothed (tanh) PCE objective as a shot-robustness device | m = 252: exact objective alone decodes to 0.591 from 1,000 shots (median correlator 0.019); with the smoothed term 0.698 (0.109) | `experiments/e2_reward_robustness.py` | `results/e2/reward_m252.csv` |
| Scaling of the discrete search | plateaus by about 20,000 evaluations at every size; m = 858 on 13 qubits | `experiments/e3_scaling.py` | `results/e3/` |
| Complete single-edit landscape of PCE circuits | 1,152 children per parent, 160 parents, two instances; edit quality predictable with Spearman 0.95 around good circuits; a correlator-response head is what transfers across instances (0.40 against 0.13) | `experiments/e7_local_landscape.py` | `results/e7/learner_matrix_m60.md`, `results/e7/geometry_hamming_m60.csv` |
| Representability of variational states by discrete circuits | an 8-layer CZ brickwork with free single-qubit gates reproduces every variational teacher state (correlation distance about 1e-4, all 60 signs) | `experiments/e10_phase_a.py`, `e10_grid_resolution.py` | `results/e10/summary_ceiling_*.csv`, `results/figs/e10_phase_a_reg3_m60_s0.png` |

Additional ablations (learned proposal distributions, preference-based training, learned edit policies with a 120-run paired confirmation, cross-instance priors) are in `experiments/e4_*`, `e5_*`, `e6` logs, `e8_*`, `e9_*` with their CSVs under `results/`; they are reported in full there and do not change the table above.

Benchmark instances are random 3-regular and Erdos-Renyi (average degree 4) graphs at m in {60, 104/105, 168, 252, 858}, with exact optima by mixed-integer programming for m <= 120 and 30-restart simulated-annealing best-known values above that (`results/instances/`). The optimum of an instance is never used by any solver, only to report approximation ratios.

## Layout

```
genpce/
  pce/        correlator sets, Walsh-Hadamard read-out, sign decoding, relaxed loss (NumPy and torch)
  sim/        evaluators: exact statevector, finite-shot, Aer noise models; same X/Y/Z circuits as hardware
  pool/       token vocabularies (linear-chain and brickwork templates), gate and non-Clifford counting
  problems/   Max-Cut instances, generators, exact (HiGHS MILP) and heuristic (SA) solvers, bit-swap post-processing
  model/      small decoder-only transformer with KV-cache sampling
  train/      evolutionary search, replay buffer, reward shaping, learned edit scorers, distillation
  baselines/  variational PCE (brickwork ansatz, parameter shift), differentiable torch path validated against Qiskit
experiments/  e0 to e10 scripts (resumable, CSV outputs) and plotting
tests/        pytest suite (read-out against Qiskit Pauli expectation values, torch against Qiskit statevectors, solvers)
results/      instances with certified or best-known values, CSV logs, figures (large .npz/.pt artefacts are not tracked)
```

## Install and test

```bash
pip install -e ".[dev,hardware]"
pytest -q          # 26 tests
```

Python 3.11 or newer; Qiskit 1.3 or newer and qiskit-aer 0.15 or newer are pulled in automatically.

## Reproduce

```bash
python -u experiments/e1_budget_curves.py --families reg3 er4 --sizes 60 252 --seeds 0 1
python -u experiments/e2_reward_robustness.py --families reg3 --sizes 252 --seeds 0
python -u experiments/e3_scaling.py --families reg3 --sizes 60 105 168 252 --budget 100000
python -u experiments/e7_local_landscape.py --m 60 --parents 60 60 40 --budget 20000
python experiments/plot_e1_e4.py
```

Every script is seeded, resumable, and appends to CSV, so an interrupted run continues rather than restarts.

## Quantum budget accounting

A real device cannot use automatic differentiation, so the variational baseline is charged the parameter-shift cost of `2 N_p + 1` circuit executions per optimiser step, each needing three measurement settings. The discrete search is charged one execution per proposed circuit. This is the currency of every budget comparison above.

## Licence

Apache-2.0 (see `LICENSE`).
