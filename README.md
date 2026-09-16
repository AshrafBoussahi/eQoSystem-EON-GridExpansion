# Pauli Correlation Encoding for Distribution Grid Expansion Planning

**Team eQoSystem** (AiQC, Ai and Quantum Community) | 2026 Global Quantum + AI Challenge, Phase 1 | Problem statement: **E.ON, Quantum-Enabled Grid Expansion Planning for Distribution System Energy Networks**

Achraf Boussahi (lead), Abir Chekroun, Zakaria Lourghi, Mouadh Abderrahmane Assal, Hiba Menacer

[<img src="https://qbraid-static.s3.amazonaws.com/logos/Launch_on_qBraid_white.png" width="150">](https://account.qbraid.com?gitHubUrl=https://github.com/AshrafBoussahi/eQoSystem-EON-GridExpansion.git)

Click the button to open this repository on qBraid with the environment ready, then run `qgridx/notebooks/01_quickstart.ipynb`. It goes from real power-flow physics to a buildable, N-1-screened plan on six qubits in a few minutes on the free tier.

---

## What this repository is

This is the public companion of our Phase 1 concept proposal. It holds three things:

| Folder | What it is | Status |
|---|---|---|
| `proposal/` | The concept proposal (6 pages) and supplementary appendix (3 pages), with LaTeX sources and the script that draws every figure from the result files below | final |
| `qgridx/` | **qGridX**, our qubit-efficient grid planning package: power-flow layer, Pauli correlation encoding, transformer circuit writer, decoder, N-1 security screen, statistics and hardware path. Snapshot of the public release v1.0.1, also on [PyPI](https://pypi.org/project/qgridx/) and [GitHub](https://github.com/ashrafboussahi/qGridX) | released |
| `genpce/` | **genpce**, our Qiskit-native Pauli-correlation-encoding circuit-synthesis library: discrete gradient-free circuit search, Walsh-Hadamard three-setting read-out, variational PCE baseline, Aer and Qiskit Runtime paths, 26 unit tests | research code, reproducible |

Every number in the proposal comes from a result file in `qgridx/src/qgridx/results/` or `genpce/results/`, and every table has a one-command reproduction listed below. Nothing was run for the proposal itself; the figures are drawn from files that already existed.

## The idea in four lines

1. E.ON's problem grows along one axis: the number of build-or-no-build line decisions `m`.
2. One-qubit-per-variable mappings need `m` qubits. Pauli correlation encoding stores decision `i` in the sign of a `k`-body Pauli correlation, so `n` qubits carry up to `3 * C(n, k)` decisions: 45 on 6, 105 on 7, 2,000 on 18, 9,009 on 15.
3. The Pauli strings split into three commuting families, so the device runs **three measurement settings whatever `m` is**. We measured this on a 108-qubit processor: 45 decisions and then 105 were read from three jobs and 3,072 shots each.
4. The circuit is written by a classical generator (a transformer, or an evolutionary search over gate tokens) and trained on the cost of the decoded, feasible plan. No angle lives on the device, no gradient crosses into it.

## Evidence already in hand

| Claim in the proposal | Where it is reproduced | Result file |
|---|---|---|
| 3 measurement settings at 45 and 105 decisions on Rigetti Cepheus-1-108Q; Bell control 24 of 24 signs | `qgridx`: `qgridx hardware --stage analyze` (notebook `02_hardware.ipynb`) | `qgridx/src/qgridx/results/doe_phase3/g3_cepheus.csv`, `results/hardware/bell_tetrahedron_qcs_summary.csv` |
| 9,009 decisions on 15 qubits; compression and constant measurement cost | `qgridx resources` | `qgridx/src/qgridx/results/doe_phase3/e7_resource_table.csv` |
| Gate budget 19 to 59 lifts exact-match from 23.8% to 48.8% (80 instances) | `qgridx gate_budget` | `qgridx/src/qgridx/results/doe_phase3/e2m_expressivity_full80.csv` |
| Benchmark table against MIP, SA, tabu, greedy at 10,000 evaluations | `qgridx benchmark` | `qgridx/src/qgridx/results/doe_phase3/e2h_main_table_v2.csv` |
| 13 weather scenarios at zero quantum cost | `qgridx scenarios` | `qgridx/src/qgridx/results/doe_phase3/e3b_scenario_weighted.csv` |
| N-1 screening: 36.4% to 96.1% secure at 200 MW, 93.5% less unserved energy | `qgridx resilience` | `qgridx/src/qgridx/results/doe_phase3/g1_resilience.csv` |
| Microgrid decisions at zero extra qubits, 60.9% exact on 64 instances | `qgridx microgrid` | `qgridx/src/qgridx/results/doe_phase3/g2_microgrid.csv` |
| Discrete PCE search leads variational PCE below 5,000 executions at 60, 252 and 858 variables | `genpce`: `experiments/e1_budget_curves.py` | `genpce/results/e1/final_all.csv`, `curves_all.csv` |
| Smoothed term restores shot robustness (0.591 to 0.698 at 1,000 shots, m = 252) | `genpce`: `experiments/e2_reward_robustness.py` | `genpce/results/e2/reward_m252.csv` |
| Scaling of the discrete search to m = 858 | `genpce`: `experiments/e3_scaling.py` | `genpce/results/e3/` |
| First place, Rigetti advanced track, Q-Volution 2026 (180-node grid, 12 qubits, Ankaa-3) | separate repository | https://github.com/AshrafBoussahi/Rigetti_Quantum_Grid_Optimization_eQoSystem |

`qgridx --list` prints every reproduction command of the grid package.

## Install

Python 3.11 or newer (genpce pins 3.11; qGridX accepts 3.10).

```bash
git clone https://github.com/AshrafBoussahi/eQoSystem-EON-GridExpansion.git
cd eQoSystem-EON-GridExpansion
pip install -e ./qgridx
pip install -e "./genpce[dev]"
```

Optional extras for the grid package: `pip install "qgridx[hardware]"` to submit circuits through qBraid, `pip install "qgridx[mip]"` for the standalone HiGHS interface.

Quick checks:

```bash
qgridx benchmark --smoke          # full grid pipeline in about a minute
cd genpce && pytest -q            # 26 tests, including read-out against Qiskit
```

## Qiskit compatibility layer

E.ON asks for a compatibility layer so results can be executed and validated in Qiskit. It exists on both sides of this repository:

- `genpce` builds every circuit as a Qiskit `QuantumCircuit`. The same three X, Y, Z measurement circuits are used by the exact statevector evaluator, by Aer with and without noise models, and by the Qiskit Runtime submission path (`genpce/sim/evaluator.py`, `genpce/pce/correlators.py`). Read-out is a Walsh-Hadamard transform over the three count tables that returns all `m` correlators at once; `tests/test_correlators.py` checks it against `Statevector.expectation_value(Pauli)` to 1e-10.
- `qgridx` emits OpenQASM 3 for its generated circuits (`qgridx/hardware/qasm.py`), which Qiskit loads with `qiskit.qasm3.loads`. That is the path we used on Rigetti devices through qBraid.

## Reproducing the proposal figures

```bash
cd proposal/source
python make_proposal_figures.py     # reads qgridx/ and genpce/ result files, writes figures/*.pdf
pdflatex proposal.tex && pdflatex proposal.tex
pdflatex appendix.tex && pdflatex appendix.tex
```

## Layout of `genpce/`

```
genpce/
  genpce/
    pce/         correlator sets, Walsh-Hadamard read-out, sign decoding, relaxed (tanh) loss
    sim/         evaluators: exact statevector, finite-shot, Aer noise models; one decoder for all
    pool/        token vocabularies (native single-qubit rotations, CZ brickwork templates)
    problems/    Max-Cut instances and generators, exact (MILP) and heuristic (SA) solvers, bit-swap
    model/       small decoder-only transformer with KV-cache sampling
    train/       evolutionary search, replay buffer, reward shaping, learned edit scorers, distillation
    baselines/   variational PCE (brickwork ansatz, parameter shift), validated against Qiskit
  experiments/   e0 to e10 scripts, resumable, CSV outputs
  results/       instances with certified or best-known values, CSV logs, figures
  tests/         pytest suite
```

See `genpce/README.md` for the per-experiment commands.

## Licence and citation

The proposal and `qgridx/` are MIT (root `LICENSE`, `qgridx/LICENSE`); `genpce/` is Apache-2.0 (`genpce/LICENSE`), as declared in its `pyproject.toml`. If you use this work, cite `CITATION.cff` at the root, or the qGridX release on PyPI.

## Contact

Achraf Boussahi, ashraf@aiqcommunity.org. Team page: https://www.aiqcommunity.org/team
