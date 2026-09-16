# *Pauli Correlation Encoding and Attention are All You Need!*


**A Hybrid Quantum-Classical Generative Circuit Synthesis via Transformer and Pauli Correlation Encoding for N-1-Secure Storage
Siting and Microgrid Formation under Data-Center Load Growth**

---


<br/>

<img width="967" height="177" alt="image" src="https://github.com/user-attachments/assets/83f1a37c-03a6-4b95-9833-ed4beabea277" />


**Team eQoSystem** &nbsp;·&nbsp;

| Achraf Boussahi | Abir Chekroun | Zakaria Lourghi |
|:---:|:---:|:---:|
| Team Lead & Quantum Expert | Quantum Expert / Commercial Lead | Software Engineer / AI & Data Science |
| [@AchrafBoussahi](https://github.com/AshrafBoussahi) | @Abeer | @ZakariaLer |

<img width="1662" height="641" alt="image" src="https://github.com/user-attachments/assets/9d35de0e-6391-4b16-a692-8db83d2394c2" />


</div>



[<img src="https://qbraid-static.s3.amazonaws.com/logos/Launch_on_qBraid_white.png" width="150">](https://account.qbraid.com?gitHubUrl=https://github.com/ashrafboussahi/qGridX.git)

Click the button to open this repository on qBraid with the environment ready.
Then run `notebooks/01_quickstart.ipynb`, which goes from real power flow
physics to a buildable, contingency-screened storage plan in a few minutes on
the free tier.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

---
## Why? and for what?
The **U.S. Department of Energy** launched its **Quantum Genesis initiative** in
June 2026, following the presidential executive order on quantum information
science and within the broader Genesis Mission
established by the White House in November 2025. The
initiative's stated goal is the deployment of the world's first
fault-tolerant, scientifically relevant quantum computing capability by
2028, with a priority on evaluating near-term quantum methods against the
hard combinatorial problems the energy sector faces. Therefore our work is a final
submission to the DOE Office of Technology Commercialization's 2026
Global Industry Challenge on Quantum-Enhanced Strategic Siting of Energy
Storage and Microgrids, and it aims to contribute to DOE's program of
building and assessing quantum optimization tools for planning-grade grid
decisions.


<img width="1272" height="275" alt="image" src="https://github.com/user-attachments/assets/67c9f175-5459-4f43-bc38-52029882a73a" />


## Why us in this challenge?

Our team has already won first place globally in the Rigetti advanced quantum challenge on grid optimization, where we presented a solution based on Pauli correlation encoding. You can read more about our solution and the very positive comments from the jury in this blog: https://www.aiqcommunity.org/news/q-volution-first-place



## The problem this solves

A planner siting storage on a 1,000-bus system, choosing among 500 candidate
sites at nine capacity tiers each, faces 4,500 binary decisions. Add a microgrid
boundary decision per site and the count grows again. Map one decision to one
qubit, as almost every quantum optimization proposal does, and that problem
needs 4,500 qubits at a fidelity no announced device offers. The gap is not a
few years of hardware progress. It is three orders of magnitude, and it is a
property of the encoding rather than of the hardware roadmap.

qGridX changes the encoding. Decision `i` lives in the **sign** of a `k`-body
Pauli string's expectation value, so an `n`-qubit register carries `3 * C(n, k)`
decisions:

| Decisions | One qubit each | qGridX | Ratio |
|---:|---:|---:|---:|
| 45 | 45 qubits | **6 qubits** | 7.5 : 1 |
| 105 | 105 qubits | **7 qubits** | 15 : 1 |
| 2,772 | 2,772 qubits | **12 qubits** | 231 : 1 |
| 9,009 | 9,009 qubits | **15 qubits** | 600 : 1 |
| 4,500 (planner scale) | 4,500 qubits | **14 qubits** | 321 : 1 |

The last row is a capacity calculation. Every other row was executed end to end.

There is a second property that matters just as much operationally. The Pauli
strings partition into three mutually commuting families, so **three
measurement settings read every decision**, at any problem size. Measured on a
108-qubit superconducting processor, 45 decisions and then 105 decisions were
both recovered from exactly three device jobs and 3,072 shots. Doubling the
problem changed neither number.

---
We also introduce the qGridX package that containes our code base and solution! Enjoy our submission!

## Install

```bash
pip install qgridx
```

From a clone, for development or to reproduce results:

```bash
git clone https://github.com/ashrafboussahi/qGridX.git
cd qGridX
pip install -e .
```

Optional extras:

```bash
pip install qgridx[hardware]   # submit circuits to a real device
pip install qgridx[mip]        # standalone HiGHS solver interface
pip install qgridx[all]        # everything, for a full reproduction
```

Python 3.10 or newer. The core dependencies are NumPy, SciPy, pandas, PyTorch,
NetworkX and Matplotlib, all of which are already present in the qBraid base
image.

---

## Sixty seconds to a plan

```python
from qgridx.problems import load_registry, rebuild_instance
from qgridx.pipeline import solve

reg = load_registry()                      # 160 instances, certified optima
row = reg[reg.grid == "IEEE-118"].iloc[0]
inst = rebuild_instance(row)               # rebuilt from DC optimal power flow

plan = solve(inst, evals=1500)

print(plan.n_qubits, "qubits")             # 6
print(plan.n_sites, "sites,", plan.sited_mw, "MW")
print(plan.cost, "vs certified", row.optimum_value)
```

`solve` does four things in order: assigns each decision to a Pauli string,
trains a transformer to write circuits, executes the best circuit it found, and
repairs the resulting correlation signs into a plan that satisfies the capital
budget and the tier structure. Infeasible output is structurally impossible, so
whatever the noise does, you receive something buildable.

To check the install without waiting:

```bash
qgridx benchmark --smoke
```

---

## Reproducing every reported result

Each command below writes a CSV and prints the summary table it produced.
Archived copies of every one of those files ship inside the package, so a fresh
run can be compared against the published numbers without rerunning anything.

```bash
qgridx --list                    # every experiment and where output goes
```

| Command | What it reproduces | Scale |
|---|---|---|
| `qgridx benchmark` | Main comparison table: the correlation solver against simulated annealing, tabu search, greedy restarts and an unconstrained-input control, all at a matched 10,000-evaluation budget | 80 instances |
| `qgridx gate_budget` | The design sweep that identified the generator's gate allowance as the one choice controlling solution quality | 6 arms, 24 instances |
| `qgridx scenarios` | Thirteen-scenario stochastic formulation, and the demonstration that scenario count costs nothing at the quantum layer | 40 instances |
| `qgridx resilience` | N-1 security screening under a 50 to 500 MW data-center load, with and without the sited storage on identical outages | 120 cells, about 32,000 contingency solves |
| `qgridx microgrid` | Storage sizing plus microgrid islanding decisions in one encoding, and what the extra decisions cost in qubits | 64 instances |
| `qgridx resources` | Measured resource table across problem sizes, and the planner-scale capacity projection | 8 configurations |
| `qgridx hardware --stage validate` | Offline verification of the device path. Free, and the recommended first step | 2 circuits |

Useful flags on every experiment:

```bash
qgridx resilience --smoke               # a few instances, reduced budget
qgridx benchmark --workers 16           # parallel processes
qgridx microgrid --per-config 8         # instances per configuration
qgridx resilience --shard 0/3 --tag _a  # split a sweep across machines
```

**Start with `--smoke`.** The full runs are hours of CPU. A smoke run exercises
the identical code path on a handful of instances and confirms the environment
first.

### Regenerating the figures

```python
from qgridx.analysis import figures
figures.render_all("out/")
```

Every figure reads the same result file the corresponding experiment wrote, so
a figure cannot drift away from the number it illustrates.

---

## The package, module by module

qGridX is layered. The grid layer knows what a bus is; nothing above it does.
The quantum layer sees only coefficients. That separation is what lets the same
solver run on four different networks without special-casing any of them.

### `qgridx.grid` : power system physics

The only part of the package that models a network.

| Module | Purpose |
|---|---|
| `grid.cases` | IEEE 14, 30, 57 and 118-bus test systems. Every case exposes the same surface (`BUS`, `BRANCH`, `GEN`, `bus_index`, `compute_ptdf`), which is what makes cross-grid sweeps uniform. `get_case("IEEE-118")` looks one up by the name used in the result files. |
| `grid.dcopf` | DC optimal power flow. Returns locational marginal prices and line-limit duals, which become the objective coefficients. |
| `grid.contingency` | N-1 screening. Rebuilds the post-outage transfer factors with the branch genuinely removed from the susceptance matrix, then solves a security-constrained power flow with per-bus load shedding, so unserved energy is a computed quantity rather than a proxy. |

### `qgridx.problems` : turning a grid into a binary program

| Module | Purpose |
|---|---|
| `problems.siting` | Builds the storage siting family. One capacity tier is a 25 MW / 100 MWh four-hour battery block, carried by a monotone domain-wall chain per bus. Benefit follows congestion; a quadratic term penalizes crowding one corridor. |
| `problems.microgrid` | Adds one islanding binary per candidate bus, so `m = B*L + B`. Islanding alone is strictly costly; the payoff is a negative coupling to that bus's siting bit, scaled by the bus's **measured** outage exposure. A bus that never sheds earns nothing. |
| `problems.scenarios` | Folds a weather scenario set into the coefficients. Because siting is a first-stage decision, the expected-value problem is a binary program of exactly the same size, which is why thirteen scenarios cost the same qubits as one. |
| `problems.registry` | Loads the frozen 160-instance benchmark and rebuilds any instance exactly from its seed. Instances are never stored as matrices; they are rebuilt through the physics layer, so a change there is visible rather than bypassed. |
| `problems.domainwall` | Encoding helpers for the monotone capacity chains. |

### `qgridx.encoding` : decisions as correlation signs

| Module | Purpose |
|---|---|
| `encoding.families` | Enumerates the three commuting Pauli families and assigns decisions to strings, randomly or with a coupling-aware heuristic. `max_capacity(n, k)` is the `3 * C(n, k)` ceiling. |
| `encoding.simulator` | Exact statevector primitives. Note the convention: qubit 0 is the **most significant** bit of the basis index, which is the opposite of most platforms and matters the moment you talk to a device. |
| `encoding.loss` | Correlator extraction from a state, plus the surrogate loss the directly trained reference arm optimizes. |
| `encoding.correlators` | Walsh-Hadamard extraction of every correlator in a family from one measurement record. This is the change that made 9,009 decisions tractable at all. |
| `encoding.shot_noise` | Finite-shot sampling model and the sign-flip law it implies. |

### `qgridx.generator` : a transformer writes the circuit

| Module | Purpose |
|---|---|
| `generator.vocab` | Gate-token vocabulary: axis rotations, entangling rotations, a discrete angle grid, and the sequence-length ceiling that turned out to be the binding design choice. |
| `generator.model` | The decoder-only transformer. |
| `generator.executor` | Exact execution of a token sequence via a precomputed per-token unitary table. |
| `generator.executor_scalable` | Batched executor with memory cost `O(2^n)` independent of vocabulary size, needed past nine qubits. |
| `generator.reward` | Decoded-cost reward, including the margin-aware variant that scores a circuit by whether its decisions would survive finite-shot read-out. |
| `generator.train` | The loop: sample circuits, decode each to a feasible plan, rank the group by cost, update the transformer. No gradient is ever taken through a circuit parameter, so there is no barren-plateau surface to fall into and no ansatz to guess. |

### `qgridx.decoder` : from signs to a buildable plan

| Module | Purpose |
|---|---|
| `decoder.repair` | Sign read-out, domain-wall monotonicity repair, budget repair, and a joint neighbourhood search over simultaneous multi-bus tier moves. |
| `decoder.ilp` | Maximum-a-posteriori decode expressed as a small integer program. |
| `decoder.portfolio` | Runs both decode paths and keeps the better plan. Every solver arm in every reported comparison shares this, so differences between arms come from the input alone. |

### `qgridx.baselines` : the classical references

| Module | Purpose |
|---|---|
| `baselines.mip` | Exact branch-and-bound through HiGHS, with the quadratic term linearized by a McCormick envelope. Supplies the certified optima everything is scored against. |
| `baselines.heuristics` | Simulated annealing and tabu search on the domain-wall representation. |
| `baselines.oracle` | Brute-force enumeration, for small instances only. |

Reporting these is part of the method. At the scale where optima can be
certified, an exact solver finishes in about four milliseconds and good
metaheuristics beat the quantum arm on solution quality. What qGridX claims is a
resource profile, not a speed-up.

### `qgridx.hardware` : running on a real device

| Module | Purpose |
|---|---|
| `hardware.qasm` | Emits device-ready OpenQASM 3 from a token sequence. |
| `hardware.runner` | Submits jobs and retrieves counts, tolerating the different result shapes providers return. Finished jobs are addressable by id, so an interrupted campaign is completed by collecting rather than by re-executing and re-paying. |
| `hardware.analysis` | Reconstructs every correlator from the three measurement records and scores sign agreement and magnitude retention against simulation. |
| `hardware.verify` | Offline checks that must pass before any device time is spent. |

Two findings are baked into this subpackage because both cost real debugging
time against the live device, and both will recur for anyone repeating the work:

1. **Submit OpenQASM 3, not OpenQASM 2.** The QASM 2 ingestion path could not
   compile these circuits at all. Its server-side translation to Quil schedules
   each measurement immediately after that qubit's last gate rather than at the
   end of the program, which the device instruction format forbids. Reordering
   the measurements, inserting a barrier, and appending a trailing entangling
   ladder all failed. The identical circuit as QASM 3 compiles and runs.
   Rejected compilations are not billed, so iterating on this costs nothing.
2. **Determine the bit ordering, do not assume it.** This package indexes qubit
   0 as the most significant bit; most platforms do the opposite. Run the same
   program on a noiseless simulator first and score both conventions against the
   known correlators. In the reported campaign the right one reproduced them to
   the shot-noise floor (0.030 at 1,024 shots) and the wrong one gave 0.140,
   which is unambiguous. `hardware.verify.determine_bit_order` does this.

### `qgridx.maxcut` : the scaling stress test

Maximum cut is where the encoding is pushed hardest, because problem size *is*
the encoding capacity, so sweeping `(n, k)` sweeps problem size directly up to
9,009 decisions on 15 qubits. Reference cut values come from a low-rank
semidefinite relaxation solved by Burer-Monteiro factorization, which is a
strong heuristic and **not** a certificate: ratios above 1.0 mean the pipeline
beat that reference run, not that it exceeded the true optimum.

### `qgridx.analysis`, `qgridx.utils`, `qgridx.pipeline`, `qgridx.cli`

`analysis` holds the statistics (Wilcoxon signed-rank for paired comparisons,
Wilson intervals for proportions, McNemar's exact test for paired proportions)
and regenerates every figure from the archived results. The unit of analysis is
always the instance, never the individual run, so extra training restarts cannot
inflate a sample size. `utils.paths` resolves data and results relative to the
installed package, so a wheel and an editable checkout behave identically.
`pipeline` is the short path through everything. `cli` is the `qgridx` command.

---

## Data

Everything needed to reproduce is inside the package. Nothing is downloaded at
run time.

| Data | Location | Notes |
|---|---|---|
| Instance registry | `qgridx/data/registry/instances_v2.csv` | 160 instances across four grids with certified optima, generation seeds, and difficulty properties |
| RTS-GMLC series | `qgridx/data/rts_gmlc/*.csv.gz` | A full year of real five-minute demand, wind and solar, 105,408 intervals. Shipped gzipped; pandas decompresses transparently |
| Archived results | `qgridx/results/` | Every reported CSV and JSON, including raw device counts |

Access them from code:

```python
from qgridx.utils.paths import data_dir, results_dir, registry_csv
```

Set `QGRIDX_DATA` or `QGRIDX_RESULTS` to point elsewhere.

### About the benchmark family

The registry is deliberately calibrated to be non-trivial in both directions.
The empty plan is optimal on only 4.4 percent of instances, no instance is
solved by building everything at maximum tier, the optimum sites 2.0 buses on
average, and a single-start greedy heuristic certifies the optimum on 10
percent. Candidate buses are not hand-picked either: each grid's buses are
ranked by mean congestion-only locational value over forty stochastic demand
draws, which is also what a planner would do.

---

## What the results say

| Result | Number |
|---|---|
| Largest problem executed | 9,009 decisions on 15 qubits |
| Measurement settings | 3, at every problem size |
| Shots per circuit evaluation | 3,072, at every problem size |
| Generated versus directly trained circuits | better on 25 of 25 graphs, p = 6.0e-8 |
| Gate budget effect | 23.8 to 48.8 percent solved exactly, p = 2.4e-7 |
| Microgrid decisions added at B=4, L=10 | 40 to 44 decisions, **0 extra qubits** |
| Thirteen weather scenarios | decision changes on 77.5 percent, never worse under stress, no quantum cost |
| N-1 security at 200 MW added load | outages served in full: 36.4 to 96.1 percent |
| Device campaign | 45 and 105 decisions, both from 3 jobs and 3,072 shots |

### What is not claimed

No quantum speed-up, and no better solution quality than purpose-built
classical metaheuristics. At the 40 to 45 decision scale where optima can be
certified, an exact mixed-integer solver finishes in 3.9 milliseconds on
average, and simulated annealing solves 65.0 percent of instances exactly
against the correlation solver's 48.8 percent. Those numbers are in
`qgridx benchmark` output and in the paper, next to ours.

The claim is the resource profile: qubit count, circuit count and shot budget
all stay bounded as the decision count grows, which is the axis along which real
planning problems explode.

On hardware, the measurement economy holds exactly as designed, but the
correlator **values** do not survive at these circuit depths. Retention is 5.1
percent at 45 decisions and 2.5 percent at 105, with sign agreement at chance. A
two-qubit Bell control in the same session got 24 of 24 signs correct, which
places the shortfall in accumulated circuit error rather than in the encoding or
the read-out pipeline. That is the honest state of the art for this method on
this hardware, and it is a quantified bar rather than a vague one: on-device
optimization needs magnitudes surviving to about `1/sqrt(N)`, or 0.03 at 1,024
shots.

---

## Provenance

`PROVENANCE.md` records where every module came from. Nothing was reimplemented
for the package: module bodies were copied verbatim from the research tree and
only their imports were rewritten, so the code you install is the code that
generated the numbers. Work that does not back a reported result was left out
rather than shipped as filler.

## Citing

See `CITATION.cff`, or:

> A. Boussahi, *qGridX: qubit-efficient transmission planning with
> correlation-encoded optimization*, version 1.0.0, 2026.

## License

MIT. See `LICENSE`.
