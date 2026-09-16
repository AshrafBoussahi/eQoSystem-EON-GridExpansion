# Provenance

Every file in `qGridX` came from the research tree that produced the reported
results. Nothing was reimplemented for the package. Module bodies were copied
verbatim and only their imports were rewritten to the new paths, so the code
you install is the code that generated the numbers.

This file records the mapping so any result can be traced back to its origin.

## Library modules

| Research tree | qGridX |
|---|---|
| `qms/metrics.py` | `qgridx/analysis/metrics.py` |
| `qms/scoring.py` | `qgridx/analysis/scoring.py` |
| `qms/stats.py` | `qgridx/analysis/stats.py` |
| `experiments/paper_s2_w3a_sa_tabu_certification.py` | `qgridx/baselines/heuristics.py` |
| `qms/mip_baseline.py` | `qgridx/baselines/mip.py` |
| `qms/oracle.py` | `qgridx/baselines/oracle.py` |
| `qms/pce/ilp.py` | `qgridx/decoder/ilp.py` |
| `qms/pce/portfolio.py` | `qgridx/decoder/portfolio.py` |
| `qms/pce/decode.py` | `qgridx/decoder/repair.py` |
| `Project 2/qms/pce/fast_correlators.py` | `qgridx/encoding/correlators.py` |
| `qms/pce/pauli_families.py` | `qgridx/encoding/families.py` |
| `qms/pce/loss.py` | `qgridx/encoding/loss.py` |
| `qms/pce/shot_noise.py` | `qgridx/encoding/shot_noise.py` |
| `qms/pce/simulator.py` | `qgridx/encoding/simulator.py` |
| `qms/gqe/executor.py` | `qgridx/generator/executor.py` |
| `qms/gqe/executor_scalable.py` | `qgridx/generator/executor_scalable.py` |
| `qms/gqe/model.py` | `qgridx/generator/model.py` |
| `qms/gqe/reward.py` | `qgridx/generator/reward.py` |
| `qms/gqe/train.py` | `qgridx/generator/train.py` |
| `qms/gqe/vocab.py` | `qgridx/generator/vocab.py` |
| `qms/ieee_case118.py` | `qgridx/grid/cases/case118.py` |
| `qms/ieee_case1354pegase.py` | `qgridx/grid/cases/case1354pegase.py` |
| `qms/ieee_case14.py` | `qgridx/grid/cases/case14.py` |
| `qms/ieee_case30.py` | `qgridx/grid/cases/case30.py` |
| `qms/ieee_case57.py` | `qgridx/grid/cases/case57.py` |
| `qms/contingency.py` | `qgridx/grid/contingency.py` |
| `qms/dcopf.py` | `qgridx/grid/dcopf.py` |
| `Project 2/qms/maxcut_blind.py` | `qgridx/maxcut/blind.py` |
| `Project 2/qms/maxcut_bm.py` | `qgridx/maxcut/burer_monteiro.py` |
| `Project 2/qms/maxcut_io.py` | `qgridx/maxcut/graphs.py` |
| `Project 2/qms/maxcut_local_search.py` | `qgridx/maxcut/local_search.py` |
| `Project 2/qms/maxcut_pce_direct.py` | `qgridx/maxcut/pce_direct.py` |
| `Project 2/qms/sciorilli_ansatz.py` | `qgridx/maxcut/reference_ansatz.py` |
| `Project 2/qms/gqe/reward_maxcut.py` | `qgridx/maxcut/reward.py` |
| `Project 2/qms/gqe/train_maxcut.py` | `qgridx/maxcut/train.py` |
| `qms/encodings.py` | `qgridx/problems/domainwall.py` |
| `qms/instance_factory.py` | `qgridx/problems/siting.py` |
| `qms/seeding.py` | `qgridx/utils/seeding.py` |

## Experiment drivers

| Research tree | qGridX |
|---|---|
| `experiments/e2h_main_table_v2.py` | `qgridx/experiments/benchmark.py` |
| `experiments/e2o_final_levers.py` | `qgridx/experiments/gate_budget.py` |
| `experiments/e3b_scenario_weighted_qubo.py` | `qgridx/experiments/scenarios.py` |
| `experiments/g1_resilience_aiload.py` | `qgridx/experiments/resilience.py` |
| `experiments/g2_bess_microgrid.py` | `qgridx/experiments/microgrid.py` |
| `experiments/g3_cepheus_pce.py` | `qgridx/experiments/hardware.py + qgridx/hardware/{qasm,runner,analysis}.py` |
| `experiments/e7e8_resources_and_scaling.py` | `qgridx/experiments/resources.py` |
| `experiments/e3a_build_scenarios.py` | `qgridx/problems/scenarios.py` |
| `SubmissionV0/make_figures.py` | `qgridx/analysis/figures.py` |
| `SubmissionV0/make_report_figures.py` | `qgridx/analysis/figures.py` |

## Data and archived results

| Research tree | qGridX |
|---|---|
| `data/rts_gmlc/*.csv` | `qgridx/data/rts_gmlc/*.csv.gz (gzipped, 19 MB -> 5 MB)` |
| `results/doe_phase3/registry_v2/instances_v2.csv` | `qgridx/data/registry/instances_v2.csv` |
| `results/doe_phase3/e2h_main_table_v2.csv` | `qgridx/results/doe_phase3/` |
| `results/doe_phase3/e2m_expressivity_full80.csv` | `qgridx/results/doe_phase3/` |
| `results/doe_phase3/e2o_merged_done.csv` | `qgridx/results/doe_phase3/` |
| `results/doe_phase3/e3a_scenarios.csv` | `qgridx/results/doe_phase3/` |
| `results/doe_phase3/e3b_scenario_weighted.csv` | `qgridx/results/doe_phase3/` |
| `results/doe_phase3/e7_resource_table.csv` | `qgridx/results/doe_phase3/` |
| `results/doe_phase3/g1_resilience.csv` | `qgridx/results/doe_phase3/` |
| `results/doe_phase3/g2_microgrid.csv` | `qgridx/results/doe_phase3/` |
| `results/doe_phase3/g3_cepheus.csv` | `qgridx/results/doe_phase3/` |
| `results/doe_phase3/g3_manifest.json` | `qgridx/results/doe_phase3/` |
| `results/doe_phase3/g3_cepheus_raw.json` | `qgridx/results/doe_phase3/` |
| `hardware-results/candidates.csv` | `qgridx/results/hardware/` |
| `hardware-results/bell_tetrahedron_qcs_summary.csv` | `qgridx/results/hardware/` |
| `hardware-results/readout_confusion.csv` | `qgridx/results/hardware/` |
| `Project 2/results/phase2_campaign_results.json` | `qgridx/results/maxcut/` |
| `Project 2/results/phase3_scaling_results.json` | `qgridx/results/maxcut/` |

## What changed in the move

1. **Imports only.** Relative imports were resolved against each module's own
   package and rewritten to absolute `qgridx.*` paths. Import aliases were
   preserved exactly, so every name a module bound before it still binds.
2. **One layering fix.** `qgridx.grid.contingency` previously imported
   `QUBOInstance` at module scope purely for a type annotation. Since the
   problem layer imports the grid layer, that made the package circular on
   import. The annotation is now a string and the import is guarded by
   `TYPE_CHECKING`. No runtime behaviour changes.
3. **Paths.** Result and data locations now resolve through
   `qgridx.utils.paths` rather than by walking up from `__file__`, so an
   installed wheel and an editable checkout behave identically.
4. **Compression.** The RTS-GMLC series ship gzipped. `pandas` decompresses
   them transparently, and the package is 5.6 MB instead of 19 MB.

## What was deliberately left out

The research tree contains work that does not back any reported result: a
synthetic cut-discovery testbed, a variational SPSA path, a circuit-cloning
experiment, and several superseded calibration sweeps. These are not part of
`qGridX`. Excluding them keeps the package to code that is actually exercised
by the reproduction commands in the README.
