"""PCE-VQA baseline (torch path == Qiskit path), MaxCut helpers, solvers, pool and model."""

import itertools

import numpy as np
import torch
from qiskit.quantum_info import Statevector

from genpce.baselines import BrickworkMSAnsatz, TorchSimulator, run_pce_vqa
from genpce.model import GPT, GPTConfig, sample_sequences, sequence_log_probs
from genpce.pce import CorrelatorSet, RelaxedLossParams, correlators_from_statevector, relaxed_loss_np, relaxed_loss_torch
from genpce.pool import native_chain_pool
from genpce.problems import MaxCutInstance, exact_milp, one_pass_bit_swap, random_regular, simulated_annealing
from genpce.sim import StatevectorEvaluator


def _brute_force(inst: MaxCutInstance) -> float:
    best = -np.inf
    for bits in itertools.product([-1, 1], repeat=inst.m):
        best = max(best, inst.cut_value(np.array(bits)))
    return best


def test_torch_simulator_matches_qiskit_statevector():
    n = 5
    ansatz = BrickworkMSAnsatz(n, layers=4)
    cset = CorrelatorSet.build(n, 2)
    rng = np.random.default_rng(0)
    params = ansatz.random_params(rng)
    sim = TorchSimulator(ansatz, cset)
    psi_torch = sim.statevector(torch.as_tensor(params)).numpy()
    psi_qiskit = Statevector(ansatz.qiskit_circuit(params)).data
    np.testing.assert_allclose(psi_torch, psi_qiskit, atol=1e-10)
    c_torch = sim.correlators(torch.as_tensor(params)).numpy()
    c_ref = correlators_from_statevector(psi_qiskit, cset)
    np.testing.assert_allclose(c_torch, c_ref, atol=1e-10)
    assert ansatz.num_params == len(params)
    assert ansatz.num_two_qubit_gates == 2 + 2 + 2 + 2


def test_relaxed_loss_torch_matches_numpy_and_is_differentiable():
    inst = random_regular(12, 3, seed=0)
    cset = CorrelatorSet.build(4, 2, m=12)
    params = RelaxedLossParams.sciorilli(inst, 4, 2)
    c = np.random.default_rng(0).uniform(-0.5, 0.5, size=12)
    ln = relaxed_loss_np(c, inst, params)
    ct = torch.tensor(c, requires_grad=True)
    lt = relaxed_loss_torch(ct, inst, params)
    assert abs(float(lt.detach()) - ln) < 1e-10
    lt.backward()
    assert torch.isfinite(ct.grad).all()
    assert params.nu == inst.num_edges / 2 + (inst.m - 1) / 4  # Edwards–Erdős


def test_pce_vqa_runs_and_improves_over_random_signs():
    inst = random_regular(12, 3, seed=1)
    cset = CorrelatorSet.build(4, 2, m=12)
    res = run_pce_vqa(inst, cset, BrickworkMSAnsatz(4, layers=3), optimizer="adam", lr=0.1, max_steps=150, seed=0)
    opt = _brute_force(inst)
    assert res.cut_ls >= res.cut
    assert res.cut_ls / opt > 0.85
    assert res.circuit_executions_equiv == res.steps * (2 * BrickworkMSAnsatz(4, 3).num_params + 1)


def test_bit_swap_and_solvers_on_small_graph():
    inst = random_regular(14, 3, seed=2)
    opt = _brute_force(inst)
    x = np.ones(inst.m, dtype=np.int8)
    assert inst.cut_value(one_pass_bit_swap(inst, x)) >= inst.cut_value(x)
    _, v_sa = simulated_annealing(inst, restarts=3, sweeps=100, seed=0)
    assert abs(v_sa - opt) < 1e-9
    _, v_milp, proven = exact_milp(inst, time_limit=60)
    assert proven and abs(v_milp - opt) < 1e-9


def test_pool_and_model_consistency():
    n = 4
    pool = native_chain_pool(n)
    cset = CorrelatorSet.build(n, 2)
    cfg = GPTConfig(vocab_size=pool.size, max_len=12, d_model=32, n_layers=2, n_heads=2)
    model = GPT(cfg)
    tokens, logps = sample_sequences(model, num=6, length=12, beta=1.3, generator=torch.Generator().manual_seed(0))
    assert tokens.shape == (6, 12) and logps.shape == (6, 12)
    logp2, _ = sequence_log_probs(model, tokens, beta=1.3)
    np.testing.assert_allclose(logp2.detach().numpy(), logps.numpy(), atol=1e-5)
    circuits = pool.to_circuits(tokens.numpy())
    corr = StatevectorEvaluator(cset).evaluate(circuits)
    assert corr.shape == (6, cset.m) and np.all(np.abs(corr) <= 1 + 1e-12)
    counts = pool.gate_counts(tokens[0].numpy())
    template_cz = pool.template_two_qubit_gates(12)
    assert counts["tokens"] == 12 and counts["gates"] + counts["identity"] == 12 + template_cz
    # the built circuit contains exactly the template's CZ gates
    ops = circuits[0].count_ops()
    assert ops.get("cz", 0) == template_cz == counts["two_qubit"]
    # slot semantics: token t acts on qubit t mod n
    free = native_chain_pool(n, template=None, slot=False, entangler_tokens=True)
    assert free.template_two_qubit_gates(12) == 0 and free.size == 24 * n + (n - 1) + 1
