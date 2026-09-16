"""Device campaign: the three-setting claim, executed.

Stages, in order. Each is a separate entry point so the offline ones can be
repeated freely and the billed ones cannot be triggered by accident.

``prepare``   Train the circuits and freeze tokens, string assignment and exact
              correlators to a manifest. Nothing here touches a device.
``validate``  Prove the emitted program reproduces the trainer's own state,
              offline. Run this before spending device time; a device
              disagreement is only informative once this passes.
``probe``     Submit exactly one job and report its real cost, so the campaign
              size is chosen from a measurement rather than an estimate.
``run``       Submit the campaign: three measurement settings per circuit plus
              read-out calibration. Resumable, and it skips anything already
              recorded.
``collect``   File an already-finished job by id. Use this instead of re-running
              when a poll was interrupted; the result is already on the server
              and resubmitting it costs real credits.
``analyze``   Reconstruct every correlator from the three records and score sign
              agreement and magnitude retention against simulation.

What the campaign measured: 45 decisions and then 105 decisions, each read from
exactly three device jobs and 3,072 shots. The job count and shot budget did not
move, which is the encoding's central operational claim. What the device
returned at those circuit depths is reported by ``analyze`` without varnish.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from qgridx.decoder.repair import sign_readout
from qgridx.encoding.families import random_assignment
from qgridx.encoding.loss import correlators
from qgridx.generator.executor import build_unitary_table, execute_batch
from qgridx.generator.train import RegimeAConfig, train_regime_a
from qgridx.generator.vocab import GQEVocab
from qgridx.grid.cases import case118
from qgridx.hardware.analysis import counts_to_correlators, tensored_mitigate
from qgridx.hardware.qasm import calibration_qasm, to_qasm3, tokens_to_qasm
from qgridx.hardware.runner import _counts, _provider, _submit
from qgridx.problems.siting import build_instance
from qgridx.utils.paths import results_dir

OUT_DIR = results_dir() / "doe_phase3"
MANIFEST = OUT_DIR / "g3_manifest.json"
RAW = OUT_DIR / "g3_cepheus_raw.json"

DEVICE = "rigetti:rigetti:qpu:cepheus-1-108q"
SIM_DEVICE = "qbraid:qbraid:sim:qir-sv"
SHOTS = 1024
CEIL_MULT = 3
SCREENED_118 = [106, 75, 107, 118, 42, 74, 76, 105, 41]
CONFIGS = [
    ("siting45", 6, 2, 45, 10_000),      # the benchmark family, paper budget
    ("compress105", 7, 3, 105, 800),     # 105 decisions on 7 qubits, k=3
]


def build_circuit(label, n, k, m, train_evals=10_000):
    """Train a generator on a real instance of the requested size and return the
    circuit it produced plus the exact correlators of the state it prepares."""
    if m == 45:
        inst = build_instance(seed=1351626271, candidate_buses=SCREENED_118[:5], L=9,
                              budget_fraction_range=BUDGET_RANGE,
                              coupling_scale_mult=COUPLING_MULT, case_module=case118,
                              lmp_reference_pool=SCREENED_118, benefit_magnitude=BENEFIT_MAG)
    else:
        # the compression family: 105 decisions, order-3 correlations, 7 qubits
        inst = build_instance(seed=1351626271, candidate_buses=SCREENED_118[:7], L=15,
                              budget_fraction_range=BUDGET_RANGE,
                              coupling_scale_mult=COUPLING_MULT, case_module=case118,
                              lmp_reference_pool=SCREENED_118, benefit_magnitude=BENEFIT_MAG)
    assert inst.m == m, f"{label}: built m={inst.m}, wanted {m}"

    assign = random_assignment(n, k, m, seed=12345)
    vocab = GQEVocab(n=n, max_len=(3 * n + 2) * CEIL_MULT)
    ut = build_unitary_table(vocab)
    cfg = RegimeAConfig(max_evals=train_evals, checkpoints=(train_evals,), seed=101)
    res = train_regime_a(inst, vocab, ut, n, assign, [], cfg)
    with torch.no_grad():
        state = execute_batch(torch.tensor(res.best_tokens[None], dtype=torch.long), ut, n)
        pi = correlators(state, assign, n).numpy()[0]
    n_gates = int((res.best_tokens != vocab.EOS_ID).sum() - 1)
    return dict(label=label, n=n, k=k, m=m, tokens=[int(t) for t in res.best_tokens],
                assignment=[[list(q), a] for (q, a) in assign], pi_exact=pi.tolist(),
                n_gates=n_gates, best_cost=float(res.best_cost), train_evals=train_evals)


def prepare():
    """Written incrementally, one entry per config: a long training run for one
    size should not be able to lose an already-finished one."""
    done = {e["label"]: e for e in json.loads(MANIFEST.read_text())} if MANIFEST.exists() else {}
    for label, n, k, m, evals in CONFIGS:
        if label in done:
            print(f"{label}: already in manifest, skipping")
            continue
        t0 = time.time()
        e = build_circuit(label, n, k, m, evals)
        e["train_s"] = time.time() - t0
        done[label] = e
        MANIFEST.write_text(json.dumps([done[l] for l, *_ in CONFIGS if l in done], indent=1))
        print(f"{label}: n={n} k={k} m={m} gates={e['n_gates']} "
              f"cost={e['best_cost']:.4f} evals={evals} ({e['train_s']:.0f}s)", flush=True)
    print("wrote", MANIFEST)


def validate():
    """Offline: the emitted QASM must reproduce the trainer's own state, and the
    three basis-rotated QASMs must reproduce the exact correlators of their
    family when read with the project's bit convention."""
    entries = json.loads(MANIFEST.read_text())
    ok = True
    for e in entries:
        n, m = e["n"], e["m"]
        vocab = GQEVocab(n=n, max_len=(3 * n + 2) * CEIL_MULT)
        assign = [(tuple(q), a) for q, a in e["assignment"]]
        tokens = np.array(e["tokens"])
        pi_exact = np.array(e["pi_exact"])

        qasm = tokens_to_qasm(tokens, vocab, n, "Z", measure=False)
        st_q = simulate_qasm(qasm, n)
        st_p = execute_batch(torch.tensor(tokens[None], dtype=torch.long),
                             build_unitary_table(vocab), n)
        fid = abs(torch.vdot(st_q[0], st_p[0]).item())
        pi_from_qasm = correlators(st_q, assign, n).numpy()[0]
        dev = float(np.max(np.abs(pi_from_qasm - pi_exact)))

        # each basis rotation must turn its family's correlators into Z-parities
        worst = 0.0
        for axis in ("X", "Y", "Z"):
            qm = tokens_to_qasm(tokens, vocab, n, axis, measure=False)
            probs = np.abs(simulate_qasm(qm, n)[0].numpy()) ** 2
            idx = np.arange(2 ** n)
            for i, (qubits, ax) in enumerate(assign):
                if ax != axis:
                    continue
                # project convention: qubit 0 is the most significant bit
                par = np.zeros(2 ** n, dtype=np.int8)
                for q in qubits:
                    par ^= ((idx >> (n - 1 - q)) & 1).astype(np.int8)
                val = float(np.dot(probs, 1.0 - 2.0 * par))
                worst = max(worst, abs(val - pi_exact[i]))

        good = fid > 1 - 1e-9 and dev < 1e-9 and worst < 1e-9
        ok &= good
        print(f"{e['label']:12s} statevector fidelity {fid:.12f}  "
              f"max |pi_qasm - pi_exact| {dev:.2e}  "
              f"max basis-rotation error {worst:.2e}  {'OK' if good else 'FAIL'}")
    print("\nvalidation", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def probe(device=DEVICE):
    """Submit exactly one job and report the real credit cost, so the campaign
    size can be chosen from a measurement rather than an estimate."""
    p = _provider()
    before = p.client.user_credits_value()
    dev = p.get_device(device)
    entries = json.loads(MANIFEST.read_text())
    e = entries[0]
    vocab = GQEVocab(n=e["n"], max_len=(3 * e["n"] + 2) * CEIL_MULT)
    qasm = to_qasm3(tokens_to_qasm(np.array(e["tokens"]), vocab, e["n"], "Z"), e["n"])
    job = _submit(dev, qasm, SHOTS, f"{e['label']}/Z")
    job.wait_for_final_state(timeout=1800, poll_interval=15)
    print("  status:", job.status(), job.metadata().get("statusMsg", ""))
    counts = _counts(job, e["n"])
    after = p.client.user_credits_value()
    print(f"  status={job.status()}  distinct bitstrings={len(counts)}  "
          f"total shots={sum(counts.values())}")
    print(f"  credits {before:.1f} -> {after:.1f}  (cost {before-after:.1f})")
    out = OUT_DIR / "g3_probe.json"
    out.write_text(json.dumps(dict(device=device, label=e["label"], axis="Z",
                                   job_id=str(job.id), counts=counts,
                                   credit_cost=before - after), indent=1))
    print("wrote", out)


def run(device=DEVICE, labels=None):
    p = _provider()
    before = p.client.user_credits_value()
    dev = p.get_device(device)
    entries = json.loads(MANIFEST.read_text())
    if labels:
        entries = [e for e in entries if e["label"] in labels]

    records = json.loads(RAW.read_text()) if RAW.exists() else {}
    records.setdefault("device", device)
    records.setdefault("shots", SHOTS)
    records.setdefault("circuits", {})
    records.setdefault("calibration", {})

    for e in entries:
        n = e["n"]
        vocab = GQEVocab(n=n, max_len=(3 * n + 2) * CEIL_MULT)
        slot = records["circuits"].setdefault(e["label"], {})
        for axis in ("X", "Y", "Z"):
            if axis in slot:
                print(f"  {e['label']}/{axis} already recorded, skipping")
                continue
            qasm = to_qasm3(tokens_to_qasm(np.array(e["tokens"]), vocab, n, axis), n)
            job = _submit(dev, qasm, SHOTS, f"{e['label']}/{axis}")
            job.wait_for_final_state(timeout=3600, poll_interval=15)
            slot[axis] = dict(job_id=str(job.id), status=str(job.status()),
                              counts=_counts(job, n))
            RAW.write_text(json.dumps(records, indent=1))
            print(f"    {e['label']}/{axis}: {sum(slot[axis]['counts'].values())} shots", flush=True)

        # readout calibration on the same logical register, same compile path
        cal = records["calibration"].setdefault(str(n), {})
        for state_name, excite in (("zeros", False), ("ones", True)):
            if state_name in cal:
                continue
            job = _submit(dev, to_qasm3(calibration_qasm(n, excite), n), SHOTS,
                          f"cal{n}/{state_name}")
            job.wait_for_final_state(timeout=3600, poll_interval=15)
            cal[state_name] = dict(job_id=str(job.id), counts=_counts(job, n))
            RAW.write_text(json.dumps(records, indent=1))
            print(f"    cal{n}/{state_name} done", flush=True)

    after = p.client.user_credits_value()
    records["credits_spent"] = before - after
    RAW.write_text(json.dumps(records, indent=1))
    print(f"\ncredits {before:.1f} -> {after:.1f} (spent {before-after:.1f})")
    print("wrote", RAW)


def collect(label, axis, job_id):
    """Record an ALREADY-SUBMITTED job's counts into the raw file.

    `run` polls with wait_for_final_state, and when that poll is interrupted the
    next `run` resubmits the same setting -- which costs real QPU credits for a
    result already sitting on the server. This action fetches a finished job by
    id and files it, so an interrupted campaign is completed by collection
    rather than by re-execution.
    """
    from qbraid.runtime import QbraidProvider
    p = QbraidProvider()
    entries = {e["label"]: e for e in json.loads(MANIFEST.read_text())}
    n = entries[label]["n"] if label in entries else int(label[-1])
    dev = p.get_device(DEVICE)
    job = dev.get_job(job_id) if hasattr(dev, "get_job") else None
    if job is None:
        from qbraid.runtime.native.job import QbraidJob
        job = QbraidJob(job_id, device=dev, client=p.client)
    st = str(job.status())
    if "COMPLETED" not in st:
        raise SystemExit(f"job {job_id} is {st}, not collectable")
    counts = _counts(job, n)

    rec = json.loads(RAW.read_text()) if RAW.exists() else {
        "device": DEVICE, "shots": SHOTS, "circuits": {}, "calibration": {}}
    if label.startswith("cal"):
        rec["calibration"].setdefault(str(n), {})[axis] = dict(job_id=job_id, counts=counts)
    else:
        rec["circuits"].setdefault(label, {})[axis] = dict(job_id=job_id, status=st,
                                                           counts=counts)
    RAW.write_text(json.dumps(rec, indent=1))
    print(f"collected {label}/{axis}: {sum(counts.values())} shots -> {RAW.name}")


def analyze():
    import pandas as pd
    entries = {e["label"]: e for e in json.loads(MANIFEST.read_text())}
    rec = json.loads(RAW.read_text())
    rows = []
    for label, slot in rec["circuits"].items():
        e = entries[label]
        n, m = e["n"], e["m"]
        assign = [(tuple(q), a) for q, a in e["assignment"]]
        pi_exact = np.array(e["pi_exact"])

        cal = rec["calibration"].get(str(n), {})
        for msb_first in (True, False):
            pi_hw = np.full(m, np.nan)
            for axis in ("X", "Y", "Z"):
                if axis not in slot:
                    continue
                v = counts_to_correlators(slot[axis]["counts"], assign, axis, n, msb_first)
                pi_hw = np.where(np.isnan(v), pi_hw, v)
            got = ~np.isnan(pi_hw)
            if not got.any():
                continue
            sign_agree = float(np.mean(np.sign(pi_hw[got]) == np.sign(pi_exact[got])))
            # Retention as a least-squares slope through the origin, NOT a mean of
            # per-correlator ratios. Many of these correlators are exactly small,
            # and dividing by a denominator near zero produces ratios above 1 that
            # say nothing about how much signal survived -- the first version of
            # this analysis reported a "retention" of 1.6 for that reason. The
            # slope weights each correlator by how much signal it actually carries.
            ex, hw = pi_exact[got], pi_hw[got]
            retention = float((ex @ hw) / (ex @ ex)) if (ex @ ex) > 0 else np.nan
            # A sign is only meaningfully recoverable if the exact correlator stands
            # above the finite-shot resolution 1/sqrt(N); below that the measured
            # sign is noise whatever the device does.
            floor = 3.0 / np.sqrt(rec["shots"])
            big = np.abs(ex) > floor
            sign_big = float(np.mean(np.sign(hw[big]) == np.sign(ex[big]))) if big.any() else np.nan
            corr = float(np.corrcoef(pi_hw[got], pi_exact[got])[0, 1])

            mit_sign, mit_ret = np.nan, np.nan
            if cal:
                p0 = np.zeros(n); p1 = np.zeros(n)
                for st, arr in (("zeros", p1), ("ones", p0)):
                    if st not in cal:
                        continue
                    c = cal[st]["counts"]; tot = sum(c.values())
                    for kk, v in c.items():
                        bs = kk if msb_first else kk[::-1]
                        for q, ch in enumerate(bs[:n]):
                            # in |0...0> a measured 1 is a 0->1 flip; in |1...1> a 0 is 1->0
                            if (st == "zeros" and ch == "1") or (st == "ones" and ch == "0"):
                                arr[q] += v / tot
                shrink = tensored_mitigate(None, p0, p1, n, msb_first)
                pi_mit = pi_hw.copy()
                for i, (qubits, _a) in enumerate(assign):
                    if got[i]:
                        pi_mit[i] = pi_hw[i] / float(np.prod(shrink[list(qubits)]))
                mit_sign = float(np.mean(np.sign(pi_mit[big]) == np.sign(ex[big])))                     if big.any() else np.nan
                hm = pi_mit[got]
                mit_ret = float((ex @ hm) / (ex @ ex)) if (ex @ ex) > 0 else np.nan

            rows.append(dict(label=label, n=n, m=m, k=e["k"], n_gates=e["n_gates"],
                             bit_order="msb_first" if msb_first else "lsb_first",
                             n_correlators=int(got.sum()), jobs=len(slot),
                             shots_total=len(slot) * rec["shots"],
                             sign_agreement=sign_agree,
                             n_above_floor=int(big.sum()),
                             sign_agreement_above_floor=sign_big,
                             magnitude_retention=retention,
                             pearson_r=corr, sign_agreement_mitigated=mit_sign,
                             magnitude_retention_mitigated=mit_ret,
                             readout_err_mean=float(np.mean((p0 + p1) / 2)) if cal else np.nan))
    df = pd.DataFrame(rows)
    out = OUT_DIR / "g3_cepheus.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\nwrote {out}")
    print(f"credits spent: {rec.get('credits_spent')}")
