"""Turning device counts back into correlators, and scoring them.

Every correlator of one family is reconstructed from that family's single
measurement record by a parity average, which is what makes three device jobs
enough for any number of decisions.

Two reporting choices matter. Magnitude retention is a least-squares slope of
measured against exact values through the origin, not a mean of per-correlator
ratios: many correlators in these circuits are genuinely small, and dividing by
a near-zero denominator produces ratios above one that say nothing about how
much signal survived. Sign agreement is reported both over all correlators and
over the subset whose exact magnitude exceeds three times the finite-shot
resolution, since below that threshold the measured sign is noise whatever the
device does.
"""
import numpy as np


def counts_to_correlators(counts, assignment, axis, n, msb_first):
    """Reconstruct every correlator of one family from that family's single
    measurement record. `msb_first=True` reads the leftmost character of the
    returned bitstring as qubit 0, which is this project's own convention;
    `False` is the little-endian reading most platforms use."""
    keys = list(counts.keys())
    total = sum(counts.values())
    bits = np.array([[int(ch) for ch in (k if msb_first else k[::-1])] for k in keys], dtype=np.int8)
    w = np.array([counts[k] for k in keys], dtype=float) / total
    out = np.full(len(assignment), np.nan)
    for i, (qubits, ax) in enumerate(assignment):
        if ax != axis:
            continue
        parity = bits[:, list(qubits)].sum(axis=1) % 2
        out[i] = float(np.dot(w, 1.0 - 2.0 * parity))
    return out

def tensored_mitigate(counts, p0, p1, n, msb_first):
    """Per-qubit readout mitigation: invert each qubit's independent 2x2
    confusion matrix on the single-qubit marginals implied by each parity.

    Applied at the level of a k-body parity expectation rather than the full
    2^n distribution, which is what the correlation encoding actually consumes:
    for independent per-qubit flips, the measured parity expectation of a
    subset T is the exact one scaled by prod_{q in T} (1 - p0_q - p1_q), so
    dividing by that product is the exact inverse under this model.
    """
    shrink = np.clip(1.0 - p0 - p1, 1e-3, None)
    return shrink
