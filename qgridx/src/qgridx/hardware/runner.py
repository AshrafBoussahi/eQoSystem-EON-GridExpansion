"""Submitting circuits to a device and getting counts back.

Thin wrappers over the cloud provider. Two behaviours here exist because of how
the real campaign went.

:func:`_counts` tolerates more than one result shape: ``get_counts()`` raises
outright on some providers, and the raw measurement array is the more reliable
source when it does.

Jobs are addressable after the fact. The polling loop can be interrupted, and
resubmitting a setting that already ran costs real credits for a result already
sitting on the server, so an interrupted campaign is completed by collecting
finished jobs by id rather than by re-executing them.
"""
import numpy as np


def _provider():
    from qbraid.runtime import QbraidProvider
    return QbraidProvider()

def _submit(dev, qasm, shots, tag):
    job = dev.run(qasm, shots=shots)
    print(f"    submitted {tag}: job {job.id}", flush=True)
    return job

def _counts(job, n):
    """Pull counts out of a finished job, tolerating the shapes different
    backends return. `get_counts()` raises outright on some providers, and the
    measurement array is the more reliable source when it does."""
    res = job.result()
    try:
        c = res.data.get_counts()
        if c:
            return {str(k): int(v) for k, v in c.items()}
    except Exception:
        pass
    meas = getattr(res.data, "measurements", None)
    if meas is None:
        raise RuntimeError(f"no counts and no measurements on job {job.id}: "
                           f"{type(res.data)} {getattr(res, 'details', '')}")
    arr = np.asarray(meas)
    if arr.ndim == 3:            # (experiments, shots, qubits)
        arr = arr.reshape(-1, arr.shape[-1])
    out = {}
    for shot in arr:
        key = "".join(str(int(b)) for b in shot[:n])
        out[key] = out.get(key, 0) + 1
    return out
