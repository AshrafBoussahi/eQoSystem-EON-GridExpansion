"""Fast Walsh–Hadamard transform (WHT).

The WHT is the workhorse of Pauli-correlation readout: if ``p`` is the probability vector of
computational-basis outcomes (length ``2**n``), then ``fwht(p)[mask]`` equals the expectation
value of the Z-parity on the qubit subset encoded by ``mask``, i.e. ``<Z_S>`` for
``S = {q : bit q of mask is 1}``. Applying the same transform to X- and Y-basis outcome
probabilities gives ``<X_S>`` and ``<Y_S>``. One transform therefore yields *all* ``2**n`` parity
correlators of a basis at once, in ``O(n 2**n)`` time.

Bit convention: index bit ``q`` (value ``1 << q``) corresponds to qubit ``q``. This matches the
little-endian convention used by Qiskit's ``Statevector`` and by ``int(bitstring, 2)`` applied to
Qiskit count keys.
"""

from __future__ import annotations

import numpy as np

__all__ = ["fwht"]


def fwht(a: np.ndarray, axis: int = -1) -> np.ndarray:
    """Unnormalised fast Walsh–Hadamard transform along ``axis``.

    Args:
        a: Real or complex array whose length along ``axis`` is a power of two.
        axis: Axis along which to transform (default: last).

    Returns:
        A new array of the same shape, ``H^{⊗n} a`` up to the missing ``2**(-n/2)`` factor
        (i.e. the transform is its own inverse up to a factor ``2**n``).
    """
    a = np.array(np.moveaxis(a, axis, -1), copy=True)
    length = a.shape[-1]
    n = length.bit_length() - 1
    if length != (1 << n):
        raise ValueError(f"WHT length must be a power of two, got {length}")
    lead = a.shape[:-1]
    h = 1
    while h < length:
        a = a.reshape(lead + (length // (2 * h), 2, h))
        x = a[..., 0, :].copy()
        y = a[..., 1, :]
        a[..., 0, :] = x + y
        a[..., 1, :] = x - y
        a = a.reshape(lead + (length,))
        h *= 2
    return np.moveaxis(a, -1, axis)
