"""Project 2: Sciorilli's exact ansatz construction, ported from their
src/circuit.py (Project 2/MarcoSciorilli/.../src/circuit.py, read in full)
into this project's own simulator primitives (qgridx/encoding/simulator.py's
apply_1q/apply_2q, RXX/RYY/RZZ), rather than depending on qibo -- the
connectivity/rotation-cycling LOGIC is theirs, ported line-for-line where
it matters; gate application uses this project's own already-verified
executor instead of theirs.

Only the ansatz configuration they demonstrate concretely in their own
repo is used: connectivity='brickwork_single_rotating', rotation='Taylor_
efficient', entanglement='Taylor_efficient' (example.ipynb, the ONE
complete worked example in their repository -- G11, n=800, k=6, qubits=11,
layer_number=157). None of Table 1's four headline instances (pm3-8-50,
G14, G23, G60) match that example's scale, so depth `p` is sweep-selected
per instance to match Supplementary Table 1's reported 2-qubit gate counts
as closely as possible (the protocol's own documented fallback for an
"ansatz not fully specified for this exact instance" situation) -- using
THEIR real connectivity/cycling algorithm, not a substitute hardware-
efficient template.

Ported connectivity function (circuit.py lines 107-122, 'brickwork_single_
rotating' branch only -- the other ~10 branches are not used and not
ported): a rotating-partner pairing where layer l pairs qubit q with
q+l-1 (mod n), refit to stay in range, giving every qubit a new partner
each entangling layer without repeats until the pattern cycles.

Ported rotation/entanglement cycling ('Taylor_efficient', circuit.py lines
330-338 / 354-361): a single shared `counter_taylor` (starts at 0,
increments once per layer -- rotation AND entangling layers both consume
it) selects the axis via `counter_taylor % 3`: 0->Z, 1->X, 2->Y (read
directly off their if/elif chain -- NOT the more "natural" 0->X,1->Y,2->Z
order an English paraphrase would suggest; got this wrong on a first pass
before re-reading the source, see below). So across l=0,1,2,3,4,5,... the
axis sequence is Z,X,Y,Z,X,Y,... regardless of rotation/entangling parity;
since rotation layers are the even l's and entangling layers are the odd
l's, the two interleave to give rotation-layers axis order Z,Y,X,Z,Y,X,...
and entangling-layers order X,Z,Y,X,Z,Y,... (both stepping backwards
through the cycle, since each skips one tick of the shared counter).

`_brickwork_single_rotating`'s wraparound pairing (below) can produce a
pair (q1,q2) with q1>q2 (e.g. n=5,layer=3 gives pair (4,0)); their
_qibo_circuit_ passes such pairs straight to qibo's gates.RXX/RYY/RZZ,
which are qubit-order-symmetric since SWAP.(P⊗P).SWAP = P⊗P for any single
Pauli P repeated on both factors -- so this project's own apply_2q (which
asserts q1<q2) can be called with the pair sorted, safely, verified
against this symmetry rather than assumed.

The reference's `refit` (circuit.py lines 111-118) recurses until every
element is < n, which risks Python's recursion limit at large depth.
Provably equivalent to `q % n` for every case that is ever actually used:
the only values that can be negative are produced at layer=0 (giving
`range(-1, n-1)`), and layer=0 only occurs at l=0, whose entang_list is
never consulted (`_qibo_circuit_`'s rotation-layer condition is
`q in [...] or l == 0`, so l==0 short-circuits to "rotate everything"
regardless of entang_list content) -- so the negative-value case is dead
code in the original too, and `% n` matches `refit` exactly everywhere
the result is actually read.
"""
import math

import numpy as np
import torch

from qgridx.encoding.simulator import CDTYPE, _rot_matrix, _rxx_yy_zz_matrix, apply_1q, apply_2q, zero_state

_TAYLOR_AXIS = ("Z", "X", "Y")  # counter_taylor % 3 == 0,1,2 -> Z,X,Y (circuit.py lines 332-337)


def _brickwork_single_rotating(layer: int, n: int) -> list:
    """Ported from Circuit._define_connectivity_, connectivity ==
    'brickwork_single_rotating' branch (circuit.py lines 107-122); `refit`
    replaced by the provably-equivalent `% n` (see module docstring)."""
    entang_list_unpaired = [q % n for q in range(layer - 1, n + layer - 1)]
    entang_list = []
    for q in range(1, len(entang_list_unpaired), 2):
        entang_list.append((entang_list_unpaired[q - 1], entang_list_unpaired[q]))
    return entang_list


def build_sciorilli_gate_sequence(n: int, depth: int) -> list:
    """Returns a flat list of gate-spec dicts:
    {'kind': '1q'|'2q', 'axis': 'X'|'Y'|'Z', 'qubits': (q,) or (q1,q2)}
    (angles are trainable parameters, filled in separately -- this function
    only fixes the STRUCTURE, matching Circuit._qibo_circuit_'s l=0..depth-1
    loop: even l = rotation layer, odd l = entangling layer, connectivity/
    rotation-axis/entangling-type exactly as Sciorilli's 'Taylor_efficient'
    + 'brickwork_single_rotating' combination)."""
    gates = []
    counter_taylor = 0
    for l in range(depth):
        entang_list = _brickwork_single_rotating(math.ceil(l / 2), n)
        axis = _TAYLOR_AXIS[counter_taylor % 3]
        counter_taylor += 1
        if l % 2 == 0:
            active_qubits = set(range(n)) if l == 0 else {x for t in entang_list for x in t}
            for q in range(n):
                if q in active_qubits:
                    gates.append(dict(kind="1q", axis=axis, qubits=(q,)))
        else:
            for (q1, q2) in entang_list:
                gates.append(dict(kind="2q", axis=axis, qubits=(q1, q2)))
    return gates


def n_two_qubit_gates(n: int, depth: int) -> int:
    return sum(1 for g in build_sciorilli_gate_sequence(n, depth) if g["kind"] == "2q")


def n_parameters(n: int, depth: int) -> int:
    return len(build_sciorilli_gate_sequence(n, depth))


def run_circuit_pytorch(params: torch.Tensor, gate_seq: list, n: int) -> torch.Tensor:
    """params: (batch, n_params) real. Returns (batch, 2^n) statevector.
    Uses this project's own apply_1q/apply_2q (already bit-exact-verified
    against the dense-table executor elsewhere in Project 2)."""
    batch = params.shape[0]
    state = zero_state(batch, n)
    for gi, g in enumerate(gate_seq):
        theta = params[:, gi]
        if g["kind"] == "1q":
            gate = _rot_matrix(g["axis"], theta)
            state = apply_1q(state, gate, g["qubits"][0], n)
        else:
            q1, q2 = g["qubits"]
            if q1 > q2:
                q1, q2 = q2, q1  # RXX/RYY/RZZ are swap-symmetric (see module docstring)
            gate = _rxx_yy_zz_matrix(g["axis"], theta)
            state = apply_2q(state, gate, q1, q2, n)
    return state
