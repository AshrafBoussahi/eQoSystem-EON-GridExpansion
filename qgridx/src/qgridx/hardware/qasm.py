"""Emitting device-ready circuits, and the two traps on the way there.

A generated token sequence becomes an OpenQASM program here. Two findings are
baked into this module because both cost real debugging time against the live
device.

**Submit OpenQASM 3, not OpenQASM 2.** The QASM 2 ingestion path on the target
platform fails to compile these circuits at all: its server-side translation to
Quil schedules each measurement immediately after that qubit's last gate rather
than at the end of the program, and ``quilc`` rejects the result with
"Misplaced or illegal instruction in ProtoQuil program". Emitting per-qubit
measurements, inserting a barrier, and appending a trailing entangling ladder
all failed. The identical circuit as QASM 3 compiles and runs. Rejected
compilations are not billed, so this iteration costs no device budget.

**Keep the Y basis rotation to one gate.** The textbook pair ``sdg`` then ``h``
is correct but on a 7-qubit circuit it pushed the vendor compiler past its
30-second limit. ``rx(pi/2)`` is a single gate and gives ``+Y`` exactly, which
:func:`qgridx.hardware.verify.validate_emission` checks numerically.

Bit ordering is never assumed. This package indexes qubit 0 as the most
significant bit of the basis index, the opposite of most platforms, so the
convention is determined empirically against a noiseless simulator before any
device time is spent.
"""
import numpy as np
import torch

from qgridx.encoding.simulator import (
    CDTYPE, _rot_matrix, apply_1q, apply_2q, zero_state,
)


def _fmt(a):
    return f"{a:.12g}"

def rzz_qasm(a, b, theta):
    return [f"cx q[{a}],q[{b}];", f"rz({_fmt(theta)}) q[{b}];", f"cx q[{a}],q[{b}];"]

def two_qubit_qasm(axis, a, b, theta):
    """exp(-i theta/2 P_a P_b) for P in {X,Y,Z}, matching
    simulator._rxx_yy_zz_matrix exactly.

    XX and YY are obtained by conjugating ZZ with a single-qubit basis change on
    both wires. The sign of that basis change cannot matter here: it appears
    once on each of the two wires, so any sign it introduces enters the product
    P_a P_b squared.
    """
    if axis == "Z":
        return rzz_qasm(a, b, theta)
    if axis == "X":
        pre = [f"h q[{a}];", f"h q[{b}];"]
        post = pre
    elif axis == "Y":
        pre = [f"rx({_fmt(np.pi/2)}) q[{a}];", f"rx({_fmt(np.pi/2)}) q[{b}];"]
        post = [f"rx({_fmt(-np.pi/2)}) q[{a}];", f"rx({_fmt(-np.pi/2)}) q[{b}];"]
    else:
        raise ValueError(axis)
    return pre + rzz_qasm(a, b, theta) + post

def basis_change_qasm(axis, n):
    """Rotate the measurement axis onto Z, so a computational-basis measurement
    returns the requested family. The measured observable is V^dag Z V:
      X: H^dag Z H = X.
      Y: RX(pi/2)^dag Z RX(pi/2) = Y, verified by direct 2x2 multiplication and
         again numerically in validate().

    Y uses the single gate RX(pi/2) rather than the textbook pair S^dag then H.
    Both are correct, but on the 7-qubit circuit the two-gate form pushed quilc
    past its 30-second compile limit and the job failed to compile; halving the
    basis-change gate count fixes it. There is no sign to correct: the
    conjugation above yields +Y, not -Y.
    """
    if axis == "Z":
        return []
    if axis == "X":
        return [f"h q[{q}];" for q in range(n)]
    if axis == "Y":
        return [f"rx({_fmt(np.pi/2)}) q[{q}];" for q in range(n)]
    raise ValueError(axis)

def to_qasm3(qasm2, n):
    """Re-express an emitted OPENQASM 2 program as OPENQASM 3.

    This is not cosmetic. The QASM2 route into this device fails to compile:
    its server-side translation to Quil schedules each measurement immediately
    after that qubit's last gate instead of at the end of the program, and
    quilc rejects the result ("Misplaced or illegal instruction in ProtoQuil
    program"). Neither reordering the measurements, nor a `barrier`, nor a
    trailing entangling ladder fixed it. Submitting the identical circuit as
    QASM3 compiles and runs, so QASM3 is the path used for every device job.
    """
    out = ["OPENQASM 3.0;", 'include "stdgates.inc";',
           f"qubit[{n}] q;", f"bit[{n}] c;"]
    for line in qasm2.splitlines():
        s = line.strip()
        if s.startswith(("OPENQASM", "include", "qreg", "creg")) or not s:
            continue
        if s.startswith("measure"):
            q = s.split("q[")[1].split("]")[0]
            cbit = s.split("c[")[1].split("]")[0]
            out.append(f"c[{cbit}] = measure q[{q}];")
        else:
            out.append(s)
    return "\n".join(out) + "\n"

def tokens_to_qasm(tokens, vocab, n, measure_axis, measure=True, sync=False):
    body = []
    for tid in tokens[1:]:
        tok = vocab.tokens[int(tid)]
        if tok.kind == "special":
            if tok.name == "EOS":
                break
            continue
        if tok.kind == "1q":
            body.append(f"r{tok.axis.lower()}({_fmt(tok.angle)}) q[{tok.qubits[0]}];")
        else:
            body += two_qubit_qasm(tok.axis, tok.qubits[0], tok.qubits[1], tok.angle)
    head = ["OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{n}];", f"creg c[{n}];"]
    tail = basis_change_qasm(measure_axis, n)
    if measure:
        # Explicit per-qubit measurements, not the register-wide `measure q -> c;`.
        # The register form round-tripped through qBraid's QASM->Quil conversion as
        # a MEASURE emitted before the remaining gates, which quilc rejects
        # outright ("Misplaced or illegal instruction in ProtoQuil program"), so the
        # first Cepheus submission failed to compile. Listing each measurement
        # separately at the end of the program keeps them where ProtoQuil requires.
        # Synchronisation ladder, then the measurements.
        #
        # Diagnosed rather than guessed. The measurements are already the final n
        # lines of the emitted program (checked directly), yet quilc rejected the
        # job: "Misplaced or illegal instruction in ProtoQuil program: MEASURE 2
        # ... >>> CNOT 0 5". The server-side QASM-to-Quil translation schedules
        # each measurement immediately after that qubit's LAST gate rather than at
        # the end of the program, and in these circuits the qubits stop being used
        # at different points (qubit 2 last appears 45 gate-lines before the end),
        # so a measurement lands mid-program and ProtoQuil forbids it. A `barrier`
        # does not survive the translation and does not help.
        #
        # The ladder below closes that gap. Each rung is exp(-i theta/2 Z Z) on a
        # neighbouring pair, which is DIAGONAL in the computational basis, so it
        # cannot change any measurement outcome in the basis we are about to
        # measure -- verified numerically in validate(). What it does change is the
        # dependency structure: every qubit is now entangled into a chain by
        # two-qubit gates placed after everything else, so no measurement can be
        # scheduled above them.
        if sync:
            for a in range(n - 1):
                tail = tail + rzz_qasm(a, a + 1, np.pi / 4)
        tail = tail + [f"measure q[{q}] -> c[{q}];" for q in range(n)]
    return "\n".join(head + body + tail) + "\n"

def calibration_qasm(n, excite):
    head = ["OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{n}];", f"creg c[{n}];"]
    body = [f"x q[{q}];" for q in range(n)] if excite else []
    return "\n".join(head + body + [f"measure q[{q}] -> c[{q}];"
                                    for q in range(n)]) + "\n"

def simulate_qasm(qasm, n):
    """Minimal statevector evaluation of the QASM subset this file emits, using
    the project's own apply_1q/apply_2q so the qubit-index convention is the
    project's by construction. Exists to prove the emitted QASM is the circuit
    the trainer actually produced, before any of it reaches a device."""
    state = zero_state(1, n)
    H = torch.tensor([[1, 1], [1, -1]], dtype=CDTYPE) / np.sqrt(2)
    SDG = torch.tensor([[1, 0], [0, -1j]], dtype=CDTYPE)
    X = torch.tensor([[0, 1], [1, 0]], dtype=CDTYPE)
    CX = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=CDTYPE)

    for line in qasm.splitlines():
        line = line.strip()
        if (not line) or line.startswith(("OPENQASM", "include", "qreg", "creg", "measure", "//")):
            continue
        name = line.split("(")[0].split()[0]
        args = [int(t.split("[")[1].split("]")[0]) for t in line.split()[-1].split(",")]
        theta = float(line[line.index("(") + 1:line.index(")")]) if "(" in line else None
        if name in ("rx", "ry", "rz"):
            g = _rot_matrix(name[1].upper(), torch.tensor([theta], dtype=torch.float64))[0]
            state = apply_1q(state, g, args[0], n)
        elif name == "h":
            state = apply_1q(state, H, args[0], n)
        elif name == "sdg":
            state = apply_1q(state, SDG, args[0], n)
        elif name == "x":
            state = apply_1q(state, X, args[0], n)
        elif name == "cx":
            a, b = args
            g = CX if a < b else CX.reshape(2, 2, 2, 2).permute(1, 0, 3, 2).reshape(4, 4)
            state = apply_2q(state, g, min(a, b), max(a, b), n)
        else:
            raise ValueError(f"unhandled QASM line: {line}")
    return state
