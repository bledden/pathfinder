"""Foundation circuit builder: BB memory in Z- or X-basis under uniform or SI1000 noise.

Wraps/extends qldpc/bb_circuit.py's validated, deterministic-detector schedule:
  - (basis="Z", noise="uniform") returns the EXACT existing build_z_memory(bb, rounds, p)
    (regression-locked by tests/foundation/test_circuits.py).
  - All other cases use a single parameterized builder that mirrors build_z_memory's
    time-reversed-monomial CNOT layering (which makes detectors deterministic) and swaps in
    the per-op SI1000 rates and the X-basis variant.

SI1000 superconducting noise (Gidney): two-qubit gate p, one-qubit/idle p/10,
measurement classical flip 5p, reset p/10, idle-during-measurement 2p. "uniform" = single p
everywhere (matching build_z_memory's depolarizing model so (Z,uniform) is bit-identical).

X-basis mirror: H_X = [A | B] checks measured by X-ancillas (which detect Z errors). Data is
init/read in the X-basis (RX / MX); ancillas are X-basis (RX / MRX); the X-check CNOTs run
ancilla->data. Pre-measure / init flips are Z_ERROR (a Z error flips an X-basis measurement).
Observable = Z-logicals (commute with H_X, flipped by Z errors -> deterministic in X-readout).
Single-basis memory only measures one check type, so all per-round measurements mutually commute
and detectors are deterministic for the mirrored schedule (verified: noiseless circuit has a DEM
with zero error terms, i.e. all detectors deterministic).
"""
import os
import sys

import numpy as np
import stim

# conftest.py puts repo root + qldpc/ on sys.path; ensure qldpc/ is importable when run directly.
_QLDPC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _QLDPC not in sys.path:
    sys.path.insert(0, _QLDPC)

from bb_circuit import X_ORDER, Z_ORDER, monomial_perms, build_z_memory


class _Noise:
    """Per-operation error rates. uniform -> single p; si1000 -> Gidney superconducting rates."""

    def __init__(self, p, model):
        self.p = p
        self.model = model
        if model == "uniform":
            self.two_qubit = p
            self.one_qubit = p
            self.idle = p
            self.meas_flip = p
            self.reset = p
            self.idle_meas = p
        elif model == "si1000":
            self.two_qubit = p
            self.one_qubit = p / 10.0
            self.idle = p / 10.0
            self.meas_flip = 5.0 * p
            self.reset = p / 10.0
            self.idle_meas = 2.0 * p
        else:
            raise ValueError(f"unknown noise model {model!r}")


def _build_general(bb, rounds, p, basis, noise):
    """Parameterized single-basis BB memory mirroring build_z_memory's deterministic schedule."""
    N = bb.N
    Ld = lambda i: i
    Rd = lambda i: N + i
    Anc = lambda c: 2 * N + c
    n_data = 2 * N
    P = monomial_perms(bb)
    nz = _Noise(p, noise)

    data = list(range(n_data))
    anc = [Anc(i) for i in range(N)]

    if basis == "Z":
        order = Z_ORDER
        # H_Z = [B^T | A^T]: B-term monomial -> left block, A-term -> right block.
        left_is_A = False
        reset_op, meas_op = "R", "MR"
        data_init_err = "X_ERROR"     # X errors are detected by Z-checks
        anc_flip_err = "X_ERROR"
        data_read = "M"
        # final Z-check cc: left support = B[:,cc], right support = A[:,cc]
        chkL = [np.where(bb.B[:, cc] == 1)[0] for cc in range(N)]
        chkR = [np.where(bb.A[:, cc] == 1)[0] for cc in range(N)]
        logicals = bb.logicals_X()
    elif basis == "X":
        order = X_ORDER
        # H_X = [A | B]: A-term monomial -> left block, B-term -> right block.
        left_is_A = True
        reset_op, meas_op = "RX", "MRX"
        data_init_err = "Z_ERROR"     # Z errors are detected by X-checks
        anc_flip_err = "Z_ERROR"
        data_read = "MX"
        # final X-check cc: left support = A[:,cc], right support = B[:,cc]
        chkL = [np.where(bb.A[:, cc] == 1)[0] for cc in range(N)]
        chkR = [np.where(bb.B[:, cc] == 1)[0] for cc in range(N)]
        logicals = bb.logicals_Z()
    else:
        raise ValueError(f"unknown basis {basis!r}")

    c = stim.Circuit()

    def cnot_pairs(key):
        """For check schedule entry `key`, return the flat CX target list for all N checks."""
        mat = P[key]
        is_A = (key[0] == "A")
        # In Z-basis: A-term -> right block, B-term -> left block.
        # In X-basis: A-term -> left block, B-term -> right block.
        if basis == "Z":
            use_right = is_A
        else:
            use_right = not is_A
        pairs = []
        for cc in range(N):
            iT = int(np.argmax(mat[:, cc]))
            dq = Rd(iT) if use_right else Ld(iT)
            if basis == "Z":
                pairs += [dq, Anc(cc)]      # CX data -> Z-anc (data control, anc target)
            else:
                pairs += [Anc(cc), dq]      # CX X-anc -> data (anc control, data target)
        return pairs

    def noisy_round(circ):
        circ.append(reset_op, anc)
        if p > 0 and nz.reset > 0:
            circ.append(anc_flip_err, anc, nz.reset)
        for step in range(6):
            pairs = cnot_pairs(order[step])
            circ.append("CX", pairs)
            if p > 0 and nz.two_qubit > 0:
                circ.append("DEPOLARIZE2", pairs, nz.two_qubit)
        if p > 0 and nz.meas_flip > 0:
            circ.append(anc_flip_err, anc, nz.meas_flip)   # classical measurement flip
        circ.append(meas_op, anc)

    c.append(reset_op, data)
    if p > 0 and nz.reset > 0:
        c.append(data_init_err, data, nz.reset)
    for r in range(rounds):
        noisy_round(c)
        if r == 0:
            for i in range(N):
                c.append("DETECTOR", [stim.target_rec(-N + i)])
        else:
            for i in range(N):
                c.append("DETECTOR", [stim.target_rec(-N + i), stim.target_rec(-2 * N + i)])
    if p > 0 and nz.meas_flip > 0:
        c.append(data_init_err, data, nz.meas_flip)        # final data readout flip
    c.append(data_read, data)
    # final check-consistency detectors: check cc parity from final data XOR last anc cc
    for cc in range(N):
        recs = []
        for d in chkL[cc]:
            recs.append(stim.target_rec(-n_data + Ld(int(d))))
        for d in chkR[cc]:
            recs.append(stim.target_rec(-n_data + Rd(int(d))))
        recs.append(stim.target_rec(-n_data - N + cc))
        c.append("DETECTOR", recs)
    for li in range(logicals.shape[0]):
        obs = [stim.target_rec(-n_data + j) for j in range(n_data) if logicals[li, j]]
        c.append("OBSERVABLE_INCLUDE", obs, li)
    return c


def build_memory(bb, rounds, p, basis="Z", noise="si1000"):
    """Build a BB memory experiment circuit.

    Args:
        bb: a BBCode instance (e.g. the [[72,12,6]] default).
        rounds: number of syndrome-extraction rounds.
        p: base physical error rate.
        basis: "Z" (detect X errors via H_Z, observable = X-logicals) or
               "X" (detect Z errors via H_X, observable = Z-logicals).
        noise: "uniform" (single rate p everywhere) or "si1000" (Gidney superconducting rates).

    Returns: stim.Circuit with rounds*N + N detectors (N = bb.N) and k observables.
    """
    if basis == "Z" and noise == "uniform":
        # Regression-locked: must be bit-identical to the existing validated builder.
        return build_z_memory(bb, rounds, p)
    if basis not in ("Z", "X"):
        raise ValueError(f"unknown basis {basis!r}")
    if noise not in ("uniform", "si1000"):
        raise ValueError(f"unknown noise model {noise!r}")
    return _build_general(bb, rounds, p, basis, noise)
