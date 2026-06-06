"""Exact coset maximum-likelihood (coset-ML) decoder via tensor-network contraction.

For a CSS code given by a check matrix ``H`` (rows = checks, cols = error
mechanisms), logical rows ``L`` and per-mechanism priors, the coset-ML class
probability for a syndrome ``s`` is

    P(class = a) = sum_{e : H e = s (mod 2), L e = a (mod 2)}
                       prod_m  priors[m]^e[m] (1-priors[m])^(1-e[m])

This module computes that probability vector by contracting a tensor network:

  * one COPY/delta tensor per mechanism, whose diagonal carries that
    mechanism's prior (value 0 -> 1-p, value 1 -> p) so the prior is counted
    exactly once and the mechanism's bit value is shared across every check /
    logical it participates in;
  * one parity tensor per check, pinned to that shot's syndrome bit (1 where
    the XOR of the incident mechanism bits equals the syndrome bit);
  * one logical-parity tensor per logical row with an extra OPEN leg carrying
    the logical-class bit.

Contracting with the logical legs left open yields the 2^n_log coset
probability vector; ``argmax`` is the coset-ML correction class.

Tractable only at small scale (contraction width must be small); the caller is
responsible for gating that. Weight-0 / weight-1 checks and logicals are
handled explicitly so rank-1 / rank-2 tensors do not crash the build.
"""

import itertools

import numpy as np
import quimb.tensor as qtn


def _parity_tensor(k, target):
    """Rank-``k`` tensor that is 1 where the XOR of its k bits equals ``target``.

    For ``k == 0`` this degenerates to a scalar: 1 if ``target == 0`` else 0
    (the empty XOR is 0).
    """
    t = np.zeros((2,) * k)
    if k == 0:
        t = np.array(1.0 if target == 0 else 0.0)
        return t
    for b in itertools.product((0, 1), repeat=k):
        if sum(b) % 2 == target:
            t[b] = 1.0
    return t


def _logical_tensor(k):
    """Parity tensor over ``k`` mechanism legs plus one OPEN leg = the parity.

    The open leg is the last index. For ``k == 0`` this is a length-2 vector
    selecting class 0 (the empty XOR is 0).
    """
    t = np.zeros((2,) * (k + 1))
    for b in itertools.product((0, 1), repeat=k):
        t[b + (sum(b) % 2,)] = 1.0
    return t


def _coset_prob(H, L, priors, syndrome):
    """Return the length-2^n_log coset-probability vector for one syndrome.

    Index ``a`` (0..2^n_log-1) carries the logical-class bits little-endian:
    bit ``o`` of ``a`` (i.e. ``(a >> o) & 1``) is the parity of logical row
    ``o`` (the open leg ``O{o}``). This matches ``coset_ml_ler``'s decode
    ``pred[o] = (best >> o) & 1``.
    """
    H = np.asarray(H) % 2
    L = np.asarray(L) % 2
    n_mech = H.shape[1]
    n_log = L.shape[0]
    priors = np.asarray(priors, dtype=float)
    syndrome = np.asarray(syndrome) % 2

    # For each mechanism, list the factors (checks / logicals) it feeds into.
    use = {m: [] for m in range(n_mech)}
    for c in range(H.shape[0]):
        for m in np.nonzero(H[c])[0]:
            use[int(m)].append(("c", c))
    for o in range(n_log):
        for m in np.nonzero(L[o])[0]:
            use[int(m)].append(("L", o))

    tn = qtn.TensorNetwork([])

    # COPY/delta tensors carrying each mechanism's prior exactly once.
    for m in range(n_mech):
        fan = use[m]
        if not fan:
            # Mechanism feeds no factor: summing it out gives (1-p)+p = 1,
            # so it contributes nothing. Drop it.
            continue
        legs = tuple(f"e{m}_{kind}{idx}" for (kind, idx) in fan)
        k = len(legs)
        copy = np.zeros((2,) * k)
        copy[(0,) * k] = 1.0 - priors[m]
        copy[(1,) * k] = priors[m]
        tn |= qtn.Tensor(copy, inds=legs)

    # Parity tensor per check, pinned to the syndrome bit.
    for c in range(H.shape[0]):
        mechs = np.nonzero(H[c])[0]
        legs = tuple(f"e{int(m)}_c{c}" for m in mechs)
        tn |= qtn.Tensor(_parity_tensor(len(mechs), int(syndrome[c])), inds=legs)

    # Logical-parity tensor per logical row with an OPEN output leg.
    for o in range(n_log):
        mechs = np.nonzero(L[o])[0]
        legs = tuple(f"e{int(m)}_L{o}" for m in mechs) + (f"O{o}",)
        tn |= qtn.Tensor(_logical_tensor(len(mechs)), inds=legs)

    out_inds = tuple(f"O{o}" for o in range(n_log))
    # Order the contracted output so that O0 is the LOWEST-order axis under a
    # C-order ``reshape(-1)`` (last axis varies fastest). That makes the flat
    # index ``a = sum_o bit(O_o) * 2^o`` (O0 = bit 0, little-endian), matching
    # ``coset_ml_ler``'s decode ``(best >> o) & 1``. We achieve this by laying
    # out the axes in REVERSED output order (O_{n-1}, ..., O1, O0) then C-order
    # reshape.
    rev_inds = out_inds[::-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        arr = tn.contract(output_inds=out_inds, optimize="auto")
    if hasattr(arr, "data"):
        # quimb may reorder indices; transpose to the reversed output order.
        arr = arr.transpose(*rev_inds)
        data = np.asarray(arr.data)
    else:
        # scalar (n_log == 0): nothing to reorder.
        data = np.asarray(arr)
    return data.reshape(-1)


def coset_ml_ler(H, L, priors, syndromes, observables):
    """Logical error rate of the coset-ML (tensor-network) decoder.

    For each shot: contract the TN to get the coset-probability vector,
    ``argmax`` -> predicted logical-class bits, compare to ``observables``.
    Returns the fraction of shots whose predicted class != observed class.
    """
    L = np.asarray(L) % 2
    n_log = L.shape[0]
    syndromes = np.asarray(syndromes)
    observables = np.asarray(observables) % 2
    if observables.ndim == 1:
        observables = observables.reshape(-1, 1)

    n_shots = len(syndromes)
    if n_shots == 0:
        return 0.0

    fails = 0
    for i in range(n_shots):
        probs = _coset_prob(H, L, priors, syndromes[i])
        best = int(np.argmax(probs))
        pred = np.array([(best >> b) & 1 for b in range(n_log)], dtype=np.int64)
        if not np.array_equal(pred, observables[i]):
            fails += 1
    return fails / n_shots


def _brute_coset_ml_single(H, L, priors, syndrome):
    """Reference brute-force coset-ML class (n_log == 1) for one syndrome.

    Enumerate all 2^n_mech error vectors; for those consistent with the
    syndrome (H e == s mod 2) accumulate prod(priors^e (1-priors)^(1-e)) into
    bucket (L e mod 2); return the argmax bucket (0/1).
    """
    H = np.asarray(H) % 2
    L = np.asarray(L) % 2
    priors = np.asarray(priors, dtype=float)
    syndrome = np.asarray(syndrome) % 2
    n = H.shape[1]
    w = np.zeros(2)
    for b in itertools.product((0, 1), repeat=n):
        e = np.array(b)
        if not np.array_equal((H @ e) % 2, syndrome):
            continue
        pr = np.prod([priors[j] if e[j] else 1.0 - priors[j] for j in range(n)])
        bucket = int(((L @ e) % 2).reshape(-1)[0])
        w[bucket] += pr
    return int(np.argmax(w))
