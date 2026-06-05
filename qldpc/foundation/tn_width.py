"""Estimate the exact coset-ML TN contraction width for a CSS check matrix + logicals.
Builds the parity-decomposed network (COPY tensors per mechanism, bond-2 XOR chains per check
and per logical). Returns log2 of the largest intermediate (cotengra). >~32 => intractable."""
from collections import defaultdict
import numpy as np
import cotengra as ctg


def _add_chain(inputs, copy_legs, cid, mechs, open_index=None):
    vlegs = []
    for k, m in enumerate(mechs):
        ix = f"v_{cid}_{k}"
        copy_legs[m].append(ix)
        vlegs.append(ix)
    w = len(mechs)
    for k in range(w):
        legs = ([f"{cid}_b{k}"] if k > 0 else []) + [vlegs[k]]
        if k < w - 1:
            legs.append(f"{cid}_b{k+1}")
        elif open_index is not None:
            legs.append(open_index)
        inputs.append(tuple(legs))


def build_indices(H, L):
    """H: (n_check, n_mech) {0,1}; L: (n_logical, n_mech) {0,1}. Returns (inputs, output, size_dict)."""
    H = np.asarray(H) % 2
    L = np.asarray(L) % 2
    n_mech = H.shape[1]
    inputs = []
    copy_legs = defaultdict(list)
    for c in range(H.shape[0]):
        _add_chain(inputs, copy_legs, f"c{c}", np.nonzero(H[c])[0].tolist())
    for o in range(L.shape[0]):
        _add_chain(inputs, copy_legs, f"L{o}", np.nonzero(L[o])[0].tolist(), open_index=f"O{o}")
    for m in range(n_mech):
        if copy_legs[m]:
            inputs.append(tuple(copy_legs[m]))
    output = tuple(f"O{o}" for o in range(L.shape[0]))
    size_dict = {ix: 2 for t in inputs for ix in t}
    return inputs, output, size_dict


def contraction_width(H, L, max_time=90, max_repeats=300):
    inputs, output, size_dict = build_indices(H, L)
    opt = ctg.HyperOptimizer(minimize="size", methods=["greedy", "random-greedy"],
                             max_repeats=max_repeats, max_time=max_time,
                             progbar=False, parallel=False)
    return float(opt.search(inputs, output, size_dict).contraction_width())
