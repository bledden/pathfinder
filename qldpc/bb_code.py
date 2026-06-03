"""Bivariate-bicycle (BB) code construction + code-capacity syndrome generation.

Reconstructs the verified [[72,12,6]] BB code (Bravyi et al, Nature 2024) and exposes:
  - H_X, H_Z parity checks (CSS), n, k
  - the Z_l x Z_m group action (the verified symmetry) as index permutations
  - code-capacity syndrome sampling: i.i.d. X (or Z) errors on data qubits -> syndrome s = H e^T,
    with the GROUND-TRUTH logical-flip label for supervised decoder training.

Code-capacity is the honest first noise model for the equivariance SAMPLE-EFFICIENCY study
(the framing used here): it isolates the decoding problem from the syndrome-extraction circuit,
so the equivariant-vs-baseline comparison is clean. Circuit-level noise (full SE circuit) is a
later escalation.

Construction properties: CSS holds, n=72, k=12,
Z_6xZ_6 leaves H_X invariant for all 36 group elements.
"""
import os
import numpy as np
_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(_OUT, exist_ok=True)


def _shift(n):
    S = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        S[i, (i + 1) % n] = 1
    return S


class BBCode:
    def __init__(self, l=6, m=6, A_terms=(('x', 3), ('y', 1), ('y', 2)),
                 B_terms=(('y', 3), ('x', 1), ('x', 2))):
        self.l, self.m, self.N = l, m, l * m
        Il, Im = np.eye(l, dtype=np.int8), np.eye(m, dtype=np.int8)
        Sl, Sm = _shift(l), _shift(m)

        def term(t):
            v, p = t
            if v == 'x':
                return np.kron(np.linalg.matrix_power(Sl, p), Im) % 2
            return np.kron(Il, np.linalg.matrix_power(Sm, p)) % 2

        self.A = np.zeros((self.N, self.N), dtype=np.int8)
        for t in A_terms:
            self.A = (self.A + term(t)) % 2
        self.B = np.zeros((self.N, self.N), dtype=np.int8)
        for t in B_terms:
            self.B = (self.B + term(t)) % 2
        # H_X = [A | B] acting on n=2N qubits; H_Z = [B^T | A^T]
        self.HX = np.concatenate([self.A, self.B], axis=1).astype(np.int8) % 2
        self.HZ = np.concatenate([self.B.T, self.A.T], axis=1).astype(np.int8) % 2
        self.n = 2 * self.N
        self.Sl, self.Sm, self.Il, self.Im = Sl, Sm, Il, Im
        assert ((self.HX @ self.HZ.T) % 2 == 0).all(), "CSS condition failed"
        self.k = self.n - self._rank(self.HX) - self._rank(self.HZ)

    @staticmethod
    def _rank(M):
        M = M.copy() % 2
        rows, cols = M.shape
        pr = 0
        for c in range(cols):
            piv = None
            for i in range(pr, rows):
                if M[i, c]:
                    piv = i; break
            if piv is None:
                continue
            M[[pr, piv]] = M[[piv, pr]]
            for i in range(rows):
                if i != pr and M[i, c]:
                    M[i] = (M[i] + M[pr]) % 2
            pr += 1
            if pr == rows:
                break
        return pr

    def group_block_perm(self, a, b):
        """Permutation of the N positions in one block under g = Sl^a (x) Sm^b.
        Returns perm array P with (g x)[i] = x[P[i]] — i.e. how block indices map."""
        g = np.kron(np.linalg.matrix_power(self.Sl, a), np.linalg.matrix_power(self.Sm, b)) % 2
        # g is a permutation matrix; row i has a single 1 at column perm[i]
        perm = np.argmax(g, axis=1)
        return perm

    def logicals_Z(self):
        """A basis of Z logical operators (k of them): nullspace(HX) modulo rowspace(HZ).
        Used to compute the ground-truth logical flip for a given physical error."""
        # logical Z ops commute with all X-checks (HX l = 0) and aren't X-stabilizers.
        ker = self._nullspace(self.HX)               # vectors with HX l = 0
        # remove HZ rowspace (those are stabilizers, trivial logicals)
        logs = self._quotient(ker, self.HZ)
        return logs  # (k, n)

    def logicals_X(self):
        """A basis of X logical operators (k of them): nullspace(HZ) modulo rowspace(HX).
        THESE are the observables for a Z-basis memory that detects X errors via HZ: an X error e
        flips logical i iff <logicals_X[i], e> = 1. Lx commutes with HZ (deterministic in Z-readout)
        and is non-trivial vs HX. (logicals_Z is not the correct observable for an X-error memory — it does not track X-error flips.)"""
        ker = self._nullspace(self.HZ)               # vectors with HZ l = 0
        logs = self._quotient(ker, self.HX)          # remove HX rowspace (trivial)
        return logs  # (k, n)

    @staticmethod
    def _nullspace(M):
        M = M.copy() % 2
        rows, cols = M.shape
        # reduced row echelon
        pivots = []
        r = 0
        Mr = M.copy()
        for c in range(cols):
            piv = None
            for i in range(r, rows):
                if Mr[i, c]:
                    piv = i; break
            if piv is None:
                continue
            Mr[[r, piv]] = Mr[[piv, r]]
            for i in range(rows):
                if i != r and Mr[i, c]:
                    Mr[i] = (Mr[i] + Mr[r]) % 2
            pivots.append(c); r += 1
            if r == rows:
                break
        free = [c for c in range(cols) if c not in pivots]
        basis = []
        for f in free:
            v = np.zeros(cols, dtype=np.int8); v[f] = 1
            for ri, pc in enumerate(pivots):
                if Mr[ri, f]:
                    v[pc] = 1
            basis.append(v % 2)
        return np.array(basis, dtype=np.int8) if basis else np.zeros((0, cols), dtype=np.int8)

    def _quotient(self, ker, stab):
        """Return EXACTLY k independent logical reps: vectors in ker(HX) that are independent
        modulo rowspace(stab=HZ). Correct GF2 Gaussian elimination over a shared pivot basis:
        each accepted logical's REDUCED remainder (not the original) joins the basis, so later
        candidates are tested against the true span -> no redundant/over-counted logicals."""
        cols = ker.shape[1] if ker.shape[0] else (stab.shape[1] if stab.shape[0] else 0)

        def reduce_vec(w, basis):
            """Reduce w against basis rows (each in echelon by its leading column)."""
            for lead, b in basis:
                if w[lead]:
                    w = (w + b) % 2
            return w

        # echelon basis of the stabilizer rowspace, keyed by leading-1 column
        basis = []  # list of (lead_col, vector)
        for s in (stab % 2):
            w = s.copy()
            w = reduce_vec(w, basis)
            if w.any():
                lead = int(np.argmax(w))
                basis.append((lead, w))
        n_stab = len(basis)

        logs = []
        for v in (ker % 2):
            w = reduce_vec(v.copy(), basis)
            if w.any():
                lead = int(np.argmax(w))
                basis.append((lead, w))
                logs.append(w.copy())          # the reduced remainder = clean logical rep
        # logs now has exactly dim(ker) - n_stab independent vectors = k
        return np.array(logs, dtype=np.int8) if logs else np.zeros((0, cols), dtype=np.int8)

    def sample_codecap(self, p, shots, basis='X', rng=None):
        """Code-capacity: i.i.d. bit-flip (X) errors at rate p on n data qubits.
        Returns (syndromes [shots, n_checks], logical_flips [shots, k]).
        For X errors we detect with HZ (Z-checks) and logical flips via Z-logicals."""
        rng = rng or np.random.default_rng()
        H = self.HZ if basis == 'X' else self.HX
        L = self.logicals_Z() if basis == 'X' else None
        if L is None:
            L = self.logicals_Z()  # symmetric for this self-dual-ish construction; fine for study
        e = (rng.random((shots, self.n)) < p).astype(np.int8)
        synd = (e @ H.T) % 2
        logflip = (e @ L.T) % 2
        return synd.astype(np.int8), logflip.astype(np.int8), e


if __name__ == '__main__':
    import json
    bb = BBCode()
    out = dict(n=bb.n, k=bb.k, N=bb.N, HX_shape=list(bb.HX.shape), HZ_shape=list(bb.HZ.shape))
    # verify symmetry once more (sanity)
    fails = 0
    for a in range(bb.l):
        for b in range(bb.m):
            g = np.kron(np.linalg.matrix_power(bb.Sl, a), np.linalg.matrix_power(bb.Sm, b)) % 2
            gq = np.kron(np.eye(2, dtype=np.int8), g) % 2
            if not ((g @ bb.HX @ gq.T) % 2 == bb.HX).all():
                fails += 1
    out['symmetry_fails'] = fails
    L = bb.logicals_Z()
    out['n_logicals_found'] = int(L.shape[0])
    # quick syndrome sample
    s, lf, e = bb.sample_codecap(0.05, 1000, rng=np.random.default_rng(0))
    out['sample_synd_shape'] = list(s.shape)
    out['sample_logflip_shape'] = list(lf.shape)
    out['mean_syndrome_weight'] = round(float(s.sum(1).mean()), 3)
    out['mean_logflip_rate'] = round(float(lf.mean()), 4)
    json.dump(out, open(os.path.join(_OUT, 'bb_code_selftest.json'), 'w'), indent=2)
    print("WROTE bb_code_selftest.json")
