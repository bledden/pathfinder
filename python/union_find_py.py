"""Pure Python Union-Find decoder for local testing.

Implements the peeling Union-Find decoder (Delfosse & Nickerson, 2021)
with weighted edges. For production speed, use the C++ pybind11 version.
"""
import numpy as np
from collections import deque
from typing import List, Tuple
from stim_interface import DecoderGraph


class UnionFindDecoder:
    def __init__(self, graph: DecoderGraph):
        self.n_det = graph.n_detectors
        self.n_obs = graph.n_observables
        self.boundary = self.n_det  # Virtual boundary node
        self.n = self.n_det + 1

        # Parse edges with weights
        self.edges = []  # (s, t, prob, obs_mask, weight)
        self.adj = [[] for _ in range(self.n)]

        for i, (src, tgt, prob, obs_mask) in enumerate(graph.edges):
            s = src if src >= 0 else self.boundary
            t = tgt if tgt >= 0 else self.boundary
            # Weight = integer-quantized LLR; higher = less likely
            if 0 < prob < 1:
                w = max(1, int(round(-np.log(prob / (1 - prob)) * 10)))
            else:
                w = 1
            self.edges.append((s, t, prob, obs_mask, w))
            self.adj[s].append(i)
            self.adj[t].append(i)

    def decode(self, detection_events: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Decode a single syndrome using weighted Union-Find with peeling.

        Args:
            detection_events: bool array [n_detectors]

        Returns:
            prediction: bool array [n_observables]
            converged: bool
        """
        n = self.n
        n_edges = len(self.edges)

        # UF forest
        parent = list(range(n))
        rank_uf = [0] * n
        cluster_size = [0] * n
        is_boundary = [False] * n
        is_boundary[self.boundary] = True

        # Mark defects
        for i in range(self.n_det):
            if detection_events[i]:
                cluster_size[i] = 1

        # Edge growth state
        edge_growth = [0] * n_edges
        edge_fused = [False] * n_edges

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def unite(a, b):
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            if rank_uf[ra] < rank_uf[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            cluster_size[ra] += cluster_size[rb]
            is_boundary[ra] = is_boundary[ra] or is_boundary[rb]
            if rank_uf[ra] == rank_uf[rb]:
                rank_uf[ra] += 1

        def is_odd(x):
            r = find(x)
            return cluster_size[r] % 2 == 1 and not is_boundary[r]

        # Weighted growth + merge loop
        max_weight = max(e[4] for e in self.edges) if self.edges else 1
        max_iters = max_weight + n + 10

        for _ in range(max_iters):
            any_odd = False
            fuse_list = []

            # Growth: grow edges touching odd clusters
            for e in range(n_edges):
                if edge_fused[e]:
                    continue
                s, t, prob, obs, w = self.edges[e]
                s_odd = is_odd(s)
                t_odd = is_odd(t)

                if s_odd or t_odd:
                    grow = (1 if s_odd else 0) + (1 if t_odd else 0)
                    edge_growth[e] += grow
                    any_odd = True

                    if edge_growth[e] >= w:
                        edge_fused[e] = True
                        fuse_list.append(e)

            if not any_odd:
                break

            # Merge
            for e in fuse_list:
                s, t, prob, obs, w = self.edges[e]
                if find(s) != find(t):
                    unite(s, t)

        # Peel phase
        obs_pred = [False] * self.n_obs

        # Build adjacency of fused edges
        fused_adj = [[] for _ in range(n)]
        for e in range(n_edges):
            if not edge_fused[e]:
                continue
            s, t, prob, obs, w = self.edges[e]
            fused_adj[s].append((t, e, w))
            fused_adj[t].append((s, e, w))

        # Build minimum spanning tree via Kruskal on fused edges
        # This ensures the peeling tree uses the shortest (most likely) paths
        fused_indices = [e for e in range(n_edges) if edge_fused[e]]
        fused_indices.sort(key=lambda e: self.edges[e][4])

        # Separate UF for spanning tree construction
        st_parent = list(range(n))
        st_rank = [0] * n

        def st_find(x):
            while st_parent[x] != x:
                st_parent[x] = st_parent[st_parent[x]]
                x = st_parent[x]
            return x

        def st_unite(a, b):
            ra, rb = st_find(a), st_find(b)
            if ra == rb:
                return False
            if st_rank[ra] < st_rank[rb]:
                ra, rb = rb, ra
            st_parent[rb] = ra
            if st_rank[ra] == st_rank[rb]:
                st_rank[ra] += 1
            return True

        tree_adj = [[] for _ in range(n)]
        degree = [0] * n

        for e in fused_indices:
            s, t, prob, obs, w = self.edges[e]
            if st_unite(s, t):
                tree_adj[s].append((t, e))
                tree_adj[t].append((s, e))
                degree[s] += 1
                degree[t] += 1

        # Peel from leaves
        leaves = deque()
        for i in range(n):
            if degree[i] == 1:
                leaves.append(i)

        node_done = [False] * n
        edge_done = [False] * n_edges
        defect_parity = [0] * n
        for i in range(self.n_det):
            if detection_events[i]:
                defect_parity[i] = 1

        while leaves:
            leaf = leaves.popleft()
            if node_done[leaf]:
                continue
            node_done[leaf] = True

            for nbr, e_idx in tree_adj[leaf]:
                if edge_done[e_idx] or node_done[nbr]:
                    continue
                edge_done[e_idx] = True

                if defect_parity[leaf] % 2 == 1:
                    s, t, prob, obs_mask, w = self.edges[e_idx]
                    for obs in obs_mask:
                        if 0 <= obs < self.n_obs:
                            obs_pred[obs] = not obs_pred[obs]
                    defect_parity[nbr] += defect_parity[leaf]

                degree[nbr] -= 1
                if degree[nbr] == 1:
                    leaves.append(nbr)

        return np.array(obs_pred, dtype=bool), True

    def decode_batch(self, syndromes: np.ndarray) -> np.ndarray:
        """Decode a batch of syndromes.

        Returns:
            predictions: bool array [n_shots, n_observables]
        """
        n_shots = syndromes.shape[0]
        predictions = np.zeros((n_shots, self.n_obs), dtype=bool)
        for i in range(n_shots):
            pred, _ = self.decode(syndromes[i])
            predictions[i] = pred
        return predictions
