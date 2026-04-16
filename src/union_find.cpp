#include "decoder/union_find.h"
#include <algorithm>
#include <queue>
#include <cmath>
#include <numeric>

namespace decoder {

UnionFindDecoder::UnionFindDecoder(const SyndromeGraph& graph)
    : graph_(graph), boundary_node_(graph.n_detectors), steering_count_(0)
{
    // Pre-compute integer weights from error probabilities.
    edge_weights_.resize(graph_.edges.size());
    for (size_t i = 0; i < graph_.edges.size(); i++) {
        double p = graph_.edges[i].error_prob;
        if (p > 0 && p < 1) {
            edge_weights_[i] = std::max(1, (int)std::ceil(-std::log(p)));
        } else {
            edge_weights_[i] = 1;
        }
    }
    baseline_weights_ = edge_weights_;  // Save baseline for reset
    graph_.build_adjacency();
}

int UnionFindDecoder::find(int x) {
    while (nodes_[x].parent != x) {
        nodes_[x].parent = nodes_[nodes_[x].parent].parent;
        x = nodes_[x].parent;
    }
    return x;
}

void UnionFindDecoder::unite(int x, int y) {
    int rx = find(x), ry = find(y);
    if (rx == ry) return;
    if (nodes_[rx].rank < nodes_[ry].rank) std::swap(rx, ry);
    nodes_[ry].parent = rx;
    nodes_[rx].cluster_size += nodes_[ry].cluster_size;
    nodes_[rx].is_boundary = nodes_[rx].is_boundary || nodes_[ry].is_boundary;
    if (nodes_[rx].rank == nodes_[ry].rank) nodes_[rx].rank++;
}

bool UnionFindDecoder::is_odd_cluster(int x) {
    int root = find(x);
    return (nodes_[root].cluster_size % 2 == 1) && !nodes_[root].is_boundary;
}

void UnionFindDecoder::initialize(const std::vector<bool>& detection_events) {
    int n = graph_.n_detectors + 1;
    nodes_.resize(n);
    for (int i = 0; i < n; i++) {
        nodes_[i] = {i, 0, 0, (i == boundary_node_)};
    }
    for (int i = 0; i < graph_.n_detectors && i < (int)detection_events.size(); i++) {
        if (detection_events[i]) {
            nodes_[i].cluster_size = 1;
        }
    }
    edge_states_.assign(graph_.edges.size(), {0, false});
}

bool UnionFindDecoder::growth_and_merge() {
    bool any_odd = false;

    // Growth: odd clusters grow their boundary edges
    for (size_t e = 0; e < graph_.edges.size(); e++) {
        if (edge_states_[e].fully_grown) continue;

        const auto& edge = graph_.edges[e];
        int src = (edge.source >= 0) ? edge.source : boundary_node_;
        int tgt = (edge.target >= 0) ? edge.target : boundary_node_;

        bool src_odd = is_odd_cluster(src);
        bool tgt_odd = is_odd_cluster(tgt);

        if (src_odd || tgt_odd) {
            // Both-side growth: each odd endpoint contributes 1 growth unit.
            // An edge with two odd endpoints grows twice as fast.
            int inc = (src_odd ? 1 : 0) + (tgt_odd ? 1 : 0);
            edge_states_[e].growth += inc;
            any_odd = true;
        }

        // Fully grown when growth reaches the weight
        if (edge_states_[e].growth >= edge_weights_[e]) {
            edge_states_[e].fully_grown = true;
        }
    }

    if (!any_odd) return false;

    // Merge: unite endpoints of fully-grown edges
    for (size_t e = 0; e < graph_.edges.size(); e++) {
        if (!edge_states_[e].fully_grown) continue;
        const auto& edge = graph_.edges[e];
        int src = (edge.source >= 0) ? edge.source : boundary_node_;
        int tgt = (edge.target >= 0) ? edge.target : boundary_node_;
        unite(src, tgt);
    }

    return true;
}

std::vector<bool> UnionFindDecoder::peel(const std::vector<bool>& detection_events) {
    std::vector<bool> obs_pred(graph_.n_observables, false);
    int n = graph_.n_detectors + 1;

    // Collect fully-grown edges, sorted by weight (low weight = high probability first).
    // This builds a spanning tree that prefers high-probability error paths.
    std::vector<int> grown_edge_indices;
    for (size_t e = 0; e < graph_.edges.size(); e++) {
        if (edge_states_[e].fully_grown) {
            grown_edge_indices.push_back(e);
        }
    }
    std::sort(grown_edge_indices.begin(), grown_edge_indices.end(),
              [&](int a, int b) { return edge_weights_[a] < edge_weights_[b]; });

    // Build a cycle-free spanning tree using union-find
    std::vector<int> tree_parent(n);
    std::iota(tree_parent.begin(), tree_parent.end(), 0);
    std::vector<int> tree_rank(n, 0);

    auto tree_find = [&](int x) {
        while (tree_parent[x] != x) {
            tree_parent[x] = tree_parent[tree_parent[x]];
            x = tree_parent[x];
        }
        return x;
    };

    std::vector<std::vector<std::pair<int, int>>> adj(n);  // (neighbor, edge_idx)
    std::vector<int> degree(n, 0);

    for (int e_idx : grown_edge_indices) {
        const auto& edge = graph_.edges[e_idx];
        int src = (edge.source >= 0) ? edge.source : boundary_node_;
        int tgt = (edge.target >= 0) ? edge.target : boundary_node_;

        int rs = tree_find(src), rt = tree_find(tgt);
        if (rs != rt) {
            // Add to spanning tree (no cycle)
            if (tree_rank[rs] < tree_rank[rt]) std::swap(rs, rt);
            tree_parent[rt] = rs;
            if (tree_rank[rs] == tree_rank[rt]) tree_rank[rs]++;

            adj[src].push_back({tgt, e_idx});
            adj[tgt].push_back({src, e_idx});
            degree[src]++;
            degree[tgt]++;
        }
    }

    // Peel from leaves: if leaf subtree has odd defect parity, include the edge
    std::queue<int> leaves;
    for (int i = 0; i < n; i++) {
        if (degree[i] == 1) leaves.push(i);
    }

    std::vector<bool> node_done(n, false);
    std::vector<bool> edge_done(graph_.edges.size(), false);

    // Use ORIGINAL detection events for parity (not merged cluster sizes)
    std::vector<int> defect_parity(n, 0);
    for (int i = 0; i < graph_.n_detectors && i < (int)detection_events.size(); i++) {
        if (detection_events[i]) {
            defect_parity[i] = 1;
        }
    }

    while (!leaves.empty()) {
        int leaf = leaves.front();
        leaves.pop();
        if (node_done[leaf]) continue;
        node_done[leaf] = true;

        for (auto& [nbr, e_idx] : adj[leaf]) {
            if (edge_done[e_idx] || node_done[nbr]) continue;
            edge_done[e_idx] = true;

            if (defect_parity[leaf] % 2 == 1) {
                // This edge is part of the correction
                const auto& edge = graph_.edges[e_idx];
                for (int obs : edge.observable_mask) {
                    if (obs >= 0 && obs < (int)obs_pred.size()) {
                        obs_pred[obs] = !obs_pred[obs];
                    }
                }
                // Propagate parity to neighbor
                defect_parity[nbr] += defect_parity[leaf];
            }

            degree[nbr]--;
            if (degree[nbr] == 1) leaves.push(nbr);
        }
    }

    return obs_pred;
}

DecoderResult UnionFindDecoder::decode(const std::vector<bool>& detection_events) {
    initialize(detection_events);

    // Growth + merge until no odd clusters remain
    int max_rounds = 0;
    for (int w : edge_weights_) max_rounds = std::max(max_rounds, w);
    max_rounds += 10;

    for (int round = 0; round < max_rounds; round++) {
        if (!growth_and_merge()) break;
    }

    auto prediction = peel(detection_events);

    DecoderResult result;
    result.observable_prediction = prediction;
    result.confidence = 1.0;
    result.converged = true;
    return result;
}

void UnionFindDecoder::update_weights(const std::vector<bool>& detection_events,
                                       double learning_rate) {
    // Initialize detection rate tracking on first call
    if (detection_rates_.empty()) {
        detection_rates_.resize(graph_.n_detectors, 0.0);
    }
    steering_count_++;

    // Exponential moving average of per-detector firing rates
    for (int i = 0; i < graph_.n_detectors && i < (int)detection_events.size(); i++) {
        double observed = detection_events[i] ? 1.0 : 0.0;
        detection_rates_[i] =
            (1.0 - learning_rate) * detection_rates_[i] +
            learning_rate * observed;
    }

    // Reweight edges every 100 syndromes (amortize overhead)
    if (steering_count_ % 100 == 0) {
        for (size_t e = 0; e < graph_.edges.size(); e++) {
            const auto& edge = graph_.edges[e];

            // Average detection rate of this edge's endpoints
            double det_rate = 0.0;
            int n_det = 0;
            if (edge.source >= 0 && edge.source < (int)detection_rates_.size()) {
                det_rate += detection_rates_[edge.source];
                n_det++;
            }
            if (edge.target >= 0 && edge.target < (int)detection_rates_.size()) {
                det_rate += detection_rates_[edge.target];
                n_det++;
            }
            if (n_det > 0) det_rate /= n_det;

            // Higher detection rate → lower weight → fuse faster
            // Blend: 80% baseline + 20% observed
            double observed_weight = (det_rate > 1e-6) ?
                std::max(1.0, -std::log(std::min(det_rate, 0.5))) :
                (double)baseline_weights_[e];
            edge_weights_[e] = std::max(1, (int)std::round(
                0.8 * baseline_weights_[e] + 0.2 * observed_weight));
        }
    }
}

void UnionFindDecoder::reset_weights() {
    edge_weights_ = baseline_weights_;
    detection_rates_.clear();
    steering_count_ = 0;
}

} // namespace decoder
