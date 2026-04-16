#pragma once

#include "decoder/types.h"
#include <vector>

namespace decoder {

class UnionFindDecoder {
public:
    UnionFindDecoder(const SyndromeGraph& graph);
    DecoderResult decode(const std::vector<bool>& detection_events);

    // Decoder steering (Sivak et al. arXiv:2511.08493):
    // Dynamically reweight edges based on observed detection patterns.
    // Call after each decode() to adapt to non-stationary noise.
    void update_weights(const std::vector<bool>& detection_events,
                        double learning_rate = 0.01);
    void reset_weights();

private:
    SyndromeGraph graph_;
    std::vector<int> edge_weights_;       // Current integer weights (may be steered)
    std::vector<int> baseline_weights_;   // Original DEM-derived weights

    // Steering state
    std::vector<double> detection_rates_; // Per-detector exponential moving average
    int steering_count_;

    // Union-Find forest (re-initialized per decode call)
    struct UFNode {
        int parent;
        int rank;
        int cluster_size;  // Number of defects in this cluster
        bool is_boundary;
    };

    std::vector<UFNode> nodes_;

    struct EdgeState {
        int growth;        // Total growth from both sides
        bool fully_grown;
    };

    std::vector<EdgeState> edge_states_;
    int boundary_node_;

    // Core operations
    int find(int x);
    void unite(int x, int y);
    bool is_odd_cluster(int x);

    // Decoder phases
    void initialize(const std::vector<bool>& detection_events);
    bool growth_and_merge();
    std::vector<bool> peel(const std::vector<bool>& detection_events);
};

} // namespace decoder
