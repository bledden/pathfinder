#pragma once

#include <cstdint>
#include <vector>
#include <string>

namespace decoder {

struct Defect {
    int detector_id;
    int x, y;
    int t;
};

struct GraphEdge {
    int source;         // Detector index (-1 = boundary)
    int target;         // Detector index (-1 = boundary)
    double weight;
    double error_prob;
    std::vector<int> observable_mask;
};

struct SyndromeGraph {
    int n_detectors;
    int n_observables;
    std::vector<GraphEdge> edges;
    std::vector<std::vector<int>> adj;  // adj[detector_id] = list of edge indices

    void build_adjacency();
};

struct DecoderResult {
    std::vector<bool> observable_prediction;
    double confidence;
    bool converged;
};

} // namespace decoder
