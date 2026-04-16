#include "decoder/types.h"

namespace decoder {

void SyndromeGraph::build_adjacency() {
    adj.resize(n_detectors);
    for (auto& a : adj) a.clear();
    for (int i = 0; i < (int)edges.size(); i++) {
        if (edges[i].source >= 0 && edges[i].source < n_detectors)
            adj[edges[i].source].push_back(i);
        if (edges[i].target >= 0 && edges[i].target < n_detectors)
            adj[edges[i].target].push_back(i);
    }
}

} // namespace decoder
