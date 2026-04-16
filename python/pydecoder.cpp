#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "decoder/types.h"
#include "decoder/union_find.h"

namespace py = pybind11;

PYBIND11_MODULE(pydecoder, m) {
    m.doc() = "QEC decoder library — Union-Find and BP decoders";

    py::class_<decoder::GraphEdge>(m, "GraphEdge")
        .def(py::init<>())
        .def_readwrite("source", &decoder::GraphEdge::source)
        .def_readwrite("target", &decoder::GraphEdge::target)
        .def_readwrite("weight", &decoder::GraphEdge::weight)
        .def_readwrite("error_prob", &decoder::GraphEdge::error_prob)
        .def_readwrite("observable_mask", &decoder::GraphEdge::observable_mask)
    ;

    py::class_<decoder::SyndromeGraph>(m, "SyndromeGraph")
        .def(py::init<>())
        .def_readwrite("n_detectors", &decoder::SyndromeGraph::n_detectors)
        .def_readwrite("n_observables", &decoder::SyndromeGraph::n_observables)
        // Use def_property for edges to avoid the copy-on-read trap
        .def_property("edges",
            [](const decoder::SyndromeGraph& sg) { return sg.edges; },
            [](decoder::SyndromeGraph& sg, const std::vector<decoder::GraphEdge>& edges) {
                sg.edges = edges;
            })
        .def("build_adjacency", &decoder::SyndromeGraph::build_adjacency)
        .def("add_edge", [](decoder::SyndromeGraph& sg, int src, int tgt,
                           double error_prob, std::vector<int> obs_mask) {
            decoder::GraphEdge e;
            e.source = src;
            e.target = tgt;
            e.error_prob = error_prob;
            e.weight = 0.0;
            e.observable_mask = std::move(obs_mask);
            sg.edges.push_back(e);
        }, py::arg("source"), py::arg("target"), py::arg("error_prob"),
           py::arg("observable_mask") = std::vector<int>{})
    ;

    py::class_<decoder::DecoderResult>(m, "DecoderResult")
        .def_readonly("observable_prediction", &decoder::DecoderResult::observable_prediction)
        .def_readonly("confidence", &decoder::DecoderResult::confidence)
        .def_readonly("converged", &decoder::DecoderResult::converged)
    ;

    py::class_<decoder::UnionFindDecoder>(m, "UnionFindDecoder")
        .def(py::init<const decoder::SyndromeGraph&>())
        .def("decode", &decoder::UnionFindDecoder::decode)
        .def("decode_batch", [](decoder::UnionFindDecoder& dec,
                                py::array_t<bool> det_events) {
            auto buf = det_events.unchecked<2>();
            int n_shots = buf.shape(0);
            int n_det = buf.shape(1);

            std::vector<bool> first_det(n_det);
            for (int d = 0; d < n_det; d++) first_det[d] = buf(0, d);
            auto first_result = dec.decode(first_det);
            int n_obs = first_result.observable_prediction.size();

            auto predictions = py::array_t<bool>({n_shots, n_obs});
            auto pred_buf = predictions.mutable_unchecked<2>();

            for (int o = 0; o < n_obs; o++) {
                pred_buf(0, o) = first_result.observable_prediction[o];
            }

            for (int s = 1; s < n_shots; s++) {
                std::vector<bool> det(n_det);
                for (int d = 0; d < n_det; d++) det[d] = buf(s, d);
                auto result = dec.decode(det);
                for (int o = 0; o < n_obs; o++) {
                    pred_buf(s, o) = result.observable_prediction[o];
                }
            }
            return predictions;
        })
        .def("update_weights", &decoder::UnionFindDecoder::update_weights,
             py::arg("detection_events"), py::arg("learning_rate") = 0.01)
        .def("reset_weights", &decoder::UnionFindDecoder::reset_weights)
    ;
}
