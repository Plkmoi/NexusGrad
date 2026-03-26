#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>

namespace py = pybind11;

void bind_layers(py::module &m) {
    m.doc() = "Neural network layers and modules";
    
    // Placeholder implementation - full implementation would wrap layer/*.hpp
    m.def("Linear", [](int in_features, int out_features) {
        return "Linear layer";
    }, "Fully connected layer");

    m.def("ReLU", []() {
        return "ReLU layer";
    }, "ReLU activation layer");

    m.def("GELU", []() {
        return "GELU layer";
    }, "GELU activation layer");

    m.def("Sequential", [](const std::vector<std::string> &layers) {
        return "Sequential module";
    }, "Sequential composition of layers");

    m.def("LayerNorm", [](const std::vector<int64_t> &shape) {
        return "LayerNorm layer";
    }, "Layer normalization");

    m.def("Attention", [](int batch_size, int in_features, int num_heads) {
        return "Attention layer";
    }, "Multi-head attention");

    m.def("Embedding", [](int vocab_size, int embed_dim) {
        return "Embedding layer";
    }, "Token embedding layer");
}
