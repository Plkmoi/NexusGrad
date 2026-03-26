#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/functional.h>

// Include layer headers
#include "layer/affine.hpp"
#include "layer/activation.hpp"
#include "layer/norm.hpp"
#include "layer/attention.hpp"
#include "layer/loss.hpp"
#include "layer/embed.hpp"

namespace py = pybind11;
using namespace ag::layer;

// Trampoline class for Layer to support Python subclassing
class PyLayer : public Layer {
public:
    using Layer::Layer;

    Value operator()(Value input) override {
        PYBIND11_OVERRIDE_PURE(Value, Layer, operator(), input);
    }

    std::vector<std::pair<std::string, std::shared_ptr<ag::Value>>> parameters() override {
        PYBIND11_OVERRIDE(std::vector<std::pair<std::string, std::shared_ptr<ag::Value>>>, 
                         Layer, parameters);
    }

    void to(OwnTensor::Device device) override {
        PYBIND11_OVERRIDE(void, Layer, to, device);
    }

    void zero_grad() override {
        PYBIND11_OVERRIDE(void, Layer, zero_grad);
    }
};

void bind_layers(py::module &m) {
    // Base Layer class
    py::class_<Layer, PyLayer, std::shared_ptr<Layer>>(m, "Layer")
        .def(py::init<>())
        .def("__call__", &Layer::operator(), "Forward pass")
        .def("parameters", &Layer::parameters, "Get learnable parameters")
        .def("to", &Layer::to, "Move layer to device")
        .def("zero_grad", &Layer::zero_grad, "Clear gradients");

    // ========== Affine Layers ==========
    
    py::class_<Linear, Layer, std::shared_ptr<Linear>>(m, "Linear")
        .def(py::init<int, int, OwnTensor::Device>(), 
             "Fully connected layer",
             py::arg("in_features"), py::arg("out_features"), 
             py::arg("device") = OwnTensor::Device::CPU)
        .def("__call__", &Linear::operator(), "Forward pass");

    py::class_<Traverse, Layer, std::shared_ptr<Traverse>>(m, "Sequential")
        .def(py::init<std::vector<std::shared_ptr<Layer>>>(), 
             "Sequential composition of layers",
             py::arg("layers"))
        .def("__call__", &Traverse::operator(), "Forward pass through all layers");

    py::class_<ResidualBlock, Layer, std::shared_ptr<ResidualBlock>>(m, "ResidualBlock")
        .def(py::init<std::vector<std::shared_ptr<Layer>>>(), 
             "Residual block: y = x + f(x)",
             py::arg("layers"))
        .def("__call__", &ResidualBlock::operator(), "Forward pass with residual connection");

    // ========== Activation Layers ==========
    
    py::class_<ReLU, Layer, std::shared_ptr<ReLU>>(m, "ReLU")
        .def(py::init<>(), "ReLU activation")
        .def("__call__", &ReLU::operator(), "Forward pass");

    py::class_<LeakyReLU, Layer, std::shared_ptr<LeakyReLU>>(m, "LeakyReLU")
        .def(py::init<float>(), "Leaky ReLU with alpha",
             py::arg("alpha") = 0.01)
        .def("__call__", &LeakyReLU::operator(), "Forward pass");

    py::class_<Tanh, Layer, std::shared_ptr<Tanh>>(m, "Tanh")
        .def(py::init<>(), "Tanh activation")
        .def("__call__", &Tanh::operator(), "Forward pass");

    py::class_<Sigmoid, Layer, std::shared_ptr<Sigmoid>>(m, "Sigmoid")
        .def(py::init<>(), "Sigmoid activation")
        .def("__call__", &Sigmoid::operator(), "Forward pass");

    py::class_<GELU, Layer, std::shared_ptr<GELU>>(m, "GELU")
        .def(py::init<>(), "GELU activation (Gaussian Error Linear Unit)")
        .def("__call__", &GELU::operator(), "Forward pass");

    py::class_<Swish, Layer, std::shared_ptr<Swish>>(m, "Swish")
        .def(py::init<>(), "Swish activation (SiLU)")
        .def("__call__", &Swish::operator(), "Forward pass");

    py::class_<Mish, Layer, std::shared_ptr<Mish>>(m, "Mish")
        .def(py::init<>(), "Mish activation")
        .def("__call__", &Mish::operator(), "Forward pass");

    py::class_<Softplus, Layer, std::shared_ptr<Softplus>>(m, "Softplus")
        .def(py::init<>(), "Softplus activation")
        .def("__call__", &Softplus::operator(), "Forward pass");

    py::class_<ParCon, Layer, std::shared_ptr<ParCon>>(m, "ParCon")
        .def(py::init<>(), "ParCon activation")
        .def("__call__", &ParCon::operator(), "Forward pass");

    py::class_<Gauss, Layer, std::shared_ptr<Gauss>>(m, "Gauss")
        .def(py::init<>(), "Gaussian activation")
        .def("__call__", &Gauss::operator(), "Forward pass");

    py::class_<LiSHT, Layer, std::shared_ptr<LiSHT>>(m, "LiSHT")
        .def(py::init<>(), "LiSHT activation")
        .def("__call__", &LiSHT::operator(), "Forward pass");

    py::class_<GCU, Layer, std::shared_ptr<GCU>>(m, "GCU")
        .def(py::init<>(), "Gated Linear Unit variant")
        .def("__call__", &GCU::operator(), "Forward pass");

    py::class_<SWIGLU, Layer, std::shared_ptr<SWIGLU>>(m, "SWIGLU")
        .def(py::init<int, int, int, OwnTensor::Device>(), 
             "SWIGLU gated linear unit",
             py::arg("batch"), py::arg("in_features"), py::arg("out_features"),
             py::arg("device") = OwnTensor::Device::CPU)
        .def("__call__", &SWIGLU::operator(), "Forward pass");

    py::class_<Softmax, Layer, std::shared_ptr<Softmax>>(m, "Softmax")
        .def(py::init<int>(), "Softmax along dimension",
             py::arg("dim") = -1)
        .def("__call__", &Softmax::operator(), "Forward pass");

    // ========== Normalization Layers ==========
    
    py::class_<LayerNorm, Layer, std::shared_ptr<LayerNorm>>(m, "LayerNorm")
        .def(py::init<std::vector<int64_t>, OwnTensor::Device>(), 
             "Layer normalization",
             py::arg("normalized_shape"), 
             py::arg("device") = OwnTensor::Device::CPU)
        .def("__call__", &LayerNorm::operator(), "Forward pass")
        .def("beta", &LayerNorm::beta, "Get beta (bias) parameters")
        .def("gamma", &LayerNorm::gamma, "Get gamma (scale) parameters");

    py::class_<RMSNorm, Layer, std::shared_ptr<RMSNorm>>(m, "RMSNorm")
        .def(py::init<int, OwnTensor::Device>(), 
             "RMS normalization",
             py::arg("dim"), 
             py::arg("device") = OwnTensor::Device::CPU)
        .def("__call__", &RMSNorm::operator(), "Forward pass");

    py::class_<DynTanh, Layer, std::shared_ptr<DynTanh>>(m, "DynTanh")
        .def(py::init<float, float, float, OwnTensor::Device>(), 
             "Dynamic Tanh normalization",
             py::arg("alpha"), py::arg("beta"), py::arg("gamma"),
             py::arg("device") = OwnTensor::Device::CPU)
        .def("__call__", &DynTanh::operator(), "Forward pass");

    // ========== Attention Layers ==========
    
    py::class_<Attention, Layer, std::shared_ptr<Attention>>(m, "Attention")
        .def(py::init<int, int, int, int, OwnTensor::Device>(), 
             "Multi-head attention",
             py::arg("batch"), py::arg("in_features"), py::arg("num_heads"), py::arg("out_features"),
             py::arg("device") = OwnTensor::Device::CPU)
        .def("__call__", &Attention::operator(), "Forward pass");

    py::class_<AlibiAttention, Layer, std::shared_ptr<AlibiAttention>>(m, "AlibiAttention")
        .def(py::init<int, int, int, int, OwnTensor::Device>(), 
             "ALiBi-scaled multi-head attention",
             py::arg("batch"), py::arg("in_features"), py::arg("out_features"), py::arg("num_heads"),
             py::arg("device") = OwnTensor::Device::CPU)
        .def("__call__", &AlibiAttention::operator(), "Forward pass");

    // ========== Loss Layers ==========
    
    py::class_<CEWithLogits, Layer, std::shared_ptr<CEWithLogits>>(m, "CrossEntropyWithLogits")
        .def(py::init<>(), "Cross entropy loss with logits")
        .def("__call__", 
             [](CEWithLogits &self, ag::Value logits, ag::Value targets) {
                 return self(logits, targets);
             }, 
             "Compute loss", py::arg("logits"), py::arg("targets"));

    py::class_<KLDivergence, Layer, std::shared_ptr<KLDivergence>>(m, "KLDivergence")
        .def(py::init<>(), "KL divergence loss")
        .def("__call__", 
             [](KLDivergence &self, ag::Value p, ag::Value q) {
                 return self(p, q);
             }, 
             "Compute KL(P||Q)", py::arg("p"), py::arg("q"));

    py::class_<MSELoss, Layer, std::shared_ptr<MSELoss>>(m, "MSELoss")
        .def(py::init<>(), "Mean squared error loss")
        .def("__call__", 
             [](MSELoss &self, ag::Value pred, ag::Value target) {
                 return self(pred, target);
             }, 
             "Compute loss", py::arg("predictions"), py::arg("targets"));

    py::class_<MAELoss, Layer, std::shared_ptr<MAELoss>>(m, "MAELoss")
        .def(py::init<>(), "Mean absolute error loss")
        .def("__call__", 
             [](MAELoss &self, ag::Value pred, ag::Value target) {
                 return self(pred, target);
             }, 
             "Compute loss", py::arg("predictions"), py::arg("targets"));

    // ========== Embedding Functions ==========
    
    m.def("make_embedding_table", [](int vocab_size, int embed_dim, OwnTensor::Device device) {
        return make_embedding_table(vocab_size, embed_dim, device);
    }, "Create embedding weight matrix",
        py::arg("vocab_size"), py::arg("embed_dim"), 
        py::arg("device") = OwnTensor::Device::CPU);

    m.def("embed_tokens", [](std::shared_ptr<OwnTensor::Tensor> W, std::vector<int64_t> token_ids) {
        return embed_tokens(W, token_ids);
    }, "Embed token sequence",
        py::arg("weight_matrix"), py::arg("token_ids"));

    m.def("embed_tokens_3d", [](std::shared_ptr<OwnTensor::Tensor> W, std::vector<std::vector<int64_t>> batch_tokens) {
        return embed_tokens_3d(W, batch_tokens);
    }, "Embed batched token sequences",
        py::arg("weight_matrix"), py::arg("batch_tokens"));
}
