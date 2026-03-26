#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/functional.h>

// Include autodiff headers
#include "ad/ag_all.hpp"

namespace py = pybind11;
using namespace ag;

void bind_autodiff(py::module &m) {
    // Value class - represents a node in the computation graph
    py::class_<Value, std::shared_ptr<Value>>(m, "Value")
        .def(py::init<>(), "Create empty Value")
        
        // Properties
        .def("val", &Value::val, "Get underlying tensor")
        .def("grad", &Value::grad, "Get gradient tensor")
        .def("shape", &Value::shape, "Get shape")
        .def("requires_grad", &Value::requires_grad, "Check if gradient tracking is enabled")
        .def("set_data", &Value::set_data, "Set underlying tensor data")
        
        // Operators
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(py::self * py::self)
        .def(py::self / py::self)
        .def(py::self += py::self)
        .def(py::self -= py::self)
        
        // Scalar operators
        .def(py::self + float())
        .def(py::self - float())
        .def(py::self * float())
        .def(py::self / float())
        .def(float() + py::self)
        
        .def("__repr__", [](const Value &v) {
            return "<Value shape=" + std::to_string(v.shape()[0]) + ">";
        });

    // Gradient computation
    m.def("backward", py::overload_cast<Value, std::shared_ptr<OwnTensor::Tensor>>(&backward),
        "Backpropagate gradients from root node", py::arg("root"), py::arg("grad_seed") = nullptr);

    m.def("zero_grad", &zero_grad,
        "Clear all gradients in computation graph rooted at Value");

    m.def("forward", [](Value root) {
        // Forward evaluation
        return forward(root);
    }, "Evaluate computation graph forward");

    // Autodiff operations - Math
    m.def("add", py::overload_cast<Value, Value>(&add), 
        "Element-wise addition");
    m.def("sub", py::overload_cast<Value, Value>(&sub), 
        "Element-wise subtraction");
    m.def("mul", py::overload_cast<Value, Value>(&mul), 
        "Element-wise multiplication");
    m.def("div", py::overload_cast<Value, Value>(&div), 
        "Element-wise division");
    
    m.def("matmul", &matmul, 
        "Matrix multiplication");
    m.def("sum", &sum, 
        "Sum all elements to scalar");
    
    // Scalar operations
    m.def("flomul", &flomul, 
        "Multiply tensor by scalar with gradient");
    m.def("floadd", &floadd, 
        "Add scalar to tensor with gradient");
    m.def("flodiv", &flodiv, 
        "Divide tensor by scalar with gradient");

    // Activation functions
    m.def("relu", &relu, "ReLU activation");
    m.def("gelu", &gelu, "GELU activation (Gaussian Error Linear Unit)");
    m.def("silu", &silu, "SiLU activation (Sigmoid Linear Unit)");
    m.def("swish", &swish, "Swish activation");
    m.def("tanh", &tanh, "Tanh activation");
    m.def("sigmoid", &sigmoid, "Sigmoid activation");
    m.def("leaky_relu", &leaky_relu, "Leaky ReLU with alpha parameter");
    m.def("mish", &mish, "Mish activation");
    m.def("softplus", &softplus, "Softplus activation");
    m.def("lisht", &lisht, "LiSHT activation");
    m.def("parcon", &parcon, "ParCon activation");
    m.def("gaus", &gaus, "Gaussian activation");
    m.def("gcu", &gcu, "Gated Linear Unit variant");

    // Advanced operations
    m.def("transpose", &transpose_op, 
        "Transpose matrix (swap last two dimensions)");
    m.def("rowsum", &rowsum, 
        "Sum across rows (reduce last dimension)");
    m.def("rowmax", &rowmax, 
        "Max across rows (reduce last dimension)");
    m.def("softmax_row", &softmax_row, 
        "Row-wise softmax");
    m.def("logsumexp_row", &logsumexp_row, 
        "Log-sum-exp across rows");
    m.def("expand", &expand, 
        "Expand tensor to new shape (broadcasting)");
    m.def("sign", &sign, 
        "Element-wise sign function");

    // Attention operations
    m.def("attention", [](Value q, Value k, Value v) {
        return attention(q, k, v);
    }, "Multi-head attention forward pass",
        py::arg("query"), py::arg("key"), py::arg("value"));

    m.def("alibiatt", [](Value q, Value k, Value v) {
        return alibiatt(q, k, v);
    }, "ALiBi-scaled attention forward pass",
        py::arg("query"), py::arg("key"), py::arg("value"));

    // Advanced layer operations
    m.def("swiglu", [](Value x, int in_features, int out_features, int hidden) {
        return swiglu(x, in_features, out_features, hidden);
    }, "SWIGLU gated linear unit",
        py::arg("x"), py::arg("in_features"), py::arg("out_features"), py::arg("hidden_dim"));

    m.def("rms", &rms, "RMS normalization");

    m.def("mambassm", [](Value x) {
        return mambassm(x);
    }, "Mamba state space model block");

    m.def("moewe", [](Value x, int num_experts, int expert_capacity) {
        return moewe(x, num_experts, expert_capacity);
    }, "Mixture of Experts with capacity",
        py::arg("x"), py::arg("num_experts"), py::arg("expert_capacity"));

    // Loss functions
    m.def("cross_entropy_with_logits", [](Value logits, Value targets) {
        return cross_entropy_with_logits(logits, targets);
    }, "Cross entropy loss with logits",
        py::arg("logits"), py::arg("targets"));

    m.def("kldivergence", [](Value p, Value q) {
        return kldivergence(p, q);
    }, "KL divergence: KL(P||Q)",
        py::arg("p"), py::arg("q"));

    m.def("mse_loss", [](Value pred, Value target) {
        return mse_loss(pred, target);
    }, "Mean Squared Error loss",
        py::arg("predictions"), py::arg("targets"));

    m.def("mae_loss", [](Value pred, Value target) {
        return mae_loss(pred, target);
    }, "Mean Absolute Error loss",
        py::arg("predictions"), py::arg("targets"));

    // Graph utilities
    m.def("make_tensor", [](std::shared_ptr<OwnTensor::Tensor> t, std::string name = "") {
        return make_tensor(t, name);
    }, "Wrap tensor in computation graph",
        py::arg("tensor"), py::arg("name") = "");

    m.def("param", [](std::shared_ptr<OwnTensor::Tensor> t, std::string name = "") {
        return param(t, name);
    }, "Create learnable parameter",
        py::arg("tensor"), py::arg("name") = "");

    m.def("constant", [](std::shared_ptr<OwnTensor::Tensor> t, std::string name = "") {
        return constant(t, name);
    }, "Create non-differentiable constant",
        py::arg("tensor"), py::arg("name") = "");

    // JIT Compilation (if available)
    // m.def("jit_compile", &jit::compile, "Compile computation graph for efficient inference");
}
