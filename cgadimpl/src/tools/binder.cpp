#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "ad/nodeops.hpp"
#include "ad/runtime.hpp"
#include "ad/kernels_api.hpp"
#include "ad/ag_all.hpp"
#include "tensor.hpp"
#include "optim.hpp"
#include "TensorLib.h"
#include "ad/graph.hpp"
#include "ad/debug.hpp"
#include "ad/autodiff.hpp"



#include <sstream>
#include "TensorLib.h"

namespace py = pybind11;
using namespace OwnTensor;


// #include "/home/blubridge-034/Downloads/Newf/cgadimpl/cgadimpl/binder_out/docstrings/nn_docs.hpp"
#include "/home/blubridge-034/Downloads/Newf/cgadimpl/cgadimpl/binder_out/docstrings/ops_docs.hpp"
#include "/home/blubridge-034/Downloads/Newf/cgadimpl/cgadimpl/binder_out/docstrings/optim_docs.hpp"
#include "/home/blubridge-034/Downloads/Newf/cgadimpl/cgadimpl/binder_out/docstrings/graph_docs.hpp"
#include "/home/blubridge-034/Downloads/Newf/cgadimpl/cgadimpl/binder_out/docstrings/tensor_docs.hpp"
#include "/home/blubridge-034/Downloads/Newf/cgadimpl/cgadimpl/binder_out/docstrings/kernels_api_docs.hpp"
#include "/home/blubridge-034/Downloads/Newf/cgadimpl/cgadimpl/binder_out/docstrings/debug_docs.hpp"
#include "/home/blubridge-034/Downloads/Newf/cgadimpl/cgadimpl/binder_out/docstrings/autodiff_docs.hpp"

using namespace ag; // or your actual namespace


namespace py_utils {

// Convert a Python list/tuple of ints into OwnTensor::Shape
inline OwnTensor::Shape to_shape(const py::sequence& seq) {
    OwnTensor::Shape s;
    s.dims.reserve(seq.size());
    for (auto item : seq)
        s.dims.push_back(py::cast<int64_t>(item));
    return s;
}

// Build TensorOptions from kwargs
inline OwnTensor::TensorOptions parse_opts(
    std::optional<std::string> device = std::nullopt,
    std::optional<std::string> dtype = std::nullopt,
    bool requires_grad = false
) {
    using namespace OwnTensor;
    TensorOptions opts;
opts = opts.with_req_grad(requires_grad);
    if (device) {
        if (device == "cuda" || device == "CUDA")
            opts = opts.with_device(DeviceIndex(Device::CUDA));
        else
            opts = opts.with_device(DeviceIndex(Device::CPU));
    }

    if (dtype) {
        if (*dtype == "float32" || *dtype == "f32")
            opts = opts.with_dtype(Dtype::Float32);
        else if (*dtype == "float64" || *dtype == "f64")
            opts = opts.with_dtype(Dtype::Float64);
        // add more if needed
    }

    return opts;
}

}

void bind_optim(py::module_ &m) {
    using namespace ag;

    py::module m_optim = m.def_submodule("optim", "ag backend optimizers");

    // Binding the SGD function (Stochastic Gradient Descent) with an optional grad_seed
m_optim.def("SGD",
    [](const Value& root, float learning_rate) {
        ag::SGD(root, nullptr, learning_rate);
    },
    py::arg("root"),
    py::arg("learning_rate") = 0.1,
    DOC(ag, SGD),  py::call_guard<py::gil_scoped_release>());
}

void bind_ops(py::module_ &m) {
    py::module m_ops = m.def_submodule("ops", "ag backend operations");

    // Arithmetic Operations
    m_ops.def("add", &ag::add, py::arg("a"), py::arg("b"), DOC(ag, add));
    m_ops.def("sub", &ag::sub, py::arg("a"), py::arg("b"), DOC(ag, sub));
    m_ops.def("mul", &ag::mul, py::arg("a"), py::arg("b"), DOC(ag, mul));
    m_ops.def("div", &ag::div, py::arg("a"), py::arg("b"), DOC(ag, div));
    m_ops.def("matmul", &ag::matmul, py::arg("a"), py::arg("b"), DOC(ag, matmul));

    // // Unary Operations
    m_ops.def("relu", &ag::relu, py::arg("x"), DOC(ag, relu));
    m_ops.def("gelu", &ag::gelu, py::arg("x"), DOC(ag, gelu));
    // m_ops.def("leaky_relu", &ag::leaky_relu, py::arg("x"), py::arg("alpha") = 0.01f, DOC(ag, leaky_relu));
    m_ops.def("silu", &ag::silu, py::arg("x"), DOC(ag, silu));
    m_ops.def("sigmoid", &ag::sigmoid, py::arg("x"), DOC(ag, sigmoid));
    m_ops.def("softplus", &ag::softplus, py::arg("x"), DOC(ag, softplus));
    m_ops.def("tanh", &ag::tanh, py::arg("x"), DOC(ag, tanh));

    // // More operations
    m_ops.def("sum", &ag::sum, py::arg("x"), DOC(ag, sum));
    m_ops.def("exp", &ag::exp, py::arg("x"), DOC(ag, exp));
    m_ops.def("log", &ag::log, py::arg("x"), DOC(ag, log));
    m_ops.def("sign", &ag::sign, py::arg("a"), py::arg("b"), DOC(ag, sign));

    // // Advanced ops
    m_ops.def("flomul", &ag::flomul, py::arg("a"), py::arg("b"), DOC(ag, flomul));
    m_ops.def("floadd", &ag::floadd, py::arg("a"), py::arg("b"), DOC(ag, floadd));
    m_ops.def("flodiv", &ag::flodiv, py::arg("a"), py::arg("b"), DOC(ag, flodiv));
    m_ops.def("fmab", &ag::fmab, py::arg("a"), py::arg("b"), py::arg("c"), DOC(ag, fmab));
    m_ops.def("linear", &ag::linear, py::arg("a"), py::arg("b"), py::arg("c"), DOC(ag, linear));
    m_ops.def("mse_loss", &ag::mse_loss, py::arg("pred"), py::arg("target"), DOC(ag, mse_loss));
    m_ops.def("mae_loss", &ag::mae_loss, py::arg("pred"), py::arg("target"), DOC(ag, mae_loss));

    // // Complex operations
    m_ops.def("attention", &ag::attention, py::arg("a"), py::arg("b"), py::arg("c"), py::arg("d"), DOC(ag, attention));
    m_ops.def("cross_entropy_with_logits", &ag::cross_entropy_with_logits, py::arg("logits"), py::arg("onehot"), DOC(ag, cross_entropy_with_logits));
    m_ops.def("kldivergence", &ag::kldivergence, py::arg("logits"), py::arg("onehot"), DOC(ag, kldivergence));

    // // Row operations
    m_ops.def("rowsum", &ag::rowsum, py::arg("x"), DOC(ag, rowsum));
    m_ops.def("rowmax", &ag::rowmax, py::arg("x"), DOC(ag, rowmax));
    m_ops.def("mean_all", &ag::mean_all, py::arg("x"), DOC(ag, mean_all));
    m_ops.def("softmax_row", &ag::softmax_row, py::arg("z"), DOC(ag, softmax_row));
    m_ops.def("logsumexp_row", &ag::logsumexp_row, py::arg("z"), DOC(ag, logsumexp_row));

    // // Special functions
    m_ops.def("rms", &ag::rms, py::arg("x"), DOC(ag, rms));
    m_ops.def("realrms", &ag::realrms, py::arg("x"), py::arg("g"), DOC(ag, realrms));
    m_ops.def("swiglu", &ag::swiglu, py::arg("x"), py::arg("a"), py::arg("b"), py::arg("c"), py::arg("d"), DOC(ag, swiglu));
    m_ops.def("parcon", &ag::parcon, py::arg("x"), DOC(ag, parcon));
    m_ops.def("gcu", &ag::gcu, py::arg("x"), DOC(ag, gcu));

    // // Other miscellaneous ops
    m_ops.def("laynor", &ag::laynor, py::arg("x"), DOC(ag, laynor));
    m_ops.def("alibiatt", &ag::alibiatt, py::arg("a"), py::arg("b"), py::arg("c"), py::arg("d"), py::arg("m"), DOC(ag, alibiatt));

    // // Transpose (yay)
    m_ops.def("transpose", &ag::transpose, py::arg("x"), DOC(ag, transpose));
}

void bind_value(py::module_ &m) {
    py::class_<Value, std::shared_ptr<Value>>(m, "Value", DOC(ag, Value))
        .def(py::init<>(), DOC(ag, Value, Value))
        .def(py::init<std::shared_ptr<Node>>(), DOC(ag, Value, Value_2))
        // node is a public member
        .def_readwrite("node", &Value::node, DOC(ag, Value, node))
        // methods returning references — use property wrappers
        .def_property_readonly(
            "val",
            (Tensor& (Value::*)()) &Value::val,
            DOC(ag, Value, val)
        )
        .def_property_readonly(
            "grad",
            (Tensor& (Value::*)()) &Value::grad,
            DOC(ag, Value, grad)
        )
        .def_property_readonly(
            "shape",
            &Value::shape,
            DOC(ag, Value, shape)
        )
        .def("shape_2d", &Value::shape_2d)
        ;
        m.def("make_tensor", &make_tensor, py::arg("v"), py::arg("name") = "", DOC(ag, make_tensor));

}


void bind_tensor(py::module_ &m) {

    py::class_<Tensor>(m, "Tensor")
        .def(py::init<>())

        // Static factory methods
        .def_static("zeros", [](py::sequence shape,
                                std::optional<std::string> device = std::nullopt,
                                std::optional<std::string> dtype = std::nullopt,
                                bool requires_grad = false) {
            auto s = py_utils::to_shape(shape);
            auto opts = py_utils::parse_opts(device, dtype, requires_grad);
            return Tensor::zeros(s, opts);
        }, py::arg("shape"), py::arg("device") = std::nullopt,
           py::arg("dtype") = std::nullopt, py::arg("requires_grad") = false)

        .def_static("ones", [](py::sequence shape,
                               std::optional<std::string> device = std::nullopt,
                               std::optional<std::string> dtype = std::nullopt,
                               bool requires_grad = false) {
            auto s = py_utils::to_shape(shape);
            auto opts = py_utils::parse_opts(device, dtype, requires_grad);
            return Tensor::ones(s, opts);
        }, py::arg("shape"), py::arg("device") = std::nullopt,
           py::arg("dtype") = std::nullopt, py::arg("requires_grad") = false)

        .def_static("randn", [](py::sequence shape,
                                std::optional<std::string> device = std::nullopt,
                                std::optional<std::string> dtype = std::nullopt,
                                bool requires_grad = false) {
            auto s = py_utils::to_shape(shape);
            auto opts = py_utils::parse_opts(device, dtype, requires_grad);
            return Tensor::randn(s, opts);
        }, py::arg("shape"), py::arg("device") = std::nullopt,
           py::arg("dtype") = std::nullopt, py::arg("requires_grad") = false)

        // Methods
        .def("is_cpu", &Tensor::is_cpu)
        .def("is_cuda", &Tensor::is_cuda)
        .def("__repr__", [](const Tensor &t) {
            std::ostringstream os;
            t.display(os, 4);
            return os.str();
        })

        // Optional — nice metadata methods
        .def_property_readonly("shape", [](const OwnTensor::Tensor &t) {
            std::vector<int64_t> out;
            for (auto d : t.shape().dims)
                out.push_back(d);
            return out;
        })
        .def_property_readonly("dtype", &OwnTensor::Tensor::dtype)
        .def_property_readonly("device", &OwnTensor::Tensor::device)
        .def_property_readonly("requires_grad", &OwnTensor::Tensor::requires_grad);
}


void bind_kernels(py::module_ &m) {
    using namespace ag::kernels;

    py::module m_kernels = m.def_submodule("kernels", "ag backend kernels");

    // CPU kernels
    py::module m_cpu = m_kernels.def_submodule("cpu", "CPU kernels");
    m_cpu.def("relu", [](const float* x, float* y, int64_t n) {
        cpu().relu(x, y, n);
    });
    m_cpu.def("relu_bwd", [](const float* x, const float* dy, float* dx, int64_t n) {
        cpu().relu_bwd(x, dy, dx, n);
    });
    m_cpu.def("matmul", [](const float* A, const float* B, float* C, int M, int K, int N) {
        cpu().matmul(A, B, C, M, K, N);
    });

    // CUDA kernels
    py::module m_cuda = m_kernels.def_submodule("cuda", "CUDA kernels");
    m_cuda.def("relu", [](const float* x, float* y, int64_t n, void* stream=nullptr) {
        ag::kernels::cuda().relu(x, y, n, (ag_cuda_stream_t)stream);
    });
    m_cuda.def("vjp_relu", [](float* gX, const float* gy, const float* X, int64_t n, void* stream=nullptr) {
        ag::kernels::cuda().vjp_relu(gX, gy, X, n, (ag_cuda_stream_t)stream);
    });
    m_cuda.def("add", [](const float* a, const float* b, float* c, int64_t n, void* stream=nullptr) {
        ag::kernels::cuda().add(a, b, c, n, (ag_cuda_stream_t)stream);
    });
    m_cuda.def("sub", [](const float* a, const float* b, float* c, int64_t n, void* stream=nullptr) {
        ag::kernels::cuda().sub(a, b, c, n, (ag_cuda_stream_t)stream);
    });
    m_cuda.def("matmul", [](const float* A, const float* B, float* C, int M, int K, int N, void* stream=nullptr) {
        ag::kernels::cuda().matmul(A, B, C, M, K, N, (ag_cuda_stream_t)stream);
    });
       m_kernels.def("load_cpu_plugin", &ag::kernels::load_cpu_plugin,
                  py::arg("path"), DOC(ag, kernels, load_cpu_plugin));
    m_kernels.def("load_cuda_plugin", &ag::kernels::load_cuda_plugin,
                  py::arg("path"), DOC(ag, kernels, load_cuda_plugin));
}



void bind_autodiff(py::module_ &m) {
    using namespace ag;

m.def("forward",  &forward,  py::arg("root"),  py::call_guard<py::gil_scoped_release>());
m.def("backward", &backward, py::arg("root"), py::arg("grad_seed") = nullptr,
      py::call_guard<py::gil_scoped_release>());
m.def("zero_grad",&zero_grad,py::arg("root"),  py::call_guard<py::gil_scoped_release>());
m.def("zero_val", &zero_val, py::arg("root"),  py::call_guard<py::gil_scoped_release>());
m.def("jvp",      &jvp,      py::arg("root"),  py::arg("seed"),
      py::call_guard<py::gil_scoped_release>());



}


void bind_debug(py::module_ &m) {
    using namespace ag::debug;

    py::module_ dbg = m.def_submodule("debug", "Debug utilities for autograd internals");

    dbg.def("enable_tracing",        &enable_tracing,        py::arg("on") = true, DOC(ag, debug, enable_tracing));
    dbg.def("print_tensor",          &print_tensor,          DOC(ag, debug, print_tensor));
    dbg.def("print_value",           &print_value,           DOC(ag, debug, print_value));
    dbg.def("print_grad",            &print_grad,            DOC(ag, debug, print_grad));
    dbg.def("print_all_values",      &print_all_values,      DOC(ag, debug, print_all_values));
    dbg.def("print_all_grads",       &print_all_grads,       DOC(ag, debug, print_all_grads));
    dbg.def("dump_dot",              &dump_dot,              DOC(ag, debug, dump_dot));
    dbg.def("enable_grad_tracing",   &enable_grad_tracing,   py::arg("on") = true, DOC(ag, debug, enable_grad_tracing));
    dbg.def("on_backprop_step",      &on_backprop_step,      DOC(ag, debug, on_backprop_step));
    dbg.def("dump_vjp_dot",          &dump_vjp_dot,          DOC(ag, debug, dump_vjp_dot));
    dbg.def("enable_jvp_tracing",    &enable_jvp_tracing,    py::arg("on") = true, DOC(ag, debug, enable_jvp_tracing));
    dbg.def("on_jvp_step",           &on_jvp_step,           DOC(ag, debug, on_jvp_step));
    dbg.def("dump_jvp_dot",          &dump_jvp_dot,          DOC(ag, debug, dump_jvp_dot));
}



PYBIND11_MODULE(cgadimpl, m) {
    m.doc() = "cgadimpl Python bindings for ag::Node, Value, and ops";
    bind_ops(m);
    bind_value(m);
    bind_tensor(m);
    bind_kernels(m);
    bind_debug(m);
    bind_autodiff(m);
    bind_optim(m);
}