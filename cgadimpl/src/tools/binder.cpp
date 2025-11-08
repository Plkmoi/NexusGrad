#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "ad/nodeops.hpp"
#include "ad/runtime.hpp"
#include "ad/kernels_api.hpp"
#include "ad/ag_all.hpp"
#include "tensor.hpp"
#include "optim.hpp"

#include "/home/blubridge-034/Downloads/Newf/cgadimpl/binder_out/docstrings/tensor_docs.hpp"
#include "/home/blubridge-034/Downloads/Newf/cgadimpl/binder_out/docstrings/nodeops_docs.hpp"
#include "/home/blubridge-034/Downloads/Newf/cgadimpl/binder_out/docstrings/graph_docs.hpp"
#include "/home/blubridge-034/Downloads/Newf/cgadimpl/binder_out/docstrings/schema_docs.hpp"
#include "/home/blubridge-034/Downloads/Newf/cgadimpl/binder_out/docstrings/kernels_api_docs.hpp"
#include "/home/blubridge-034/Downloads/Newf/cgadimpl/binder_out/docstrings/autodiff_docs.hpp"
#include "/home/blubridge-034/Downloads/Newf/cgadimpl/binder_out/docstrings/ops_docs.hpp"
#include "/home/blubridge-034/Downloads/Newf/cgadimpl/binder_out/docstrings/optim_docs.hpp"

namespace py = pybind11;
using namespace ag;

void bind_enums(py::module_ &m) {
    py::enum_<Op>(m, "Op")
        .value("Leaf", Op::Leaf)
        .value("Add", Op::Add)
        .value("Mul", Op::Mul)
        .value("Sub", Op::Sub)
        .value("Div", Op::Div)
        // ... add all your ops
        ;
}


void bind_tensor(py::module_ &m) {
    py::class_<Tensor>(m, "Tensor", DOC(ag, Tensor))
    .def(py::init<>(), DOC(ag, Tensor, Tensor))
    .def_static("zeros", &Tensor::zeros, DOC(ag, Tensor, zeros))
    .def_static("ones",  &Tensor::ones,  DOC(ag, Tensor, ones))
    .def_static("zeros_like", &Tensor::zeros_like, DOC(ag, Tensor, zeros_like))
    .def_static("ones_like",  &Tensor::ones_like,  DOC(ag, Tensor, ones_like))
    .def_static("vals_like",  &Tensor::vals_like,  DOC(ag, Tensor, vals_like))
    .def_static("randn", &Tensor::randn, DOC(ag, Tensor, randn))
    .def_static("vales", &Tensor::vales, DOC(ag, Tensor, vales))
    .def("device", &Tensor::device, DOC(ag, Tensor, device))
    .def("to", &Tensor::to, py::arg("device"), DOC(ag, Tensor, to))
    .def("is_cpu", &Tensor::is_cpu, DOC(ag, Tensor, is_cpu))
    .def("is_cuda", &Tensor::is_cuda, DOC(ag, Tensor, is_cuda))
    .def("__repr__", [](const Tensor &t) {
    std::ostringstream os;
    os << t;
    return os.str();
});

}

void bind_device(py::module_ &m) {
    py::enum_<Device>(m, "Device")
        .value("CPU", Device::CPU)
        .value("CUDA", Device::CUDA)
        .export_values();  // optional, lets you do cgadimpl.CPU
}

void bind_graph(py::module_ &m) {
    using namespace ag;

    m.def("make_tensor", &make_tensor,
          py::arg("tensor"), py::arg("name") = "", py::arg("requires_grad") = false,
          DOC(ag, make_tensor));

    m.def("make_tensornode", &make_tensornode,
          py::arg("tensor"), py::arg("name") = "", py::arg("requires_grad") = false,
          DOC(ag, make_tensornode));

    m.def("constant", &constant,
          py::arg("tensor"), py::arg("name") = "",
          DOC(ag, constant));

    m.def("param", &param,
          py::arg("tensor"), py::arg("name") = "",
          DOC(ag, param));

    m.def("topo_from", &topo_from,
          py::arg("root"),
          DOC(ag, topo_from));
}

void bind_ops(py::module_ &m) {
    py::module m_ops = m.def_submodule("ops", "ag backend operations");

    // Arithmetic Operations
    m_ops.def("add", &ag::add, py::arg("a"), py::arg("b"), DOC(ag, add));
    m_ops.def("sub", &ag::sub, py::arg("a"), py::arg("b"), DOC(ag, sub));
    m_ops.def("mul", &ag::mul, py::arg("a"), py::arg("b"), DOC(ag, mul));
    m_ops.def("div", &ag::div, py::arg("a"), py::arg("b"), DOC(ag, div));

    // Unary Operations
    m_ops.def("relu", &ag::relu, py::arg("x"), DOC(ag, relu));
    m_ops.def("gelu", &ag::gelu, py::arg("x"), DOC(ag, gelu));
    m_ops.def("leaky_relu", &ag::leaky_relu, py::arg("x"), py::arg("alpha") = 0.01f, DOC(ag, leaky_relu));
    m_ops.def("silu", &ag::silu, py::arg("x"), DOC(ag, silu));
    m_ops.def("sigmoid", &ag::sigmoid, py::arg("x"), DOC(ag, sigmoid));
    m_ops.def("softplus", &ag::softplus, py::arg("x"), DOC(ag, softplus));
    m_ops.def("tanh", &ag::tanh, py::arg("x"), DOC(ag, tanh));

    // More operations
    m_ops.def("sum", &ag::sum, py::arg("x"), DOC(ag, sum));
    m_ops.def("exp", &ag::exp, py::arg("x"), DOC(ag, exp));
    m_ops.def("log", &ag::log, py::arg("x"), DOC(ag, log));
    m_ops.def("sign", &ag::sign, py::arg("a"), py::arg("b"), DOC(ag, sign));

    // Advanced ops
    m_ops.def("flomul", &ag::flomul, py::arg("a"), py::arg("b"), DOC(ag, flomul));
    m_ops.def("floadd", &ag::floadd, py::arg("a"), py::arg("b"), DOC(ag, floadd));
    m_ops.def("flodiv", &ag::flodiv, py::arg("a"), py::arg("b"), DOC(ag, flodiv));
    m_ops.def("fmab", &ag::fmab, py::arg("a"), py::arg("b"), py::arg("c"), DOC(ag, fmab));
    m_ops.def("linear", &ag::linear, py::arg("a"), py::arg("b"), py::arg("c"), DOC(ag, linear));
    m_ops.def("mse_loss", &ag::mse_loss, py::arg("pred"), py::arg("target"), DOC(ag, mse_loss));
    m_ops.def("mae_loss", &ag::mae_loss, py::arg("pred"), py::arg("target"), DOC(ag, mae_loss));

    // Complex operations
    m_ops.def("attention", &ag::attention, py::arg("a"), py::arg("b"), py::arg("c"), py::arg("d"), py::arg("Bw"), py::arg("nh"), DOC(ag, attention));
    m_ops.def("cross_entropy_with_logits", &ag::cross_entropy_with_logits, py::arg("logits"), py::arg("onehot"), DOC(ag, cross_entropy_with_logits));
    m_ops.def("kldivergence", &ag::kldivergence, py::arg("logits"), py::arg("onehot"), DOC(ag, kldivergence));

    // Row operations
    m_ops.def("rowsum", &ag::rowsum, py::arg("x"), DOC(ag, rowsum));
    m_ops.def("rowmax", &ag::rowmax, py::arg("x"), DOC(ag, rowmax));
    m_ops.def("mean_all", &ag::mean_all, py::arg("x"), DOC(ag, mean_all));
    m_ops.def("softmax_row", &ag::softmax_row, py::arg("z"), DOC(ag, softmax_row));
    m_ops.def("logsumexp_row", &ag::logsumexp_row, py::arg("z"), DOC(ag, logsumexp_row));

    // Special functions
    m_ops.def("rms", &ag::rms, py::arg("x"), DOC(ag, rms));
    m_ops.def("realrms", &ag::realrms, py::arg("x"), py::arg("g"), DOC(ag, realrms));
    m_ops.def("swiglu", &ag::swiglu, py::arg("x"), py::arg("a"), py::arg("b"), py::arg("c"), py::arg("d"), DOC(ag, swiglu));
    m_ops.def("parcon", &ag::parcon, py::arg("x"), DOC(ag, parcon));
    m_ops.def("gcu", &ag::gcu, py::arg("x"), DOC(ag, gcu));

    // Other miscellaneous ops
    m_ops.def("laynor", &ag::laynor, py::arg("x"), DOC(ag, laynor));
    m_ops.def("alibiatt", &ag::alibiatt, py::arg("a"), py::arg("b"), py::arg("c"), py::arg("d"), py::arg("m"), DOC(ag, alibiatt));

    // Transpose (yay)
    m_ops.def("transpose", &ag::transpose, py::arg("x"), DOC(ag, transpose));
}




// ------------------ Node ------------------
void bind_node(py::module_ &m) {
    py::class_<Node, std::shared_ptr<Node>>(m, "Node", DOC(ag, Node))
        .def(py::init<>(), DOC(ag, Node, Node))
        .def(py::init<const Tensor&, bool, Op, const char*>(),
             py::arg("value"), py::arg("requires_grad"),
             py::arg("op") = Op::Leaf, py::arg("debug_name") = "",
             DOC(ag, Node, Node))
        .def_readwrite("debug_name", &Node::debug_name)
        .def_readwrite("grad", &Node::grad)
        .def_readwrite("op", &Node::op)
        .def_readwrite("value", &Node::value)
        .def_readwrite("inputs", &Node::inputs)
        .def_readwrite("requires_grad", &Node::requires_grad)
        .def("__repr__", [](const Node &self) {
            return std::string("<cgadimpl.Node '") + self.debug_name + "'>";
        });
}

// ------------------ Value ------------------
void bind_value(py::module_ &m) {
    py::class_<Value>(m, "Value", DOC(ag, Value))
        .def(py::init<>())
        .def(py::init<std::shared_ptr<Node>>(), py::arg("node"))
        .def("val", &Value::val)
        .def("grad", &Value::grad)
        .def("shape", &Value::shape)
        .def("__repr__", [](const Value &self) {
            auto s = self.shape();
            return "<cgadimpl.Value shape=(" + std::to_string(s.first) + "," +
                   std::to_string(s.second) + ")>";
        });
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
        cuda().relu(x, y, n, (ag_cuda_stream_t)stream);
    });
    m_cuda.def("vjp_relu", [](float* gX, const float* gy, const float* X, int64_t n, void* stream=nullptr) {
        cuda().vjp_relu(gX, gy, X, n, (ag_cuda_stream_t)stream);
    });
    m_cuda.def("add", [](const float* a, const float* b, float* c, int64_t n, void* stream=nullptr) {
        cuda().add(a, b, c, n, (ag_cuda_stream_t)stream);
    });
    m_cuda.def("sub", [](const float* a, const float* b, float* c, int64_t n, void* stream=nullptr) {
        cuda().sub(a, b, c, n, (ag_cuda_stream_t)stream);
    });
    m_cuda.def("matmul", [](const float* A, const float* B, float* C, int M, int K, int N, void* stream=nullptr) {
        cuda().matmul(A, B, C, M, K, N, (ag_cuda_stream_t)stream);
    });
       m_kernels.def("load_cpu_plugin", &ag::kernels::load_cpu_plugin,
                  py::arg("path"), DOC(ag, kernels, load_cpu_plugin));
    m_kernels.def("load_cuda_plugin", &ag::kernels::load_cuda_plugin,
                  py::arg("path"), DOC(ag, kernels, load_cuda_plugin));
}



// ------------------ NodeOps (free functions) ------------------
void bind_nodeops(py::module_ &m) {
    m.def("add_nodeops", &ag::detail::add_nodeops,
          py::arg("a"), py::arg("b"),
          DOC(ag, detail, add_nodeops));
    m.def("sub_nodeops", &ag::detail::sub_nodeops,
          py::arg("a"), py::arg("b"),
          DOC(ag, detail, add_nodeops));
}

void bind_autodiff(py::module_ &m) {
    using namespace ag;

    m.def("zero_grad", &zero_grad, py::arg("root"), DOC(ag, zero_grad));
    m.def("zerono_grad", &zerono_grad, py::arg("root"), DOC(ag, zerono_grad));
    m.def("backward_node", &backward_node, py::arg("root"), py::arg("grad_seed") = nullptr, DOC(ag, backward_node));
    m.def("backward", &backward, py::arg("root"), py::arg("grad_seed") = nullptr, DOC(ag, backward));

    m.def("valsend", &valsend, py::arg("root"), DOC(ag, valsend));
    m.def("grasend", &grasend, py::arg("root"), DOC(ag, grasend));
    m.def("unisend", &unisend, py::arg("root"), DOC(ag, unisend));
    m.def("newunisend", &newunisend, py::arg("root"), DOC(ag, newunisend));

    m.def("jvp", &jvp, py::arg("root"), py::arg("seed"), DOC(ag, jvp));

    m.def("save_safetensors", &save_safetensors,
          py::arg("tensors"), py::arg("filename"), DOC(ag, save_safetensors));
    m.def("save_all_values_and_grads", &save_all_values_and_grads,
          py::arg("root"), DOC(ag, save_all_values_and_grads));
}


void bind_optim(py::module_ &m) {
    using namespace ag;

    py::module m_optim = m.def_submodule("optim", "ag backend optimizers");

    // Binding the SGD function (Stochastic Gradient Descent) with an optional grad_seed
    m_optim.def("SGD", &SGD, 
                py::arg("root"), 
                py::arg("learning_rate") = 0.1,   // default learning rate
                py::arg("grad_seed") = nullptr,    // optional grad_seed argument
                DOC(ag, SGD));  // Make sure you have a proper docstring for SGD
}

// ------------------ CUDA Runtime ------------------
void bind_runtime(py::module_ &m) {
#ifdef __CUDACC__
    m.def("cuda_available", []() { return true; });
#else
    m.def("cuda_available", []() { return false; });
#endif
}

// ------------------ Module Entry ------------------
PYBIND11_MODULE(cgadimpl, m) {
    m.doc() = "cgadimpl Python bindings for ag::Node, Value, and ops";
    bind_enums(m);
    bind_device(m);
    bind_tensor(m);
    bind_node(m);
    bind_value(m);
    bind_graph(m); 
    bind_nodeops(m);
    bind_runtime(m);
    bind_kernels(m);
    bind_autodiff(m);
    bind_ops(m);
    bind_optim(m);
}