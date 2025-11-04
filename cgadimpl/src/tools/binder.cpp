#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "ad/nodeops.hpp"
#include "ad/runtime.hpp"
#include "ad/kernels_api.hpp"
#include "ad/ag_all.hpp"
#include "tensor.hpp"

#include "/home/blubridge-034/Downloads/Newf/cgadimpl/binder_out/docstrings/tensor_docs.hpp"
#include "/home/blubridge-034/Downloads/Newf/cgadimpl/binder_out/docstrings/nodeops_docs.hpp"
#include "/home/blubridge-034/Downloads/Newf/cgadimpl/binder_out/docstrings/graph_docs.hpp"
#include "/home/blubridge-034/Downloads/Newf/cgadimpl/binder_out/docstrings/schema_docs.hpp"

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

// ------------------ NodeOps (free functions) ------------------
void bind_nodeops(py::module_ &m) {
    m.def("add_nodeops", &ag::detail::add_nodeops,
          py::arg("a"), py::arg("b"),
          DOC(ag, detail, add_nodeops));
    m.def("sub_nodeops", &ag::detail::sub_nodeops,
          py::arg("a"), py::arg("b"),
          DOC(ag, detail, add_nodeops));
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
    bind_nodeops(m);
    bind_runtime(m);
}
