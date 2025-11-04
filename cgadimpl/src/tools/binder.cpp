// =====================
// file: cgadimpl/src/bindings.cpp
// =====================
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "ad/ops.hpp"
#include "ad/graph.hpp"

namespace py = pybind11;

PYBIND11_MODULE(cgadimpl, m) {
    m.doc() = "Custom autodiff engine bindings";

    // Expose your Value class to Python
    py::class_<ag::Value, std::shared_ptr<ag::Value>>(m, "Value")
        .def(py::init<>())
        .def("__add__", &ag::add)
        .def("__sub__", &ag::sub)
        .def("__mul__", &ag::mul)
        .def("__truediv__", &ag::div)
        .def("relu", &ag::relu)
        .def("val", &ag::Value::val, py::return_value_policy::reference_internal)
        .def("grad", &ag::Value::grad, py::return_value_policy::reference_internal);

    // Expose some top-level helper functions
    m.def("make_tensor", &ag::make_tensor, "Create a tensor Value");
    m.def("inplace_checkpoint", &ag::inplace_checkpoint, "Mark an inplace checkpoint");
}
