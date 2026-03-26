#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>

// Include tensor headers
#include "TensorLib.h"

namespace py = pybind11;
using namespace OwnTensor;

void bind_tensor(py::module &m) {
    // Dtype enum
    py::enum_<Dtype>(m, "Dtype")
        .value("Float32", Dtype::Float32)
        .value("Float64", Dtype::Float64)
        .value("Int32", Dtype::Int32)
        .value("Int64", Dtype::Int64)
        .export_values();

    // Device enum
    py::enum_<Device>(m, "Device")
        .value("CPU", Device::CPU)
        .value("CUDA", Device::CUDA)
        .export_values();

    // TensorOptions class for configurable tensor creation
    py::class_<TensorOptions>(m, "TensorOptions")
        .def(py::init<>())
        .def("dtype", static_cast<TensorOptions& (TensorOptions::*)(Dtype)>(&TensorOptions::dtype), 
             "Set tensor dtype", py::return_value_policy::reference_internal)
        .def("device", static_cast<TensorOptions& (TensorOptions::*)(Device)>(&TensorOptions::device), 
             "Set tensor device (CPU/CUDA)", py::return_value_policy::reference_internal)
        .def("requires_grad", static_cast<TensorOptions& (TensorOptions::*)(bool)>(&TensorOptions::requires_grad), 
             "Enable/disable gradient tracking", py::return_value_policy::reference_internal);

    // Tensor class - main API
    py::class_<Tensor, std::shared_ptr<Tensor>>(m, "Tensor")
        .def(py::init<>(), "Create empty tensor")
        
        // Properties
        .def("shape", &Tensor::shape, "Get tensor shape")
        .def("dtype", &Tensor::dtype, "Get tensor dtype")
        .def("device", &Tensor::device, "Get tensor device")
        .def("numel", &Tensor::numel, "Get total number of elements")
        .def("nbytes", &Tensor::nbytes, "Get size in bytes")
        .def("requires_grad", static_cast<bool (Tensor::*)() const>(&Tensor::requires_grad), 
             "Check if gradient tracking is enabled")
        .def("is_contiguous", &Tensor::is_contiguous, "Check if tensor is contiguous in memory")
        
        // Data access
        .def("grad", static_cast<std::shared_ptr<Tensor> (Tensor::*)() const>(&Tensor::grad), 
             "Get gradient tensor")
        .def("set_grad", &Tensor::set_grad, "Set gradient tensor")
        .def("zero_grad", &Tensor::zero_grad, "Clear gradient data")
        
        // View operations
        .def("reshape", &Tensor::reshape, "Reshape tensor to new shape")
        .def("view", &Tensor::view, "Create view with new shape")
        .def("transpose", &Tensor::transpose, "Transpose matrix (swap last two dimensions)")
        .def("t", &Tensor::t, "2D transpose shorthand")
        .def("flatten", &Tensor::flatten, "Flatten to 1D")
        .def("contiguous", &Tensor::contiguous, "Return contiguous copy if needed")
        
        // Device management
        .def("to", static_cast<std::shared_ptr<Tensor> (Tensor::*)(Device)>(&Tensor::to), 
             "Move tensor to device")
        .def("to_cpu", &Tensor::to_cpu, "Move tensor to CPU")
        .def("to_cuda", &Tensor::to_cuda, "Move tensor to CUDA device")
        
        // Operations
        .def("fill", &Tensor::fill, "Fill tensor with scalar value")
        .def("mean", &Tensor::mean, "Compute mean of all elements")
        .def("sum", &Tensor::sum, "Sum all elements")
        .def("max", &Tensor::max, "Get maximum element")
        .def("min", &Tensor::min, "Get minimum element")
        
        // Operators
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(py::self * py::self)
        .def(py::self / py::self)
        .def(py::self += py::self)
        .def(py::self -= py::self)
        .def(py::self *= py::self)
        .def(py::self /= py::self)
        
        // Scalar operators
        .def(py::self + float())
        .def(py::self - float())
        .def(py::self * float())
        .def(py::self / float())
        
        // String representation
        .def("__repr__", [](const Tensor &t) {
            return "<Tensor shape=" + std::to_string(t.shape()[0]) + ">";
        });

    // Tensor factory functions
    m.def("zeros", [](const std::vector<int64_t> &shape, const TensorOptions &opts = TensorOptions()) {
        return Tensor::zeros(Shape(shape), opts);
    }, "Create tensor filled with zeros", py::arg("shape"), py::arg("options") = TensorOptions());

    m.def("ones", [](const std::vector<int64_t> &shape, const TensorOptions &opts = TensorOptions()) {
        return Tensor::ones(Shape(shape), opts);
    }, "Create tensor filled with ones", py::arg("shape"), py::arg("options") = TensorOptions());

    m.def("rand", [](const std::vector<int64_t> &shape, const TensorOptions &opts = TensorOptions()) {
        return Tensor::rand(Shape(shape), opts);
    }, "Create tensor with random values [0, 1)", py::arg("shape"), py::arg("options") = TensorOptions());

    m.def("randn", [](const std::vector<int64_t> &shape, const TensorOptions &opts = TensorOptions()) {
        return Tensor::randn(Shape(shape), opts);
    }, "Create tensor with normal distribution samples", py::arg("shape"), py::arg("options") = TensorOptions());

    m.def("full", [](const std::vector<int64_t> &shape, float value, const TensorOptions &opts = TensorOptions()) {
        return Tensor::full(Shape(shape), value, opts);
    }, "Create tensor filled with scalar value", py::arg("shape"), py::arg("value"), 
    py::arg("options") = TensorOptions());

    m.def("from_numpy", [](py::array_t<float> arr) {
        // Convert numpy array to tensor (zero-copy if possible)
        auto buf = arr.request();
        std::vector<int64_t> shape;
        for (auto s : arr.shape()) {
            shape.push_back(s);
        }
        return Tensor::from_buffer(static_cast<float*>(buf.ptr), Shape(shape));
    }, "Create tensor from numpy array");

    m.def("to_numpy", [](const std::shared_ptr<Tensor> &t) {
        // Convert tensor to numpy array
        auto shape = t->shape();
        std::vector<size_t> np_shape(shape.begin(), shape.end());
        return py::array_t<float>(np_shape, t->data<float>());
    }, "Convert tensor to numpy array");
}
