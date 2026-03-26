#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/buffer_protocol.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// Forward declarations for binding functions
void bind_tensor(py::module &m);
void bind_autodiff(py::module &m);
void bind_layers(py::module &m);
void bind_optimizers(py::module &m);

PYBIND11_MODULE(cgad, m) {
    m.doc() = R"pbdoc(
        CGADImpl - High-Performance C++ Autodifferentiation Framework
        
        A PyTorch-like framework combining:
        - Efficient tensor operations (OwnTensor)
        - Automatic differentiation with computation graphs
        - Neural network layers and modules
        - Optimizers and training utilities
        - GPU support (CUDA kernels)
        
        Example:
            import cgad as cg
            
            # Create tensors
            x = cg.tensor.randn([2, 3])
            w = cg.tensor.randn([3, 4], requires_grad=True)
            
            # Forward pass
            y = cg.autodiff.matmul(x, w)
            loss = cg.autodiff.sum(y)
            
            # Backward pass
            cg.autodiff.backward(loss)
            
            # Gradient descent step
            cg.optim.sgd(w, learning_rate=0.01)
    )pbdoc";

    // Create submodules for organizing API
    auto tensor_mod = m.def_submodule("tensor", "Tensor creation and operations");
    auto autodiff_mod = m.def_submodule("autodiff", "Autodifferentiation and computation graph");
    auto layer_mod = m.def_submodule("layer", "Neural network layers");
    auto nn_mod = m.def_submodule("nn", "Neural network modules");
    auto optim_mod = m.def_submodule("optim", "Optimizers");

    // Bind each component
    bind_tensor(tensor_mod);
    bind_autodiff(autodiff_mod);
    bind_layers(layer_mod);
    bind_optimizers(optim_mod);

    // Module-level version info
    m.attr("__version__") = "0.1.0";
}
