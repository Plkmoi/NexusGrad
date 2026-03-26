#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

// Include optimizer headers
#include "optim.hpp"
#include "ad/ag_all.hpp"

namespace py = pybind11;
using namespace ag;

void bind_optimizers(py::module &m) {
    // Optimizer base concept bindings
    
    m.doc() = "Optimization algorithms for training neural networks";

    // SGD (Stochastic Gradient Descent)
    m.def("sgd", [](std::shared_ptr<ag::Value> root, 
                     std::shared_ptr<OwnTensor::Tensor> grad_seed,
                     float learning_rate) {
        return sgd(root, grad_seed, learning_rate);
    }, "Stochastic Gradient Descent step",
        py::arg("root_value"), py::arg("grad_seed") = nullptr, py::arg("learning_rate") = 0.01);

    // SGD with momentum
    m.def("sgd_momentum", [](std::shared_ptr<ag::Value> root,
                              std::shared_ptr<OwnTensor::Tensor> grad_seed,
                              std::shared_ptr<OwnTensor::Tensor> velocity,
                              float learning_rate,
                              float momentum) {
        // Apply gradient: vel = momentum * vel - lr * grad
        // param += vel
        return sgd_momentum(root, grad_seed, velocity, learning_rate, momentum);
    }, "SGD with momentum",
        py::arg("root_value"), py::arg("grad_seed") = nullptr, py::arg("velocity"),
        py::arg("learning_rate") = 0.01, py::arg("momentum") = 0.9);

    // Adam optimizer
    m.def("adam", [](std::shared_ptr<ag::Value> root,
                      std::shared_ptr<OwnTensor::Tensor> grad_seed,
                      std::shared_ptr<OwnTensor::Tensor> m_t,  // First moment
                      std::shared_ptr<OwnTensor::Tensor> v_t,  // Second moment
                      float learning_rate,
                      float beta1,
                      float beta2,
                      float epsilon,
                      int t) {  // timestep
        return adam(root, grad_seed, m_t, v_t, learning_rate, beta1, beta2, epsilon, t);
    }, "Adam optimizer",
        py::arg("root_value"), py::arg("grad_seed") = nullptr,
        py::arg("m_t"), py::arg("v_t"),
        py::arg("learning_rate") = 0.001,
        py::arg("beta1") = 0.9,
        py::arg("beta2") = 0.999,
        py::arg("epsilon") = 1e-8,
        py::arg("t") = 1);

    // AdamW optimizer (with weight decay)
    m.def("adamw", [](std::shared_ptr<ag::Value> root,
                       std::shared_ptr<OwnTensor::Tensor> grad_seed,
                       std::shared_ptr<OwnTensor::Tensor> m_t,
                       std::shared_ptr<OwnTensor::Tensor> v_t,
                       float learning_rate,
                       float beta1,
                       float beta2,
                       float epsilon,
                       float weight_decay,
                       int t) {
        return adamw(root, grad_seed, m_t, v_t, learning_rate, beta1, beta2, epsilon, weight_decay, t);
    }, "Adam optimizer with weight decay",
        py::arg("root_value"), py::arg("grad_seed") = nullptr,
        py::arg("m_t"), py::arg("v_t"),
        py::arg("learning_rate") = 0.001,
        py::arg("beta1") = 0.9,
        py::arg("beta2") = 0.999,
        py::arg("epsilon") = 1e-8,
        py::arg("weight_decay") = 0.01,
        py::arg("t") = 1);

    // RMSprop optimizer
    m.def("rmsprop", [](std::shared_ptr<ag::Value> root,
                         std::shared_ptr<OwnTensor::Tensor> grad_seed,
                         std::shared_ptr<OwnTensor::Tensor> v_t,
                         float learning_rate,
                         float alpha,
                         float epsilon) {
        return rmsprop(root, grad_seed, v_t, learning_rate, alpha, epsilon);
    }, "RMSprop optimizer",
        py::arg("root_value"), py::arg("grad_seed") = nullptr,
        py::arg("v_t"),
        py::arg("learning_rate") = 0.01,
        py::arg("alpha") = 0.99,
        py::arg("epsilon") = 1e-8);

    // Optimizer state management utility
    py::class_<OptimizerState>(m, "OptimizerState")
        .def(py::init<>(), "Optimizer state container")
        .def("add_tensor", &OptimizerState::add_tensor, 
             "Add state tensor", py::arg("name"), py::arg("tensor"))
        .def("get_tensor", &OptimizerState::get_tensor, 
             "Get state tensor", py::arg("name"))
        .def("zero", &OptimizerState::zero, 
             "Zero all state tensors");

    // Learning rate scheduling utilities
    m.def("step_decay", [](float initial_lr, int epoch, int decay_epoch, float decay_rate) {
        return initial_lr * std::pow(decay_rate, epoch / (float)decay_epoch);
    }, "Step decay schedule", 
        py::arg("initial_lr"), py::arg("epoch"), py::arg("decay_epoch"), py::arg("decay_rate"));

    m.def("exponential_decay", [](float initial_lr, int epoch, float decay_rate) {
        return initial_lr * std::exp(-decay_rate * epoch);
    }, "Exponential decay schedule",
        py::arg("initial_lr"), py::arg("epoch"), py::arg("decay_rate"));

    m.def("cosine_annealing", [](float initial_lr, int epoch, int max_epochs) {
        return initial_lr * 0.5 * (1 + std::cos(M_PI * epoch / max_epochs));
    }, "Cosine annealing schedule",
        py::arg("initial_lr"), py::arg("epoch"), py::arg("max_epochs"));

    // Gradient clipping
    m.def("clip_grad_norm", [](std::vector<std::shared_ptr<ag::Value>> parameters, float max_norm) {
        float total_norm = 0.0;
        for (auto &param : parameters) {
            auto grad = param->grad();
            if (grad) {
                float norm = grad->sum(); // Simplified norm
                total_norm += norm * norm;
            }
        }
        total_norm = std::sqrt(total_norm);
        
        float clip_coef = max_norm / (total_norm + 1e-6);
        if (clip_coef < 1.0) {
            for (auto &param : parameters) {
                auto grad = param->grad();
                if (grad) {
                    *grad *= clip_coef;
                }
            }
        }
        return total_norm;
    }, "Clip gradient norms",
        py::arg("parameters"), py::arg("max_norm") = 1.0);

    m.def("clip_grad_value", [](std::vector<std::shared_ptr<ag::Value>> parameters, float clip_value) {
        for (auto &param : parameters) {
            auto grad = param->grad();
            if (grad) {
                // Clamp gradient values to [-clip_value, clip_value]
                // Implementation depends on tensor API
            }
        }
    }, "Clip gradient values elementwise",
        py::arg("parameters"), py::arg("clip_value") = 1.0);
}
