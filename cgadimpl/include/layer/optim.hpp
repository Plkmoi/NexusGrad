#pragma once
#include "layer/affine.hpp"

#include "ad/graph.hpp"
#include "ad/ops.hpp"
#include <vector>

namespace ag::layer {

// Base Optimizer Class
class Optimizer {
public:
    // Constructor takes the list of parameters to optimize
    explicit Optimizer(std::vector<Value> params) 
        : params_(std::move(params)) {}

    virtual ~Optimizer() = default;

    // The main update function to be implemented by subclasses (SGD, Adam, etc.)
    virtual void step() = 0;

    // Helper to zero out gradients for all parameters
    void zero_grad() {
        for (auto& p : params_) {
            if (p.node && p.node->requires_grad()) {
                // Assuming OwnTensor has a zeros_like or similar functionality
                // Or simply assigning a new zero tensor
                // p.grad() returns a reference to the grad tensor
                // We overwrite it with zeros.
                p.node->grad = OwnTensor::Tensor::zeros(
                    p.val().shape(), 
                    ag::options(p.val()) // Helper to get options from tensor
                );
            }
        }
    }
    void checklayer();

protected:
    std::vector<Value> params_;
};

// Stochastic Gradient Descent
class SGD : public Optimizer {
public:
    SGD(std::vector<Value> params, float learning_rate, float momentum = 0.0f)
        : Optimizer(std::move(params)), lr_(learning_rate), momentum_(momentum) 
    {
        // specific initialization for momentum buffers if needed
        if (momentum_ > 0.0f) {
            for (const auto& p : params_) {
                // Create a velocity buffer for each parameter on the same device
                velocities_.push_back(OwnTensor::Tensor::zeros(
                    p.val().shape(), 
                    ag::options(p.val())
                ));
            }
        }
    }

    void step() override {
        for (size_t i = 0; i < params_.size(); ++i) {
            auto& p = params_[i];

            // Skip parameters that don't require gradients or have no node
            if (!p.node || !p.node->requires_grad()) continue;

            // Get Tensor references
            Tensor& data = p.val();
            const Tensor& grad = p.grad();

            if (momentum_ > 0.0f) {
                // Momentum update
                // v = momentum * v + grad
                // w = w - lr * v
                Tensor& v = velocities_[i];
                v = (v * momentum_) + grad; 
                data = data - (v * lr_);
            } else {
                // Standard SGD update
                // w = w - lr * grad
                data = data - (grad * lr_);
            }
        }
    }
    

private:
    float lr_;
    float momentum_;
    std::vector<Tensor> velocities_; // Stores velocity state for momentum
};



} // namespace ag::nn