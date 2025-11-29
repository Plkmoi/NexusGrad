#include "ad/ag_all.hpp"
#include <iostream>
#include <vector>
#include <iomanip>
#include <cassert>
#include <layer/optim.hpp>

// Use the official ReLU module from the framework
using ReLU = ag::layer::ReLU;

int main() {
    try {
        std::cout << "========================================\n";
        std::cout << "--- Starting End-to-End CPU Training ---\n";
        std::cout << "========================================\n\n";

        const int batch_size = 2;
        const int in_features = 4;
        const int hidden_features = 4;
        const int out_features = 2;
        const float learning_rate = 0.01f;

        ag::layer::Traverse model_cpu({
            new ag::layer::Linear(batch_size, hidden_features, in_features),
            new ReLU(),
            new ag::layer::Linear(batch_size, out_features, hidden_features)
        });
        std::cout << "CPU Model created successfully.\n";

        auto cpu_opts = OwnTensor::TensorOptions().with_device(Device::CPU);
        ag::Value input = ag::make_tensor(OwnTensor::Tensor::randn(OwnTensor::Shape{{batch_size, out_features, in_features}}, cpu_opts), "input");
        ag::Value labels = ag::make_tensor(OwnTensor::Tensor::zeros(OwnTensor::Shape{{batch_size, out_features, out_features}}, cpu_opts), "labels");
        
        // --- Initial Pass ---
        ag::Value output = model_cpu(input);
        ag::Value loss = ag::mse_loss(output, labels);
        ag::backward(loss);

        float initial_loss_val = loss.val().to_cpu().data<float>()[0];
        std::cout << "Initial Loss (CPU): " << initial_loss_val << std::endl;
        
        const auto& w1_grad_cpu = model_cpu.parameters()[0].grad();
        float grad_sum_cpu = OwnTensor::reduce_sum(OwnTensor::abs(w1_grad_cpu, nullptr)).to_cpu().data<float>()[0];
        
        assert(grad_sum_cpu > 0.0f && "FATAL: Gradients are zero after first backward pass!");
        std::cout << "PASS: Gradients were computed successfully on CPU.\n";

        // ag::layer::SGD newsgd(model_cpu.parameters(), learning_rate);

ag::debug::print_all_values(loss);
        // // --- SGD Step ---
        ag::opti.SGDm(loss, learning_rate, 0.9);
        ag::opti.epoch();
        std::cout << "Performed one SGD step.\n";
        ag::debug::print_all_values(loss);

        // --- Second Pass ---
        model_cpu.zero_grad();
        ag::Value new_output = model_cpu(input);
        ag::Value new_loss = ag::mse_loss(new_output, labels);
        float new_loss_val = new_loss.val().to_cpu().data<float>()[0];
        std::cout << "New Loss (CPU): " << new_loss_val << std::endl;

        assert(new_loss_val < initial_loss_val && "FATAL: Loss did not decrease after training step!");
        std::cout << "PASS: Loss decreased on CPU, indicating successful training.\n";

    } catch (const std::exception& e) {
        std::cerr << "\nERROR: An exception occurred during the test: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "\n✅ All end-to-end training tests passed successfully!\n";
    return 0;
}