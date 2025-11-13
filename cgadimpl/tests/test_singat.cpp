#include "ad/ag_all.hpp"   // Main umbrella header for the framework
#include <iostream>
#include <vector>
#include <iomanip>
#include <optim.hpp>

int main() {
    using namespace ag;
    using namespace OwnTensor;

    std::cout << "========================================\n";
    std::cout << "--- Tiny LLaMA-Style Transformer Block ---\n";
    std::cout << "========================================\n\n";

    // 1. --- Hyperparameters ---
    const int batch_size      = 16;
    const int d_model         = 128;  // model / residual width (like LLaMA d_model)
    const int output_features = 10;   // e.g. regression / tiny "vocab"
    const int num_layers      = 2;    // number of (Attn + MLP) block pairs
    const float learning_rate = 0.001f;
    const int epochs          = 50;

    // 2. --- Build a LLaMA-ish stack of layers ---
    //
    // Pattern per layer:
    //   x -> Residual( RMSNorm -> Attention(d_model -> d_model) )
    //     -> Residual( RMSNorm -> SWIGLU(d_model -> d_model) )
    //
    // After all layers:
    //     -> RMSNorm
    //     -> Linear(d_model -> output_features)
    //
    std::vector<ag::nn::Module*> layers;
    layers.reserve(num_layers * 2 + 2);

    for (int i = 0; i < num_layers; ++i) {
        // Attention sub-block: x + Attn(RMSNorm(x))
        layers.push_back(new ag::nn::ResidualBlock({
            new ag::nn::RMSNorm(),
            new ag::nn::Attention(d_model, d_model)
        }));

        // MLP / SWIGLU sub-block: x + SWIGLU(RMSNorm(x))
        layers.push_back(new ag::nn::ResidualBlock({
            new ag::nn::RMSNorm(),
            new ag::nn::SWIGLU(d_model, d_model)
        }));
    }

    // Final norm + head
    layers.push_back(new ag::nn::RMSNorm());
    layers.push_back(new ag::nn::Linear(d_model, output_features));

    ag::nn::Sequential model(layers);

    std::cout << "Model created with " << model.parameters().size()
              << " parameter tensors.\n\n";

    // 3. --- Random data (B, d_model) -> (B, output_features) ---
    Tensor x_tensor = Tensor::randn(
        Shape{{batch_size, d_model}},
        TensorOptions().with_req_grad(true)
    );
    Tensor y_tensor = Tensor::randn(
        Shape{{batch_size, output_features}},
        TensorOptions().with_req_grad(true)
    );

    Value X = make_tensor(x_tensor, "X_data");
    Value Y = make_tensor(y_tensor, "Y_target");

    // 4. --- Training loop ---
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Forward
        Value predictions = model(X);          // [B, output_features]
        Value loss        = mse_loss(predictions, Y);

        float loss_value = loss.val().data<float>()[0];
        std::cout << "Epoch " << std::setw(2) << epoch
                  << ", Loss: " << std::fixed << std::setprecision(6)
                  << loss_value << std::endl;

        // Backward
        model.zero_grad();
        backward(loss);

        // SGD step
        ag::SGD(loss, nullptr, learning_rate);
    }

    // 5. --- Cleanup (top-level modules) ---
    for (auto* layer : model.get_layers()) {
        delete layer;
    }

    std::cout << "\n✅ Tiny LLaMA-style training finished successfully.\n";
    return 0;
}
