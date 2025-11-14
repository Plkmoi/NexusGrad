#include "ad/ag_all.hpp" // Main umbrella header for the framework
#include <iostream>
#include <vector>
#include <iomanip>
#include <optim.hpp>

int main() {
    using namespace ag;
    using namespace OwnTensor;

    std::cout << "========================================\n";
    std::cout << "--- Starting End-to-End MLP Training ---\n";
    std::cout << "========================================\n\n";

    // 1. --- Define Hyperparameters ---
    const int batch_size = 16;
    const int input_features = 128;
    const int hidden_features = 64;
    const int output_features = 10;
    const float learning_rate = 0.01f;
    const int epochs = 15;

    // 2. --- Create the Model ---
    ag::layer::Traverse model({
            new ag::layer::ResidualBlock({
        new ag::layer::RMSNorm(),
        new ag::layer::Attention(input_features, input_features)
    }),

    new ag::layer::SWIGLU(input_features, input_features),
        new ag::layer::Linear(input_features, output_features)
    });

    std::cout << "Model created with " << model.parameters().size() << " parameter tensors.\n\n";

    // 3. --- Generate Random Data ---
    // ✅ FIX: input should match model input (batch_size, input_features)
    Tensor x_tensor = Tensor::randn(Shape{{batch_size, input_features}}, TensorOptions().with_req_grad(true));
    // ✅ FIX: target should match model output (batch_size, output_features)
    Tensor y_tensor = Tensor::randn(Shape{{batch_size, output_features}}, TensorOptions().with_req_grad(true));

    Value X = make_tensor(x_tensor, "X_data");
    Value Y = make_tensor(y_tensor, "Y_target");

    // 4. --- The Training Loop ---
    for (int epoch = 0; epoch < epochs; ++epoch) {
        Value predictions = model(X);  // (batch_size, output_features)
        Value loss = mse_loss(predictions, Y);

        float loss_value = loss.val().data<float>()[0];
        std::cout << "Epoch " << std::setw(2) << epoch 
                  << ", Loss: " << std::fixed << std::setprecision(4) << loss_value << std::endl;

        model.zero_grad();
        backward(loss);

        ag::SGD(loss, nullptr, learning_rate);
    }

    // 5. --- Clean up dynamically allocated modules ---
    for (auto* layer : model.get_layers()) {
        delete layer;
    }

    std::cout << "\n✅ Training finished successfully.\n";
    return 0;
}
