#include "ad/ag_all.hpp"   // Main umbrella header for the framework
#include <iostream>
#include <vector>
#include <iomanip>
#include <random>
#include <optim.hpp>

int main() {
    using namespace ag;
    using namespace OwnTensor;

    std::cout << "========================================\n";
    std::cout << "--- Tiny LLaMA-Style CE-with-Logits ---\n";
    std::cout << "========================================\n\n";

    // 1. --- Hyperparameters ---
    const int B            = 32;    // batch size
    const int d_model      = 128;   // model / residual width
    const int num_classes  = 10;    // number of classes (logits dim)
    const int num_layers   = 2;     // (Attn + SWIGLU) block pairs
    const float lr         = 0.001f;
    const int epochs       = 50;

    // 2. --- Build a LLaMA-ish stack of modules ---
    //
    // Repeated block:
    //   x -> Residual( RMSNorm -> Attention(d_model -> d_model) )
    //     -> Residual( RMSNorm -> SWIGLU(d_model -> d_model) )
    //
    // Then:
    //     -> RMSNorm
    //     -> Linear(d_model -> num_classes)  // logits
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

    // Final norm + classification head (logits)
    layers.push_back(new ag::nn::RMSNorm());
    layers.push_back(new ag::nn::Linear(d_model, num_classes));

    ag::nn::Sequential model(layers);

    std::cout << "Model created with " << model.parameters().size()
              << " parameter tensors.\n\n";

    // 3. --- Data: X (inputs) and Y (one-hot labels) ---
    //
    // X: [B, d_model], just random for now, but fixed across epochs.
    //
    Tensor Xt = Tensor::randn(
        Shape{{B, d_model}},
        TensorOptions()   // inputs are not trainable
    );
    Value X = make_tensor(Xt, "X");

    //
    // Y: [B, num_classes] one-hot, using your pattern with mt19937.
    //
    Tensor Yt(Shape{{B, num_classes}}, TensorOptions());
    float* yt_data = Yt.data<float>();

    std::mt19937 gen(42);
    std::uniform_int_distribution<int> pick(0, num_classes - 1);

    for (int i = 0; i < B; ++i) {
        int k = pick(gen);  // class index
        for (int j = 0; j < num_classes; ++j) {
            yt_data[i * num_classes + j] = (j == k) ? 1.0f : 0.0f;
        }
    }
    Value Y = make_tensor(Yt, "Y_target");   // one-hot labels

    // 4. --- Training loop with cross-entropy-with-logits ---
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Forward: logits [B, num_classes]
        Value logits = model(X);

        // Your CE-with-logits implementation taking (logits, one_hot_targets)
        Value loss = cross_entropy_with_logits(logits, Y);

        float loss_value = loss.val().data<float>()[0];
        std::cout << "Epoch " << std::setw(2) << epoch
                  << ", Loss: " << std::fixed << std::setprecision(6)
                  << loss_value << std::endl;

        // Backward + update
        model.zero_grad();
        backward(loss);
        ag::SGD(loss, nullptr, lr);
    }

    // 5. --- Cleanup for heap-allocated modules ---
    for (auto* layer : model.get_layers()) {
        delete layer;
    }

    std::cout << "\n✅ Tiny LLaMA-style CE-with-logits training finished.\n";
    return 0;
}
