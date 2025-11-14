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
    const int B            = 64;    // batch size
    const int d_model      = 512;   // model / residual width
    const int num_classes  = 3;    // number of classes (logits dim)
    const int num_layers   = 4;     // (Attn + SWIGLU) block pairs
    const float lr         = 0.0001f;
    const int epochs       = 100;

    // 2. --- Build a LLaMA-ish stack of Layers ---
    //
    // Repeated block:
    //   x -> Residual( RMSNorm -> Attention(d_model -> d_model) )
    //     -> Residual( RMSNorm -> SWIGLU(d_model -> d_model) )
    //
    // Then:
    //     -> RMSNorm
    //     -> Linear(d_model -> num_classes)  // logits
    //
    std::vector<ag::layer::Layer*> layers;
    layers.reserve(num_layers * 2 + 2);

    Device dev=Device::CUDA;

    for (int i = 0; i < num_layers; ++i) {
        // Attention sub-block: x + Attn(RMSNorm(x))
        layers.push_back(new ag::layer::ResidualBlock({
            new ag::layer::RMSNorm(),
            new ag::layer::Attention(d_model, d_model, dev)
        }));

        // MLP / SWIGLU sub-block: x + SWIGLU(RMSNorm(x))
        layers.push_back(new ag::layer::ResidualBlock({
            new ag::layer::RMSNorm(),
            new ag::layer::SWIGLU(d_model, d_model, dev)
        }));
    }

    // Final norm + classification head (logits)
    layers.push_back(new ag::layer::RMSNorm());
    layers.push_back(new ag::layer::Linear(d_model, num_classes, dev));

    ag::layer::Traverse model(layers);

    std::cout << "Model created with " << model.parameters().size()
              << " parameter tensors.\n\n";

    // 3. --- Data: X (inputs) and Y (one-hot labels) ---
    //
    // X: [B, d_model], just random for now, but fixed across epochs.
    //
    Tensor Xt = Tensor::randn(
        Shape{{B, d_model}},
        TensorOptions().with_device(dev)   // inputs are not trainable
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

                ag::disten(Y, dev);

        // Value logits = model(X);

        // // Your CE-with-logits implementation taking (logits, one_hot_targets)
        // Value loss = cross_entropy_with_logits(logits, Y);

        // zero_val(loss);
    // 4. --- Training loop with cross-entropy-with-logits ---
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Forward: logits [B, num_classes]

        // forward(loss);

        Value logits = model(X);
        // Value newlogits = softmax_row(logits);

        // Your CE-with-logits implementation taking (logits, one_hot_targets)
        Value loss = cross_entropy_with_logits(logits, Y);

        ag::disten(loss, Device::CPU);


        float loss_value = loss.val().data<float>()[0];
        std::cout << "Epoch " << std::setw(2) << epoch
                  << ", Loss: " << std::fixed << std::setprecision(6)
                  << loss_value << std::endl;

        // Backward + update

        ag::disten(loss, dev);

        model.zero_grad();
        backward(loss);
        ag::SGD(loss, nullptr, lr);
    }

    // 5. --- Cleanup for heap-allocated Layers ---
    for (auto* layer : model.get_layers()) {
        delete layer;
    }

    std::cout << "\n✅ Tiny LLaMA-style CE-with-logits training finished.\n";
    return 0;
}
