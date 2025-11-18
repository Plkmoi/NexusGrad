#include "ad/ag_all.hpp"
#include "layer/affine.hpp"
#include "layer/attention.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <cmath>
#include <limits>
#include <optim.hpp>

using namespace ag;
using namespace OwnTensor;

// ==================================================
// PURE CPU EMBEDDING TABLE
// vocab_size x d_model
// ==================================================
Tensor make_embedding_table(int vocab_size, int d_model) {
    Tensor emb(
        Shape{{vocab_size, d_model}},
        TensorOptions().with_device(Device::CPU)
    );

    float* ptr = emb.data<float>();
    int total = vocab_size * d_model;

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f / std::sqrt(static_cast<float>(d_model)));

    for (int i = 0; i < total; ++i) {
        ptr[i] = dist(rng);
    }

    return emb;
}

// ==================================================
// tokens: vector<int> of length B
// emb:   vocab_size x d_model
// RETURN: X = [B, d_model] CPU tensor
// ==================================================
Tensor embed_tokens(const std::vector<int>& tokens,
                    const Tensor& emb)
{
    int B          = static_cast<int>(tokens.size());
    int vocab_size = emb.shape().dims[0];
    int d_model    = emb.shape().dims[1];

    Tensor X(
        Shape{{B, d_model}},
        TensorOptions().with_device(Device::CPU)
    );

    const float* E = emb.data<float>();
    float* out     = X.data<float>();

    for (int b = 0; b < B; ++b) {
        int id = tokens[b];

        // Safety
        if (id < 0 || id >= vocab_size) {
            std::cerr << "Invalid token id: " << id << std::endl;
            id = 0; // fallback pad id
        }

        std::memcpy(
            out + b * d_model,
            E   + id * d_model,
            d_model * sizeof(float)
        );
    }

    return X;
}

// ----------------------------------------
// Helper: check for NaN / Inf in a Tensor
// ----------------------------------------
bool has_nan_or_inf(const Tensor& t, float* max_abs_out = nullptr) {
    Tensor cpu = t.to(Device::CPU);
    const auto& dims = cpu.shape().dims;

    std::size_t n = 1;
    for (auto d : dims) n *= static_cast<std::size_t>(d);

    const float* data = cpu.data<float>();
    bool bad = false;
    float max_abs = 0.f;

    for (std::size_t i = 0; i < n; ++i) {
        float v = data[i];
        if (std::isnan(v) || std::isinf(v)) {
            bad = true;
        }
        max_abs = std::max(max_abs, std::fabs(v));
    }

    if (max_abs_out) *max_abs_out = max_abs;
    return bad;
}

// ==================================================
// main: tiny LLaMA-style model with AlibiAttention
// ==================================================
int main() {
    std::cout << "========================================\n";
    std::cout << "--- Tiny ALiBi-LLaMA CE-with-Logits ---\n";
    std::cout << "========================================\n\n";

    // --- Hyperparameters ---
    const int B            = 32;    // batch size
    const int d_model      = 128;   // model width
    const int num_classes  = 10;    // classification logits dim
    const int num_layers   = 2;     // (Attn + SWIGLU) block pairs
    const float lr         = 0.001f;
    const int epochs       = 50;
    const int vocab_size   = 5000;  // fake vocab

    Device dev = Device::CUDA;

    // ---- 0. Embedding + fake tokens on CPU ----
    Tensor embedding_table = make_embedding_table(vocab_size, d_model);

    std::vector<int> tokens(B);
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> pick(0, vocab_size - 1);

    for (int i = 0; i < B; ++i) {
        tokens[i] = pick(rng);
    }

    Tensor X_cpu = embed_tokens(tokens, embedding_table);  // [B, d_model] on CPU

    // ---- 1. Build model: LLaMA-ish stack ----
    using namespace ag::layer;

    std::vector<Layer*> layers;
    layers.reserve(num_layers * 2 + 2);

    for (int i = 0; i < num_layers; ++i) {
        layers.push_back(new ResidualBlock({
            new RMSNorm(), // norm is parameter-free, will follow input device
            new AlibiAttention(d_model, /*n_heads=*/4, /*m=*/1.0f, dev)
        }));

        layers.push_back(new ResidualBlock({
            new RMSNorm(),
            new SWIGLU(d_model, d_model, dev)
        }));
    }

    layers.push_back(new RMSNorm());
    layers.push_back(new Linear(d_model, num_classes, dev));

    Traverse model(layers);

    std::cout << "Model created with " << model.parameters().size()
              << " parameter tensors.\n\n";

    // ---- 2. Move X to CUDA and wrap as Value ----
    Tensor X_gpu = X_cpu.to(dev);        // [B, d_model] on CUDA
    Value X = make_tensor(X_gpu, "tokens");

    // ---- 3. Build one-hot Y on CPU, then move once to CUDA ----
    Tensor Y_cpu(Shape{{B, num_classes}}, TensorOptions().with_device(Device::CPU));
    float* yp = Y_cpu.data<float>();

    std::uniform_int_distribution<int> classpick(0, num_classes - 1);
    std::mt19937 gen(42);

    for (int i = 0; i < B; ++i) {
        int c = classpick(gen);
        for (int j = 0; j < num_classes; ++j) {
            yp[i * num_classes + j] = (j == c) ? 1.0f : 0.0f;
        }
    }

    Tensor Y_gpu = Y_cpu.to(dev);
    Value Y = make_tensor(Y_gpu, "Y_target");

    // ==================================================
    // 4. Training loop (NO disten(loss, ...) ping-pong)
    // ==================================================
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Forward: logits [B, num_classes]
        Value logits = model(X);                    // stays on CUDA
        Value loss   = cross_entropy_with_logits(logits, Y);

        // Inspect loss safely: copy just the scalar to CPU
        Tensor loss_cpu = loss.val().to(Device::CPU);
        float loss_value = loss_cpu.data<float>()[0];

        // Optional: check logits for NaNs / exploding values
        float max_logit_abs = 0.f;
        bool bad_logits = has_nan_or_inf(logits.val(), &max_logit_abs);

        std::cout << "Epoch " << std::setw(2) << epoch
                  << ", Loss: " << std::fixed << std::setprecision(6)
                  << loss_value
                  << ", max|logits| = " << max_logit_abs
                  << (bad_logits ? "  (⚠️ NaN/Inf in logits)" : "")
                  << "\n";

        // Backward + SGD on CUDA
        model.zero_grad();
        backward(loss);
        ag::SGD(loss, nullptr, lr);

        // If things explode, you can early-break for debugging
        if (std::isnan(loss_value) || std::isinf(loss_value) || bad_logits) {
            std::cout << "Stopping early due to NaN/Inf.\n";
            break;
        }
    }

    // ---- Cleanup for heap-allocated Layers ----
    for (Layer* L : model.get_layers()) {
        delete L;
    }

    std::cout << "\n✅ Tiny ALiBi-LLaMA training finished.\n";
    return 0;
}
