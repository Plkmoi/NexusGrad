#include "ad/ag_all.hpp"
#include <iostream>
#include <vector>
#include <iomanip>
#include <random>
#include <optim.hpp>


// ==================================================
// PURE CPU EMBEDDING TABLE
// vocab_size x d_model
// ==================================================
Tensor make_embedding_table(int vocab_size, int d_model) {
    Tensor emb(
        Shape{{vocab_size, d_model}},
        OwnTensor::TensorOptions().with_device(Device::CPU)
    );

    float* ptr = emb.data<float>();
    int total = vocab_size * d_model;

    // Xavier normal or simple N(0, 1/sqrt(d_model))
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f / std::sqrt(d_model));

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
    int B = tokens.size();
    int vocab_size = emb.shape().dims[0];
    int d_model    = emb.shape().dims[1];

    Tensor X(
        Shape{{B, d_model}},
        OwnTensor::TensorOptions().with_device(Device::CPU)
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

        // Copy the embedding row: E[id] → X[b]
        std::memcpy(
            out + b * d_model,
            E   + id * d_model,
            d_model * sizeof(float)
        );
    }

    return X;
}





int main() {
    using namespace ag;
    using namespace OwnTensor;

    std::cout << "========================================\n";
    std::cout << "--- Tiny LLaMA-Style CE-with-Logits ---\n";
    std::cout << "========================================\n\n";

    // ---- Hyperparameters ----
    const int B            = 64;     // batch size
    const int d_model      = 256;    // embedding + model width
    const int vocab_size   = 5000;     // integer tokens 0..19
    const int num_classes  = 3;      // classification
    const int num_layers   = 2;
    const float lr         = 0.0005f;
    const int epochs       = 100;

    // 0. Choose a vocabulary size for your fake tokens

// 1. Create embedding table (CPU)
Tensor embedding_table = make_embedding_table(vocab_size, d_model);

// 2. Generate fake tokens
std::vector<int> tokens(B);
std::mt19937 rng(42);
std::uniform_int_distribution<int> pick(0, vocab_size - 1);

for (int i = 0; i < B; ++i) {
    tokens[i] = pick(rng);
}

// 3. Embed tokens → produce X [B, d_model]
Tensor X_cpu = embed_tokens(tokens, embedding_table);


    Device dev = Device::CUDA;

    // ---- Build model ----
    using namespace ag::layer;

    std::vector<Layer*> layers;
    layers.reserve(2 + num_layers * 2);

    // Embedding first (integer token -> vector)

    for (int i = 0; i < num_layers; ++i) {
        layers.push_back(new ResidualBlock({
            new RMSNorm(),
            new Attention(d_model, d_model, dev)
        }));
        layers.push_back(new ResidualBlock({
            new RMSNorm(),
            new SWIGLU(d_model, d_model, dev)
        }));
    }

    layers.push_back(new RMSNorm());
    layers.push_back(new Linear(d_model, num_classes, dev));

    Traverse model(layers);

    std::cout << "Model created with "
              << model.parameters().size()
              << " parameter tensors.\n\n";

    // ---- Create integer tokens X ----
    //
    // X: [B] integers
    //


    std::mt19937 gen(42);
Tensor X_gpu = X_cpu.to(Device::CUDA);
Value X = make_tensor(X_gpu, "tokens");

    // ---- Create integer class labels Y (one-hot) ----
    Tensor Yt(Shape{{B, num_classes}}, TensorOptions());
    float* yp = Yt.data<float>();

    std::uniform_int_distribution<int> classpick(0, num_classes - 1);

    for (int i = 0; i < B; ++i) {
        int c = classpick(gen);
        for (int j = 0; j < num_classes; ++j)
            yp[i * num_classes + j] = (j == c) ? 1.0f : 0.0f;
    }

    Value Y = make_tensor(Yt, "Y_target");
    disten(Y, dev);

    // ---- Train ----
    for (int epoch = 0; epoch < epochs; ++epoch) {
        Value logits = model(X);
        Value loss   = cross_entropy_with_logits(logits, Y);

        disten(loss, Device::CPU);
        float loss_value = loss.val().data<float>()[0];

        std::cout << "Epoch " << std::setw(2) << epoch
                  << ", Loss: " << std::fixed << std::setprecision(6)
                  << loss_value << "\n";

        disten(loss, dev);

        model.zero_grad();
        backward(loss);
        SGD(loss, nullptr, lr);
    }

    // ---- Cleanup ----
    for (Layer* L : model.get_layers()) delete L;

    std::cout << "\nDone!\n";
    return 0;
}
