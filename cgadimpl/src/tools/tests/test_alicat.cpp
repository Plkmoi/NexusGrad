#include "ad/ag_all.hpp"   // Main umbrella header for the framework
#include <iostream>
#include <vector>
#include <iomanip>
#include <random>
#include <optim.hpp>
#include "layer/tokeni.hpp"




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

    // 1. --- Hyperparameters ---
    const int B            = 32;    // batch size
    const int d_model      = 128;   // model / residual width
    const int num_classes  = 10;    // number of classes (logits dim)
    const int num_layers   = 2;     // (Attn + SWIGLU) block pairs
    const float lr         = 0.001f;
    const int epochs       = 410;
    int vocab_size   = 5000;     // integer tokens 0..19

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

      // 0. Choose a vocabulary size for your fake tokens

// -------------------------------------------
// Load corpus + tokenize
// -------------------------------------------
std::string text = ag::layer::load_text_file("/home/blubridge-034/Downloads/Newf/cgadimpl/cgadimpl/src/layer/corpus.txt");
std::vector<int> tokens = ag::layer::tokenize_bytes(text);

// derive vocab_size dynamically from corpus
for (int id : tokens)
    vocab_size = std::max(vocab_size, id);
vocab_size += 1;

// -------------------------------------------
// Build embedding table (CPU)
// -------------------------------------------
Tensor embedding_table = make_embedding_table(vocab_size, d_model);

// -------------------------------------------
// Create one batch from corpus tokens
// -------------------------------------------
if ((int)tokens.size() < B) {
    std::cerr << "Corpus size < batch size\n";
    return -1;
}

std::vector<int> batch(tokens.begin(), tokens.begin() + B);

// -------------------------------------------
// Embed → produces X_cpu [B, d_model]
// -------------------------------------------
Tensor X_cpu = embed_tokens(batch, embedding_table);



    std::vector<ag::layer::Layer*> layers;
    layers.reserve(num_layers * 2 + 2);
auto dev = Device::CUDA;
    for (int i = 0; i < num_layers; ++i) {
        // Attention sub-block: x + Attn(RMSNorm(x))
        layers.push_back(new ag::layer::ResidualBlock({
            new ag::layer::RMSNorm(),
            new ag::layer::AlibiAttention(/*d_model=*/d_model, /*n_heads=*/4, /*m=*/1.0f, dev)

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


    //
    // Y: [B, num_classes] one-hot, using your pattern with mt19937.
    //
    
    std::mt19937 gen(42);
Tensor X_gpu = X_cpu.to(dev);
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
                ag::disten(Y, dev);

                Value logits = model(X);
                Value loss = cross_entropy_with_logits(logits, Y);
                zero_val(loss);

    // 4. --- Training loop with cross-entropy-with-logits ---
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Forward: logits [B, num_classes]
        forward(loss);
        ag::disten(loss, Device::CPU);

        float loss_value = loss.val().data<float>()[0];
        std::cout << "Epoch " << std::setw(2) << epoch
                  << ", Loss: " << std::fixed << std::setprecision(6)
                  << loss_value << std::endl;
        ag::disten(loss, dev);

        // Backward + update
        zero_grad(logits);
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
