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
Tensor embed_tokens(const std::vector<int>& tokens, const Tensor& emb) {
    int B = tokens.size();
    int vocab_size = emb.shape().dims[0];
    int d_model = emb.shape().dims[1];

    Tensor X(
        Shape{{B, d_model}},
        OwnTensor::TensorOptions().with_device(Device::CPU)
    );

    const float* E = emb.data<float>();
    float* out = X.data<float>();

    for (int b = 0; b < B; ++b) {
        int id = tokens[b];

        // Safety check
        if (id < 0 || id >= vocab_size) {
            std::cerr << "Invalid token id: " << id << std::endl;
            id = 0; // fallback pad id
        }

        // Copy the embedding row: E[id] → X[b]
        std::memcpy(
            out + b * d_model,
            E + id * d_model,
            d_model * sizeof(float)
        );
    }

    return X;
}

#include <cmath>
#include <vector>
#include <random>
#include <iostream>
int safe_sample_from_logits_tensor(const Tensor& logits_tensor_cpu, float temp = 1.0f) {
    const auto& dims = logits_tensor_cpu.shape().dims;

    if (dims.empty()) {
        std::cerr << "safe_sample_from_logits_tensor: empty logits shape\n";
        return 0;
    }

    // Take the LAST dimension as V (class count)
    int V = static_cast<int>(dims.back());

    if (V <= 0) {
        std::cerr << "Invalid logits vocab size V = " << V << std::endl;
        return 0;  // Safe fallback
    }

    const float* ptr = logits_tensor_cpu.data<float>();
    if (!ptr) {
        std::cerr << "Null pointer to logits data!" << std::endl;
        return 0;  // Safe fallback
    }

    // If shape is [B, V] with B>1, you probably want the last row.
    // For now, assume [1, V] or [V]. If [B,V], use row 0:
    if (dims.size() == 2) {
        int B = static_cast<int>(dims[0]);
        if (B != 1) {
            std::cerr << "Warning: logits has shape [" << B << ", " << V
                      << "], using row 0 for sampling.\n";
        }
        // ptr already points to row 0, contiguous V floats.
    }

    // Find max logit for numerical stability
    float maxlog = ptr[0];
    for (int i = 1; i < V; ++i) {
        if (ptr[i] > maxlog) {
            maxlog = ptr[i];
        }
    }

    // Convert logits to probabilities
    std::vector<double> probs(V);
    double sum = 0.0;
    bool any_finite = false;

    for (int i = 0; i < V; ++i) {
        float l = ptr[i];
        if (!std::isfinite(l)) {
            probs[i] = 0.0;
            continue;
        }
        double ex = std::exp(double(l - maxlog) / temp);
        probs[i] = ex;
        sum += ex;
        any_finite = true;
    }

    // Fallback if everything is zero/NaN
    if (!any_finite || sum <= 0.0 || std::isnan(sum)) {
        std::fill(probs.begin(), probs.end(), 1.0);
        sum = static_cast<double>(V);
    }

    // Normalize
    for (int i = 0; i < V; ++i) {
        probs[i] /= sum;
    }

    static thread_local std::mt19937 rng(42); // or seed however you like
    std::discrete_distribution<int> dist(probs.begin(), probs.end());

    int sampled = dist(rng);
    if (sampled < 0) sampled = 0;
    if (sampled >= V) sampled = V - 1;

    return sampled;
}



int main() {
    using namespace ag;
    using namespace OwnTensor;

    std::cout << "========================================\n";
    std::cout << "--- Tiny LLaMA-Style Autoregressive ---\n";
    std::cout << "========================================\n\n";

    // 1. --- Hyperparameters ---
    const int B = 32;            // batch size
    const int d_model = 128;     // model / residual width
    // const int vocab_size = 5000; // number of classes (logits dim)
    const int num_layers = 2;    // (Attn + SWIGLU) block pairs
    const float lr = 0.001f;
    const int epochs = 2100;
    int vocab_size = 5000;       // integer tokens 0..19

    // -------------------------------------------
    // Load corpus + tokenize
    // -------------------------------------------
    std::string text = ag::layer::load_text_file("/home/blubridge-034/Downloads/Newf/cgadimpl/cgadimpl/src/layer/corpus.txt");
    std::vector<int> tokens = ag::layer::tokenize_bytes(text);

    // derive vocab_size dynamically from corpus
    for (int id : tokens) {
        vocab_size = std::max(vocab_size, id);
    }
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

    // -------------------------------------------
    // Create Autoregressive Inputs and Targets
    // -------------------------------------------
    std::vector<int> batch(tokens.begin(), tokens.begin() + B); // batch of tokens

    // X is the input sequence, Y is the target sequence (shifted)
    Tensor X_cpu = embed_tokens(batch, embedding_table); // [B, d_model]

    std::vector<ag::layer::Layer*> layers;
    layers.reserve(num_layers * 2 + 2);
    auto dev = Device::CUDA;

    // Build model layers
    for (int i = 0; i < num_layers; ++i) {
        layers.push_back(new ag::layer::ResidualBlock({
            new ag::layer::RMSNorm(),
            new ag::layer::AlibiAttention(d_model, 4, 1.0f, dev)
        }));

        layers.push_back(new ag::layer::ResidualBlock({
            new ag::layer::RMSNorm(),
            new ag::layer::SWIGLU(d_model, d_model, dev)
        }));
    }

    layers.push_back(new ag::layer::RMSNorm());
    layers.push_back(new ag::layer::Linear(d_model, vocab_size, dev));

    ag::layer::Traverse model(layers);

    std::cout << "Model created with " << model.parameters().size()
              << " parameter tensors.\n\n";



    std::mt19937 gen(42);
Tensor X_gpu = X_cpu.to(dev);
Value X = make_tensor(X_gpu, "tokens");

    // ---- Create integer class labels Y (one-hot) ----
    Tensor Yt(Shape{{B, vocab_size}}, TensorOptions());
    float* yp = Yt.data<float>();

        for (int i = 0; i < B - 1; ++i) {
            int c = batch[i + 1];  // next token becomes target
            for (int j = 0; j < vocab_size; ++j)
                yp[i * vocab_size + j] = (j == c) ? 1.0f : 0.0f;
        }

    Value Y = make_tensor(Yt, "Y_target");
                ag::disten(Y, dev);

                Value logits = model(X);
                Value loss = cross_entropy_with_logits(logits, Y);
                zero_val(loss);


    // -------------------------------------------
    // Autoregressive Training Loop
    // -------------------------------------------
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Create X and Y: shift tokens for autoregressive training
        
            int start = epoch % (tokens.size() - B - 1);
    std::vector<int> batch(tokens.begin() + start, tokens.begin() + start + B);

        Tensor X_cpu = embed_tokens(batch, embedding_table);
    X.node->value = X_cpu.to(dev);
    ag::disten(X, dev);

        Tensor Ytn(Shape{{B, vocab_size}}, TensorOptions());
    float* ypA = Ytn.data<float>();


        for (int i = 0; i < B - 1; ++i) {
            int c = batch[i + 1]; // This is the correct target token ID
            if (c >= 0 && c < vocab_size) {
                ypA[i * vocab_size + c] = 1.0f; // This will not segfault now
            }
        }

        Y.node->value = Ytn.to(dev);
    ag::disten(Y, dev);

        // Forward pass
        Value logits = model(X);
        Value loss = cross_entropy_with_logits(logits, Y);
        zero_val(loss);

        // Training loop with cross-entropy
        forward(loss);
        ag::disten(loss, Device::CPU);

        float loss_value = loss.val().data<float>()[0];
        std::cout << "Epoch " << std::setw(2) << epoch
                  << ", Loss: " << std::fixed << std::setprecision(6)
                  << loss_value << std::endl;
        ag::disten(loss, dev);

        // Backward pass and optimization
        zero_grad(logits);
        backward(loss);
        ag::SGD(loss, nullptr, lr);
    }


    
    // -----------------------------
    // Autoregressive Generation
    // -----------------------------
    std::cout << "\n========================================\n";
    std::cout << "--- GENERATING TEXT ---\n";
    std::cout << "========================================\n\n";

    int start_token = 32;  // Start token (e.g., space)
    int token = start_token;
    std::string generated;
    bool printed_debug = false;
    const int GEN_STEPS = 200;

    for (int step = 0; step < GEN_STEPS; ++step) {
        std::vector<int> one = { token };
        Tensor x_cpu = embed_tokens(one, embedding_table); // [1, d_model]
        Tensor x_dev = x_cpu.to(dev);
        Value X_in = make_tensor(x_dev, "infer_tok");

        // Forward pass to get logits
        Value out_val = model(X_in);
        ag::disten(out_val, Device::CPU);
        Tensor logits_cpu = out_val.val(); // now on CPU

        // Sample the next token from the logits
        int next_tok = safe_sample_from_logits_tensor(logits_cpu, 1.0f);

        // Add the next token to the generated string
        token = next_tok;
        if (token >= 32 && token < 127) {
            generated.push_back(static_cast<char>(token));
        } else {
            generated.push_back('?');
        }
    }

    std::cout << "---- GENERATED TEXT ----\n" << generated << "\n\n";

    // Cleanup for heap-allocated Layers
    for (auto* layer : model.get_layers()) {
        delete layer;
    }

    std::cout << "\n✅ Tiny LLaMA-style autoregressive training finished.\n";
    return 0;
}


