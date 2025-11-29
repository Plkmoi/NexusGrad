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


Tensor embed_tokens_3d(const std::vector<std::vector<int>>& sequences, const Tensor& emb) {
    int B = sequences.size();           // Batch size
    if (B == 0) {
        return Tensor(Shape{{0, 0, 0}}, OwnTensor::TensorOptions().with_device(Device::CPU));
    }

    int S = sequences[0].size();        // Sequence length (assumes all sequences in batch are same length)
    int vocab_size = emb.shape().dims[0];
    int d_model = emb.shape().dims[1];

    // Allocate the 3D output tensor: Shape [B, S, D]
    Tensor X(Shape{{B, S, d_model}}, OwnTensor::TensorOptions().with_device(Device::CPU));

    const float* E = emb.data<float>();
    float* out = X.data<float>();

    for (int b = 0; b < B; ++b) {
        if (sequences[b].size() != S) {
            // Handle cases where sequences have different lengths (e.g., throw or pad)
            throw std::runtime_error("All sequences in the batch must have the same length (S) for a single 3D tensor.");
        }

        for (int s = 0; s < S; ++s) {
            int id = sequences[b][s];

            // Safety check
            if (id < 0 || id >= vocab_size) {
                std::cerr << "Invalid token id found: " << id << std::endl;
                id = 0; // Fallback to pad id (index 0)
            }

            // Calculate the correct offset in the flat output buffer
            // We are moving along the B * S dimension before hitting D
            size_t output_offset = (b * S + s) * d_model;
            size_t emb_offset = id * d_model;

            // Copy the embedding row: E[id] → X[b][s]
            std::memcpy(out + output_offset,
                        E + emb_offset,
                        d_model * sizeof(float));
        }
    }
    return X;
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
    // const int vocab_size = 5000; // number of classes (logits dim)
    const int num_layers = 2;    // (Attn + SWIGLU) block pairs
    const float lr = 0.001f;
    const int epochs = 2100;
    int vocab_size = 5000;       // integer tokens 0..19
    int Heads = 4;

    const int S = 256; // Sequence length (needs to be defined)
    const int d_model = 512; // Embedding dimension
    auto dev = Device::CPU;
    int K = 10;

    // -------------------------------------------
    // Load corpus + tokenize
    // -------------------------------------------
    std::string text = ag::layer::load_text_file("/home/blubridge-034/Downloads/Newf/cgadimpl/cgadimpl/src/layer/corpus.txt");
    std::vector<int> all_tokens = ag::layer::tokenize_bytes(text); // Renamed to clarify it's the full corpus

    for (int id : all_tokens) {
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

    // You need enough tokens for BATCH * SEQUENCE LENGTH
    int required_tokens = B * S;
    if ((int)all_tokens.size() < required_tokens) {
        std::cerr << "Corpus size (" << all_tokens.size() << ") is too small for batch size B * S (" << required_tokens << ")\n";
        return -1;
    }

    // Prepare a 2D input structure: std::vector<std::vector<int>>
    std::vector<std::vector<int>> batch_sequences;
    
    // Extract the required block of tokens from the large corpus vector
    // This example just takes the first B * S tokens and reshapes them
    
    for (int i = 0; i < B; ++i) {
        std::vector<int> current_sequence(
            all_tokens.begin() + i * S, 
            all_tokens.begin() + (i + 1) * S
        );
        batch_sequences.push_back(current_sequence);
    }



    std::vector<ag::layer::Layer*> layers;
    layers.reserve(num_layers * 2 + 2);

    // Build model layers
    for (int i = 0; i < num_layers; ++i) {
        layers.push_back(new ag::layer::ResidualBlock({
            new ag::layer::RMSNorm(),
            new ag::layer::AlibiAttention(Heads, B, S, d_model, dev)
        }));

        layers.push_back(new ag::layer::ResidualBlock({
            new ag::layer::RMSNorm(),
            new ag::layer::SWIGLU(B, S, d_model, K, dev)
        }));
    }

    layers.push_back(new ag::layer::RMSNorm());
    layers.push_back(new ag::layer::Linear(B, S, d_model, dev));

    ag::layer::Traverse model(layers);

    std::cout << "Model created with " << model.parameters().size()
              << " parameter tensors.\n\n";




Tensor X_cpu = Tensor(Shape{{B, S, d_model}}, OwnTensor::TensorOptions().with_device(Device::CPU));
    Tensor X_gpu = X_cpu.to(dev);
    Value X = make_tensor(X_gpu, "tokens");


    // ---- Create integer class labels Y (targets) ----
    // The target tensor must be the same shape as your logits output before cross_entropy.
    // If your final linear layer is [B, S, D] -> [B, S, vocab_size]
    Tensor Yt(Shape{{B, S, vocab_size}}, TensorOptions()); // Correct 3D target shape
    // Note: Yt should generally be integer/long type if using typical cross_entropy implementations
    // but your original code used float*, implying one-hot encoding or a specific loss function.
    // We will stick to your float* format for now, assuming your `cross_entropy_with_logits` handles this format.
    
    float* yp = Yt.data<float>(); // Pointer to the flat memory of Yt

    // We don't populate Yt here statically, it gets populated dynamically in the loop.

    Value Y = make_tensor(Yt, "Y_target");
    ag::disten(Y, dev); // Distribute Y placeholder to device

    // Placeholder run (might crash if model weights aren't initialized/sized correctly yet)
    // Value logits = model(X); 
    // Value loss = cross_entropy_with_logits(logits, Y);
    // zero_val(loss);


    // -------------------------------------------
    // Autoregressive Training Loop
    // -------------------------------------------
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Create X and Y: shift tokens for autoregressive training

        // Calculate a starting point in the corpus that allows for a full B*S sequence + targets
        int start = (epoch * B * S) % (all_tokens.size() - B * S - 1); 
        // A better way to loop through all data might be needed, but this works for epoch cycling

        std::vector<std::vector<int>> batch_sequences;
        std::vector<std::vector<int>> target_sequences; // We need target sequences too

        for (int i = 0; i < B; ++i) {
            int sequence_start_index = start + i * S;
            
            // Input X sequence: [start ... start + S - 1]
            std::vector<int> current_x_seq(
                all_tokens.begin() + sequence_start_index, 
                all_tokens.begin() + sequence_start_index + S
            );
            
            // Target Y sequence (shifted): [start + 1 ... start + S]
            std::vector<int> current_y_seq(
                all_tokens.begin() + sequence_start_index + 1, 
                all_tokens.begin() + sequence_start_index + S + 1
            );

            batch_sequences.push_back(current_x_seq);
            target_sequences.push_back(current_y_seq);
        }
        
        // Use the 3D embedding function for X
        X_cpu = embed_tokens_3d( batch_sequences, embedding_table);
        
        // Update the Value object X with the new data and push to GPU
        X.node->value = X_cpu.to(dev);
        // ag::disten(X, dev); // This line is likely redundant if to() already moved it

        // Prepare the Target Tensor Ytn (Y-target-now)
        // It must be [B, S, vocab_size] shape (one-hot encoded)
        Tensor Ytn(Shape{{B, S, vocab_size}}, OwnTensor::TensorOptions().with_device(Device::CPU));
        float* ypA = Ytn.data<float>();
        
        // One-hot encode the target_sequences into the flat Ytn tensor memory
        for (int b = 0; b < B; ++b) {
            for (int s = 0; s < S; ++s) {
                int target_id = target_sequences[b][s];
                if (target_id < 0 || target_id >= vocab_size) {
                    throw std::out_of_range("Target ID is invalid during Y prep.");
                }
                // Calculate the flat index in the 3D tensor data [b][s][target_id]
                size_t flat_index = (static_cast<size_t>(b) * S + s) * vocab_size + target_id;
                ypA[flat_index] = 1.0f; // Set the correct class probability to 1.0
            }
        }
        
        // Update the Value object Y with the new data and push to GPU
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
        ag::opti.SGD(loss, lr);
        opti.epoch();
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

