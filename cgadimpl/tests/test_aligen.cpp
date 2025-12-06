#include "ad/ag_all.hpp"   // Main umbrella header for the framework
#include <iostream>
#include <vector>
#include <iomanip>
#include <random>
#include <optim.hpp>
#include "layer/tokeni.hpp"
#include "layer/embed.hpp"



int main() {
    using namespace ag;
    using namespace OwnTensor;

    std::cout << "========================================\n";
    std::cout << "--- Tiny LLaMA-Style Autoregressive ---\n";
    std::cout << "========================================\n\n";

    // 1. --- Hyperparameters ---
    const int B = 4;            // batch size
    // const int vocab_size = 5000; // number of classes (logits dim)
    const int num_layers = 2;    // (Attn + SWIGLU) block pairs
    const float lr = 0.001f;
    const int epochs = 2100;
    int vocab_size = 5000;       // integer tokens 0..19
    int Heads = 8;

    const int S = 128; // Sequence length (needs to be defined)
    const int d_model = 256; // Embedding dimension
    auto dev = Device::CUDA;
    int K = 5;

    // -------------------------------------------
    // Load corpus + tokenize
    // -------------------------------------------

    ag::layer::TokeniTokenizer tok("tokenizer.model");

    std::string text = ag::layer::load_text_file("/home/blubridge-034/Downloads/Newf/cgadimpl/cgadimpl/src/layer/corpus.txt");
    // std::vector<int> all_tokens = ag::layer::tokenize_bytes(text); // Renamed to clarify it's the full corpus


    // -------------------------------------------
    // Build embedding table (CPU)
    // -------------------------------------------
        std::vector<int> all_tokens = tok.encode(text);

            for (int id : all_tokens) {
        vocab_size = std::max(vocab_size, id);
    }
    vocab_size += 1;

    auto make_batch = [&](std::vector<std::vector<int>>& batch_seq,
                      std::vector<std::vector<int>>& target_seq,
                      int offset)
{
    batch_seq.clear();
    target_seq.clear();

    for (int b = 0; b < B; ++b) {
        std::vector<int> seq;
        seq.reserve(S);

        for (int s = 0; s < S; ++s)
            seq.push_back(all_tokens[offset + b*S + s]);

        batch_seq.push_back(seq);

        // autoregressive target = next token
        std::vector<int> tgt;
        tgt.reserve(S);

        for (int s = 0; s < S; ++s)
            tgt.push_back(all_tokens[offset + b*S + s + 1]);

        target_seq.push_back(tgt);
    }
};



    // Tensor W = ag::layer::make_embedding_table(/*vocab*/ vocab_size, /*dim*/ 128);
    // Tensor E = ag::layer::embed_tokens(W, all_tokens);

    // // -------------------------------------------
    // // Create one batch from corpus tokens
    // // -------------------------------------------

    // // You need enough tokens for BATCH * SEQUENCE LENGTH
    // int required_tokens = B * S;
    // if ((int)all_tokens.size() < required_tokens) {
    //     std::cerr << "Corpus size (" << all_tokens.size() << ") is too small for batch size B * S (" << required_tokens << ")\n";
    //     return -1;
    // }

    // // Prepare a 2D input structure: std::vector<std::vector<int>>
    // std::vector<std::vector<int>> batch_sequences;
    
    // // Extract the required block of tokens from the large corpus vector
    // // This example just takes the first B * S tokens and reshapes them
    
    // for (int i = 0; i < B; ++i) {
    //     std::vector<int> current_sequence(
    //         all_tokens.begin() + i * S, 
    //         all_tokens.begin() + (i + 1) * S
    //     );
    //     batch_sequences.push_back(current_sequence);
    // }



    std::vector<ag::layer::Layer*> layers;
    layers.reserve(num_layers * 2 + 2);

    // Build model layers
    for (int i = 0; i < num_layers; ++i) {
        layers.push_back(new ag::layer::ResidualBlock({
            new ag::layer::RMSNorm(),
            new ag::layer::AlibiAttention(B, S, d_model, Heads, dev)
        }));

        layers.push_back(new ag::layer::ResidualBlock({
            new ag::layer::RMSNorm(),
            new ag::layer::SWIGLU(B, S, d_model, K, dev)
            // new ag::layer::Mish()
        }));
    }

    layers.push_back(new ag::layer::RMSNorm());
    layers.push_back(new ag::layer::Linear(B, vocab_size, d_model, dev));

    ag::layer::Traverse model(layers);

    std::cout << "Model created with " << model.parameters().size()
              << " parameter tensors.\n\n";





// Tensor X_cpu = Tensor(Shape{{B, S, d_model}}, OwnTensor::TensorOptions().with_device(dev));
//     Tensor X_gpu = X_cpu.to(dev);
//     Value X = make_tensor(X_gpu, "tokens");

//     ag::debug::print_tensor("xwesee", X.val().to_cpu());


    // ---- Create integer class labels Y (targets) ----
    // The target tensor must be the same shape as your logits output before cross_entropy.
    // If your final linear layer is [B, S, D] -> [B, S, vocab_size]
    // Tensor Yt(Shape{{B, S, vocab_size}}, OwnTensor::TensorOptions().with_device(dev).with_req_grad(true)); // Correct 3D target shape
    // Note: Yt should generally be integer/long type if using typical cross_entropy implementations
    // but your original code used float*, implying one-hot encoding or a specific loss function.
    // We will stick to your float* format for now, assuming your `cross_entropy_with_logits` handles this format.
    
    // float* yp = Yt.data<float>(); // Pointer to the flat memory of Yt

    // We don't populate Yt here statically, it gets populated dynamically in the loop.

    // Value Y = make_tensor(Yt, "Y_target");
    // ag::disten(Y, dev); // Distribute Y placeholder to device

    // Placeholder run (might crash if model weights aren't initialized/sized correctly yet)
    // Value logits = model(X); 
    // Value loss = cross_entropy_with_logits(logits, Y);
    // zero_val(loss);


    // -------------------------------------------
    // Autoregressive Training Loop
    // -------------------------------------------
// Allocate X and Y ONCE with correct shapes

Tensor embedding_table =
    ag::layer::make_embedding_table(vocab_size, d_model);

Tensor X_cpu_init(Shape{{B, S, d_model}}, OwnTensor::TensorOptions().with_device(Device::CPU));
Tensor X_gpu_init = X_cpu_init.to(dev);
Value X = make_tensor(X_gpu_init, "X_input");

Tensor Y_cpu_init(Shape{{B, S, vocab_size}}, OwnTensor::TensorOptions().with_device(Device::CPU));
Tensor Y_gpu_init = Y_cpu_init.to(dev);
Value Y = make_tensor(Y_gpu_init, "Y_target");

// ---- Build the graph one time ----
Value logits = model(X);
Value loss = cross_entropy_with_logits(logits, Y);

// ---- Capture the complete graph ----
zero_val(loss);  // clears activations AFTER graph construction
std::vector<std::vector<int>> batch_sequences;
std::vector<std::vector<int>> target_sequences;



    for (int epoch = 0; epoch < epochs; ++epoch) {

    // Prepare batch_sequences & targets here...

    // ─────────────────────────────────────────────
    // Overwrite X directly (NO new tensors!)
    // ─────────────────────────────────────────────
    Tensor X_cpu_new = ag::layer::embed_tokens_3d(embedding_table, batch_sequences);
    X.node->value = (X_cpu_new.to(dev));


    // ─────────────────────────────────────────────
    // Overwrite Y (one-hot)
    // ─────────────────────────────────────────────
    Tensor Y_cpu_new(Shape{{B, S, vocab_size}}, OwnTensor::TensorOptions().with_device(Device::CPU));
    float* yp = Y_cpu_new.data<float>();
    std::fill(yp, yp + B*S*vocab_size, 0.0f);

    for (int b = 0; b < B; ++b)
        for (int s = 0; s < S; ++s)
            yp[(b*S + s)*vocab_size + target_sequences[b][s]] = 1.f;

    Y.node->value = (Y_cpu_new.to(dev));

    // ─────────────────────────────────────────────
    // TRAIN ONE ITERATION (GRAPH REPLAY)
    // ─────────────────────────────────────────────
    zero_val(loss);
    forward(loss);
    backward(loss);
    opti.SGD(loss, lr);
    opti.epoch();

    // Read loss
    ag::disten(loss, Device::CPU);
    float l = loss.val().data<float>()[0];
    std::cout << "Epoch " << epoch << " Loss: " << l << "\n";
}

    
    // // -----------------------------
    // // Autoregressive Generation
    // // -----------------------------
    // std::cout << "\n========================================\n";
    // std::cout << "--- GENERATING TEXT ---\n";
    // std::cout << "========================================\n\n";

    // int start_token = 32;  // Start token (e.g., space)
    // int token = start_token;
    // std::string generated;
    // bool printed_debug = false;
    // const int GEN_STEPS = 200;

    // for (int step = 0; step < GEN_STEPS; ++step) {
    //     std::vector<int> one = { token };
    //     Tensor x_cpu_flat = embed_tokens(one, embedding_table); // [1, d_model]

    //     Tensor x_cpu(
    //         Shape{{1, 1, d_model}},   
    //         OwnTensor::TensorOptions().with_device(Device::CPU)
    //     );

    //     std::memcpy(
    //         x_cpu.data<float>(),
    //         x_cpu_flat.data<float>(),
    //         d_model * sizeof(float)
    //     );
    //     Tensor x_dev = x_cpu.to(dev);
    //     Value X_in = make_tensor(x_dev, "infer_tok");

    //     // Forward pass to get logits
    //     Value out_val = model(X_in);
    //     ag::disten(out_val, Device::CPU);
    //     Tensor logits_cpu = out_val.val(); // now on CPU

    //     // Sample the next token from the logits
    //     int next_tok = safe_sample_from_logits_tensor(logits_cpu, 1.0f);

    //     // Add the next token to the generated string
    //     token = next_tok;
    //     if (token >= 32 && token < 127) {
    //         generated.push_back(static_cast<char>(token));
    //     } else {
    //         generated.push_back('?');
    //     }
    // }

    // std::cout << "---- GENERATED TEXT ----\n" << generated << "\n\n";

    // Cleanup for heap-allocated Layers
    for (auto* layer : model.get_layers()) {
        delete layer;
    }

    std::cout << "\n✅ Tiny LLaMA-style autoregressive training finished.\n";
    return 0;
}

