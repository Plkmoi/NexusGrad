#include "ad/ag_all.hpp"   // Main umbrella header for the framework
#include <iostream>
#include <vector>
#include <iomanip>
#include <random>
#include <optim.hpp>
#include "layer/tokeni.hpp"
#include "layer/embed.hpp"
#include "layer/tokenus.hpp"



int main() {
    using namespace ag;
    using namespace OwnTensor;

    std::cout << "========================================\n";
    std::cout << "--- Tiny LLaMA-Style Autoregressive ---\n";
    std::cout << "========================================\n\n";

    // 1. --- Hyperparameters ---
    const int B = 4;            // batch size
    // const int vocab_size = 5000; // number of classes (logits dim)
    const int num_layers = 12;    // (Attn + SWIGLU) block pairs
    const float lr = 0.00000001f;
    const int epochs = 1000;
    int vocab_size = 50000;       // integer tokens 0..19
    int Heads = 12;

    const int S = 512; // Sequence length (needs to be defined)
    const int d_model = 768; // Embedding dimension
    auto dev = Device::CUDA;
    int K = 5;

    // -------------------------------------------
    // Load corpus + tokenize
    // -------------------------------------------


// -------------------------------------------
// Load corpus + train tokenizer
// -------------------------------------------
ag::layer::Tokenizer tok;

std::string text =
    ag::layer::load_text_file("/home/blubridge-034/Downloads/Newf/cgadimpl/cgadimpl/src/layer/corpus.txt");

std::vector<std::string> train_samples = { text };

// Train tokenizer to target vocab size
tok.train(train_samples, vocab_size);

// -------------------------------------------
// Encode the entire corpus
// -------------------------------------------
std::vector<uint32_t> all_tokens_u32 = tok.encode(text);
std::vector<int> all_tokens(all_tokens_u32.begin(), all_tokens_u32.end());

// Adjust vocab_size to include merges
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
opti.SGDm(loss, lr);

// ---- Capture the complete graph ----
zero_val(loss);  // clears activations AFTER graph construction
std::vector<std::vector<int>> batch_sequences;
std::vector<std::vector<int>> target_sequences;



for (int epoch = 0; epoch < epochs; ++epoch) {

    // --- FIX: MAKE THE BATCH ---
    int offset = (epoch * B * S) % (all_tokens.size() - (B*S + 2));
    make_batch(batch_sequences, target_sequences, offset);

    if (batch_sequences.empty()) {
        throw std::runtime_error("make_batch returned empty batch");
    }

    // --- EMBEDDING ---
    Tensor X_cpu_new = ag::layer::embed_tokens_3d(embedding_table, batch_sequences);
    X.node->value = (X_cpu_new.to(dev));

    // --- BUILD ONE-HOT TARGET ---
    Tensor Y_cpu_new(Shape{{B, S, vocab_size}}, OwnTensor::TensorOptions().with_device(Device::CPU));
    float* yp = Y_cpu_new.data<float>();
    std::fill(yp, yp + B*S*vocab_size, 0.0f);

    for (int b = 0; b < B; ++b)
        for (int s = 0; s < S; ++s)
            yp[(b*S + s)*vocab_size + target_sequences[b][s]] = 1.f;

    Y.node->value = (Y_cpu_new.to(dev));

    // --- TRAIN ---
    forward(loss);
    zero_grad(loss);
    backward(loss);
    opti.epoch();

    ag::disten(loss, Device::CPU);
    float l = loss.val().data<float>()[0];
    std::cout << "Epoch " << epoch << " Loss: " << l << "\n";
}



// ---------------------------------------------------------
// Autoregressive Text Generation (BPE-correct)
// ---------------------------------------------------------
std::cout << "\n========================================\n";
std::cout << "--- GENERATING TEXT (BPE) ---\n";
std::cout << "========================================\n\n";

const int GEN_STEPS = 200;

// Start from BOS or an empty prompt
int token = tok.encode(" ")[0];   // or any char, must not be empty
      // or tok.encode("")[0]
std::vector<uint32_t> gen_tokens;    // store token IDs
gen_tokens.push_back(token);

// Pre-allocate CPU input tensor [1,1,d_model]
Tensor x_cpu(
    Shape{{1, 1, d_model}},
    OwnTensor::TensorOptions().with_device(Device::CPU)
);

for (int step = 0; step < GEN_STEPS; ++step)
{
    // -------------------------
    // 1. Embed current token
    // -------------------------
    std::vector<int> single_tok = { token };

    Tensor emb_row = ag::layer::embed_tokens(embedding_table, single_tok).unflatten(1, Shape({1, d_model}));
    // emb_row: [1, d_model]

    // reshape into [1,1,d_model]


    // Move to device
    Tensor x_dev = x_cpu.to(dev);
    Value X_in = make_tensor(x_dev, "infer_tok");

    // -------------------------
    // 2. Forward pass → logits
    // -------------------------
    Value out_val = model(X_in);

    // bring logits back to CPU
    ag::disten(out_val, Device::CPU);
    Tensor logits_cpu = out_val.val();   // [1,1,V]

    // -------------------------
    // 3. Sample next token
    // -------------------------
    int next_tok = ag::layer::safe_sample_from_logits_tensor(logits_cpu, 1.0f);

    // save
    gen_tokens.push_back(next_tok);

    token = next_tok;
}

// ---------------------------------------------------------
// Decode full sequence using tokenizer
// ---------------------------------------------------------
std::string decoded = tok.decode(gen_tokens);

std::cout << "Generated:\n" << decoded << "\n\n";


    // Cleanup for heap-allocated Layers
    for (auto* layer : model.get_layers()) {
        delete layer;
    }

    std::cout << "\n✅ Tiny LLaMA-style autoregressive training finished.\n";
    return 0;
}