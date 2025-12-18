#include "layer/archlist.hpp"
#include <iostream>
#include <vector>
#include <iomanip>
#include <random>
#include <optim.hpp>




int main() {
    using namespace flow;
    using namespace OwnTensor;

    std::cout << "========================================\n";
    std::cout << "--- Tiny LLaMA-Style Autoregressive ---\n";
    std::cout << "========================================\n\n";

    // 1. --- Hyperparameters ---
    const int B = 4;            // batch size
    // const int vocab_size = 5000; // number of classes (logits dim)
    const int num_layers = 3;    // (Attn + SWIGLU) block pairs
    const float lr = 0.0000007f;
    const int epochs = 1100;
    int vocab_size = 15000;      // integer tokens 0..19
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
flow::Tokenizer tok;

std::string text =
    flow::load_text_file("/home/blubridge-034/Downloads/Newf/cgadimpl/arch/src/layer/corpus.txt");

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


    std::vector<flow::Layer*> layers;
    layers.reserve(num_layers * 2 + 2);

    // Build model layers
    for (int i = 0; i < num_layers; ++i) {
        layers.push_back(new flow::ResidualBlock({
            new flow::RMSNorm(),
            new flow::AlibiAttention(B, S, d_model, Heads, dev)
        }));

        layers.push_back(new flow::ResidualBlock({
            new flow::RMSNorm(),
            new flow::SWIGLU(B, S, d_model, K, dev)
            // new flow::Mish()
        }));
    }

    layers.push_back(new flow::RMSNorm());
    layers.push_back(new flow::Linear(B, vocab_size, d_model, dev));

    flow::Traverse model(layers);

    std::cout << "Model created with " << model.parameters().size()
              << " parameter tensors.\n\n";


Tensor embedding_table =
    flow::make_embedding_table(vocab_size, d_model);

Tensor X_cpu_init(Shape{{B, S, d_model}}, OwnTensor::TensorOptions().with_device(Device::CPU));
Tensor X_gpu_init = X_cpu_init.to(dev);
Value X = make_tensor(X_gpu_init, "X_input");

Tensor Y_cpu_init(Shape{{B, S, vocab_size}}, OwnTensor::TensorOptions().with_device(Device::CPU));
Tensor Y_gpu_init = Y_cpu_init.to(dev);
Value Y = make_tensor(Y_gpu_init, "Y_target");

// ---- Build the graph one time ----
Value logits = model(X);
Value loss = cross_entropy_with_logits(logits, Y);
flow::opti.SGD(loss, lr);

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
    Tensor X_cpu_new = flow::embed_tokens_3d(embedding_table, batch_sequences);
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
    flow::opti.epoch();
    auto ll = loss.val().to(Device::CPU);

    // ag::disten(loss, Device::CPU);
    float l = ll.data<float>()[0];
    std::cout << "Epoch " << epoch << " Loss: " << l << "\n";
}


// ---------------------------------------------------------
// Autoregressive Text Generation (BPE-correct)
// ---------------------------------------------------------
std::cout << "\n========================================\n";
std::cout << "--- GENERATING TEXT (BPE) ---\n";
std::cout << "========================================\n\n";

const int GEN_STEPS = 200;

// 1. We use a vector to keep track of EVERY token generated so far
std::vector<int> history_tokens; 
int start_token = tok.encode(" ")[0];
history_tokens.push_back(start_token);


for (int step = 0; step < GEN_STEPS; ++step)
{
    // -------------------------------------------------------
    // 1. Embed the FULL history, not just the last token
    // -------------------------------------------------------
    // This produces a tensor of shape [Current_T, d_model]
    int T = history_tokens.size();
    // std::cout<<flow::embed_tokens(embedding_table, history_tokens).shape().dims[0]<<"   "<<flow::embed_tokens(embedding_table, history_tokens).shape().dims[1]<<"\n";
    Tensor emb_seq = flow::embed_tokens(embedding_table, history_tokens).unflatten(0, Shape({1, T}));

    // 2. Reshape to [Batch=1, SeqLen=T, Dim=D]
    // int T = history_tokens.size();
    // Tensor x_dev = emb_seq.unflatten(1, Shape({1, T, d_model})).to(dev);
    // std::cout<<"Begin";

    // Move to device (GPU/NPU)
    Tensor x_dev = emb_seq.clone().to(dev);
    Value X_in = make_tensor(x_dev, "infer_seq");

    // -------------------------------------------------------
    // 3. Forward pass
    // -------------------------------------------------------
    Value out_val = model(X_in); // Output shape: [1, T, Vocab_Size]
    // std::cout<<"End";

    // Bring back to CPU
    // ag::disten(out_val, Device::CPU);
    Tensor logits_all = out_val.val().to(Device::CPU); 

    // -------------------------------------------------------
    // 4. Slice to get ONLY the last position's logits
    // -------------------------------------------------------
    // The model predicts a next token for EVERY position in history.
    // We only care about what comes after the VERY LAST token.
    int V = logits_all.shape().dims[2]; // Vocab size
    
    // Create a view or copy of just the last vector in the T dimension
    // Pointer math: Start of tensor + (Last_Index * Vocab_Size)
    float* last_token_logits_ptr = &logits_all.data<float>()[(T - 1) * V];
    
    // Wrap this pointer back into a 1D tensor for your sample function
    Tensor last_logits(Shape({1, 1, V}), false); 
    std::memcpy(last_logits.data<float>(), last_token_logits_ptr, V * sizeof(float));

    // -------------------------------------------------------
    // 5. Sample and Append
    // -------------------------------------------------------
    int next_tok = flow::safe_sample_from_logits_tensor(last_logits, 1.0f);

    history_tokens.push_back(next_tok); // <--- THIS is the "memory"
    
    // Optional: print token immediately to see text forming
    std::cout << tok.decode({(uint32_t)next_tok}) << std::flush;
}
std::vector<uint32_t> histi_tokens;

histi_tokens.reserve(history_tokens.size()); // Optional but recommended for efficiency

std::transform(history_tokens.begin(), history_tokens.end(), std::back_inserter(histi_tokens),
               [](int value) { return static_cast<uint32_t>(value); });

// ---------------------------------------------------------
// Decode full sequence using tokenizer
// ---------------------------------------------------------
std::string decoded = tok.decode(histi_tokens);

std::cout << "Generated:\n" << decoded << "\n\n";


    // Cleanup for heap-allocated Layers
    for (auto* layer : model.get_layers()) {
        delete layer;
    }

    std::cout << "\n✅ Tiny LLaMA-style autoregressive training finished.\n";
    return 0;
}