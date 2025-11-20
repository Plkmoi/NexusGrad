// #include "ad/ag_all.hpp"
// #include <iostream>
// #include <vector>
// #include <random>
// #include <iomanip>
// #include <cmath>
// #include <cstring>
// #include <optim.hpp>

// using namespace ag;
// using namespace OwnTensor;

// // =======================================================
// // 1. --- CREATE CPU EMBEDDING TABLE (vocab_size × d_model)
// // =======================================================
// Tensor make_embedding_table(int vocab_size, int d_model) {
//     Tensor emb(
//         Shape{{vocab_size, d_model}},
//         TensorOptions().with_device(Device::CPU)
//     );

//     std::mt19937 rng(42);
//     std::normal_distribution<float> dist(0.0f, 1.0f / std::sqrt(d_model));

//     float* p = emb.data<float>();
//     for (int i = 0; i < vocab_size * d_model; ++i)
//         p[i] = dist(rng);

//     return emb;
// }


// // =======================================================
// // 2. --- SEQUENCE EMBEDDING + POSITIONAL ENCODING
// // tokens: [B, T] integers
// // Return X : [B, T, d_model]
// // =======================================================
// Tensor embed_sequence(
//     const std::vector<std::vector<int>>& tokens,  // [B][T]
//     const Tensor& emb_table                     // [vocab, d_model]
// ) {
//     int B = tokens.size();
//     int T = tokens[0].size();
//     int vocab = emb_table.shape().dims[0];
//     int d_model = emb_table.shape().dims[1];

//     Tensor X(
//         Shape{{B, T, d_model}},
//         TensorOptions().with_device(Device::CPU)
//     );

//     float* out = X.data<float>();
//     const float* E = emb_table.data<float>();

//     for (int b = 0; b < B; ++b) {
//         for (int t = 0; t < T; ++t) {
//             int id = tokens[b][t];
//             if (id < 0 || id >= vocab) id = 0;  // fallback
//             std::memcpy(
//                 out + (b*T + t) * d_model,
//                 E   + id * d_model,
//                 d_model * sizeof(float)
//             );
//         }
//     }

//     return X;
// }


// // =======================================================
// // 3. --- ADD SINE-COSINE POSITIONAL ENCODING (CPU)
// // =======================================================
// void add_positional_encoding(Tensor& X) {
//     int B = X.shape().dims[0];
//     int T = X.shape().dims[1];
//     int d = X.shape().dims[2];

//     float* x = X.data<float>();

//     for (int t = 0; t < T; ++t) {
//         for (int i = 0; i < d; i += 2) {
//             float angle = t / std::pow(10000.f, (i / (float)d));

//             float sinv = std::sin(angle);
//             float cosv = std::cos(angle);

//             for (int b = 0; b < B; ++b) {
//                 int idx = (b*T + t) * d + i;
//                 x[idx]     += sinv;
//                 if (i+1 < d)
//                     x[idx+1] += cosv;
//             }
//         }
//     }
// }


// // =======================================================
// //            TINY GPT with ag::layer blocks
// // =======================================================
// //
// // Input : X [B, T, d_model]
// // Model :
// //   embed + pos →
// //   For N layers:
// //      X = Residual(RMSNorm → Attention → proj)
// //      X = Residual(RMSNorm → SWIGLU → Linear)
// //   logits = Linear([B*T, d_model] → vocab)
// // =======================================================


// int main() {
//     std::cout << "=========================\n";
//     std::cout << "--- Tiny GPT Training ---\n";
//     std::cout << "=========================\n\n";

//     Device dev = Device::CUDA;

//     // ----------------------------
//     // Hyperparameters
//     // ----------------------------
//     const int B = 16;             // batch size
//     const int T = 32;             // sequence length
//     const int vocab = 1000;       // fake vocab size
//     const int d_model = 128;      // hidden width
//     const int n_layers = 2;       // transformer blocks
//     const float lr = 0.0005f;
//     const int epochs = 200;

//     // ---------------------------------------------
//     // Step 1: Create CPU embedding table
//     // ---------------------------------------------
//     Tensor emb_table = make_embedding_table(vocab, d_model);

//     // ---------------------------------------------
//     // Step 2: Generate fake token sequences [B,T]
//     // ---------------------------------------------
//     std::vector<std::vector<int>> tokens(B, std::vector<int>(T));
//     std::mt19937 g(42);
//     std::uniform_int_distribution<int> pick(0, vocab-1);

//     for (int b = 0; b < B; ++b)
//         for (int t = 0; t < T; ++t)
//             tokens[b][t] = pick(g);

//     // ---------------------------------------------
//     // Step 3: Embed and add positional encoding
//     // ---------------------------------------------
//     Tensor Xcpu = embed_sequence(tokens, emb_table);
//     add_positional_encoding(Xcpu);

//     Tensor Xgpu = Xcpu.to(dev);
//     Value X = make_tensor(Xgpu, "input_tokens");

//     // ---------------------------------------------
//     // Step 4: Build GPT using ag::layer
//     // ---------------------------------------------
//     using namespace ag::layer;

//     std::vector<Layer*> layers;
//     layers.reserve(n_layers * 2 + 1);

//     // N transformer blocks
//     for (int i = 0; i < n_layers; ++i) {
//         layers.push_back(new ResidualBlock({
//             new RMSNorm(),
//             new Attention(d_model, d_model, dev)
//         }));
//         layers.push_back(new ResidualBlock({
//             new RMSNorm(),
//             new SWIGLU(d_model, d_model, dev)
//         }));
//     }

//     // Final RMSNorm and LM Head → logits over vocab
//     layers.push_back(new RMSNorm());
//     layers.push_back(new Linear(d_model, vocab, dev));

//     Traverse model(layers);

//     std::cout << "Model created with "
//               << model.parameters().size()
//               << " trainable tensors.\n\n";

//     // ---------------------------------------------
//     // Step 5: Create fake next-token labels Y
//     // ---------------------------------------------
//     Tensor Yt(Shape{{B*T, vocab}}, TensorOptions());
//     float* yp = Yt.data<float>();
//     std::uniform_int_distribution<int> cpick(0, vocab-1);

//     for (int i = 0; i < B*T; ++i) {
//         int c = cpick(g);
//         for (int j = 0; j < vocab; ++j)
//             yp[i*vocab + j] = (j == c) ? 1.0f : 0.0f;
//     }
//     Value Y = make_tensor(Yt, "target");
//     disten(Y, dev);

//     // ---------------------------------------------
//     // Step 6: Training loop
//     // ---------------------------------------------
//     for (int e = 0; e < epochs; ++e) {
//         Value logits = model(X);                    // [B*T, vocab]
//         Value loss = cross_entropy_with_logits(logits, Y);

//         disten(loss, Device::CPU);
//         float lv = loss.val().data<float>()[0];

//         std::cout << "Epoch "
//                   << std::setw(3) << e
//                   << "  Loss: " << lv << "\n";

//         disten(loss, dev);
//         model.zero_grad();
//         backward(loss);
//         SGD(loss, nullptr, lr);
//     }

//     // Cleanup
//     for (Layer* L : model.get_layers()) delete L;

//     std::cout << "\nTraining complete.\n";
//     return 0;
// }
