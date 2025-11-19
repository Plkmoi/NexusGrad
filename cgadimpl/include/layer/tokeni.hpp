#include <fstream>
#include <vector>
#include <string>
#include <iostream>
namespace ag::layer {
// ----------------------------------------------------
// Read entire file into std::string
// ----------------------------------------------------
std::string load_text_file(const std::string& path) ;

// ----------------------------------------------------
// Byte-level tokenize: each char → unsigned value 0..255
// ----------------------------------------------------
std::vector<int> tokenize_bytes(const std::string& text) ;

// ----------------------------------------------------
// Example usage
// ----------------------------------------------------
// int main() {
//     try {
//         std::string path = "corpus.txt";     // change to your file
//         std::string text = load_text_file(path);

//         std::vector<int> tokens = tokenize_bytes(text);

//         std::cout << "Loaded " << text.size() << " bytes.\n";
//         std::cout << "Tokenized into " << tokens.size() << " tokens.\n";

//         std::cout << "First 20 tokens: ";
//         for (int i = 0; i < 20 && i < (int)tokens.size(); i++)
//             std::cout << tokens[i] << " ";
//         std::cout << "\n";

//     } catch (const std::exception& e) {
//         std::cerr << e.what() << "\n";
//     }

//     return 0;
// }

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

// Place this function somewhere before main()
Tensor make_shifted_one_hot_targets(const std::vector<int>& batch, int vocab_size, Device dev) {
    const int B = batch.size();

    // Create a CPU tensor first
    Tensor Y_cpu(
        Shape{{B, vocab_size}},
        OwnTensor::TensorOptions().with_device(Device::CPU)
    );
    float* yp = Y_cpu.data<float>();
    // Zero out the memory
    std::memset(yp, 0, sizeof(float) * B * vocab_size);

    // Create the "shifted" one-hot labels
    // The prediction for token i should be token i+1
    for (int i = 0; i < B - 1; ++i) {
        int target_token_id = batch[i + 1];
        if (target_token_id >= 0 && target_token_id < vocab_size) {
            yp[i * vocab_size + target_token_id] = 1.0f;
        }
    }
    // Note: The last token in the batch has no target; its row in Y is all zeros.

    // Move to the target device and return
    return Y_cpu.to(dev);
}

}