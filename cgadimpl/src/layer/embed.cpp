#include "layer/embed.hpp"
#include <stdexcept>
#include <sstream>
#include "ad/ag_all.hpp"

namespace ag::layer {

// -------------------------------------------
// Create embedding table
// -------------------------------------------
Tensor make_embedding_table(int vocab_size, int dim) {
    return Tensor(Shape({vocab_size, dim}), true);  // requires_grad = true
}



// -------------------------------------------
// 1D embedding (sequence → T × D)
// -------------------------------------------
// -------------------------------------------
// 1D embedding (sequence → T × D)
// -------------------------------------------
Tensor embed_tokens(const Tensor& W, const std::vector<int>& tokens) {
    int T = tokens.size();
    int V = W.shape().dims[0];
    int D = W.shape().dims[1];

    Tensor out(Shape({T, D}), false);

    for (int t = 0; t < T; ++t) {
        int id = tokens[t];

        // ---- lol bad token fix ----
        if (id < 0 || id >= V) {
            std::cerr << "[embed_tokens] BAD TOKEN LOL: "
                      << id << " → clamped to 0\n";
            id = 0;        // <-- safe pad token
        }

        float* orow = &out.data<float>()[t * D];
        const float* wrow = &W.data<float>()[id * D];


        for (int d = 0; d < D; ++d)
            orow[d] = wrow[d];
    }

    return out;
}


// -------------------------------------------
// 3D embedding (batch → B × T × D)
// -------------------------------------------
// -------------------------------------------
// 3D embedding (batch → B × T × D)
// -------------------------------------------
Tensor embed_tokens_3d(
        const Tensor& W,
        const std::vector<std::vector<int>>& batch_tokens)
{
    int B = batch_tokens.size();
    if (B == 0)
        throw std::runtime_error("embed_tokens_3d: empty batch!");

    int T = batch_tokens[0].size();
    if (T == 0)
        throw std::runtime_error("embed_tokens_3d: empty sequence!");

    // Validate all sequences have equal length
    for (int b = 0; b < B; ++b) {
        if ((int)batch_tokens[b].size() != T) {
            std::stringstream ss;
            ss << "embed_tokens_3d: inconsistent sequence lengths: "
               << "batch[0] = " << T << ", batch[" << b
               << "] = " << batch_tokens[b].size();
            throw std::runtime_error(ss.str());
        }
    }

    int V = W.shape().dims[0];
    int D = W.shape().dims[1];

    Tensor out(Shape({B, T, D}), false);
    float* odata = out.data<float>();
    const float* wdata = W.data<float>();

    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {

            int id = batch_tokens[b][t];

            // ---- lol bad token fix ----
            if (id < 0 || id >= V) {
                std::cerr << "[embed_tokens_3d] BAD TOKEN LOL: "
                          << id << " at (b=" << b << ", t=" << t
                          << ") → clamped to 0\n";
                id = 0;   // <- pad token avoids INF
            }

            float* orow = &odata[(b*T + t) * D];
            const float* wrow = &wdata[id * D];

            for (int d = 0; d < D; ++d)
                orow[d] = wrow[d];
        }
    }

    return out;
}


} // namespace ag::layer
