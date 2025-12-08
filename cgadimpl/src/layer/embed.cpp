#include "layer/embed.hpp"
#include <stdexcept>
#include "ad/ag_all.hpp"

namespace ag::layer {

Tensor make_embedding_table(int vocab_size, int dim) {
    return Tensor(Shape({vocab_size, dim}), true);   // true = requires_grad
}

Tensor embed_tokens(const Tensor& W, const std::vector<int>& tokens) {
    int T = tokens.size();
    int D = W.shape().dims[1];

    Tensor out(Shape({T, D}), false);

    for (int t = 0; t < T; ++t) {
        int id = tokens[t];
        if (id < 0 || id >= W.shape().dims[0])
            throw std::runtime_error("Token index out of range");

        for (int d = 0; d < D; ++d)
            out.data<float>()[t * D + d] = W.data<float>()[id * D + d];
    }

    return out;
}

Tensor embed_tokens_3d(const Tensor& W,
                       const std::vector<std::vector<int>>& batch_tokens)
{
    int B = batch_tokens.size();
    int T = batch_tokens[0].size();
    int D = W.shape().dims[1];

    Tensor out(Shape({B, T, D}), false);

    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            int id = batch_tokens[b][t];
            for (int d = 0; d < D; ++d)
                out.data<float>()[(b*T + t)*D + d] = W.data<float>()[id*D + d];
        }
    }

    return out;
}

} // namespace ag::layer