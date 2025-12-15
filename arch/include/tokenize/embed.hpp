#pragma once
#include <vector>
#include "layer/archlist.hpp"    // your CGAD class

namespace flow {

// Make embedding weight table
Tensor make_embedding_table(int vocab_size, int dim);

// Embed a batch of sequences: [B, T] → [B, T, D]
Tensor embed_tokens_3d(const Tensor& W, const std::vector<std::vector<int>>& tokens);

// Embed a single sequence: [T] → [T, D]
Tensor embed_tokens(const Tensor& W, const std::vector<int>& tokens);

}