// =====================
// file: cgadimpl/include/ag/ops.hpp (declarations only)
// =====================
#pragma once
#include "ad/graph.hpp"
#include "nodeops.hpp"
#include "ad/checkpoint.hpp"


namespace flow {

struct CheckpointOptions;


Value relu (const Value& x);

Value gcu (const Value& x);
Value mish (const Value& x);
Value gaus (const Value& x);
Value parcon(const Value& x);
Value sigmoid(const Value& x);
Value softplus(const Value& x);
Value reluatt(const Value& a, const Value& b, const Value& c, const Value& d); 
Value expand(const Value& x, int q);
Value sigatt(const Value& a, const Value& b, const Value& c, const Value& d); 


Value gelu (const Value& x); // tanh approx
Value silu (const Value& x); // x * sigmoid(x)
Value leaky_relu(const Value& x, float alpha=0.01f); // alpha via const input
Value lisht(const Value& x);
Value swiglu(const Value& x, const Value& a, const Value& b, const Value& c, const Value& d);
Value rms(const Value& x); // root mean square normalization
Value realrms(const Value& x, float g); // with learned scale
Value dyntanh(const Value& x, float a=1.0f, float b=0.0f, float g=0.5f); // dynamic tanh via mean_all
Value relaynor(const Value& x, float b, float g); // with learned scale and bias
Value mambassm(const Value& z, const Value& a, const Value& b, const Value& c, const Value& d); // state space model
Value moewe(const Value& x, const Value& w, const Value& b);


Value softmax_row(const Value& z); // [B,C] -> [B,C]
Value logsumexp_row(const Value& z); // [B,C] -> [B,1]
Value laynor(const Value& x);
Value alibiatt(const Value& a, const Value& b, const Value& c, const Value& d, int m); // m = max seq len

// composite loss (one-hot targets)
Value cross_entropy_with_logits(const Value& logits, const Value& onehot);
Value kldivergence(const Value& logits, const Value& onehot);
Value fmab(const Value& a, const Value& b, const Value& c); // fused multiply-add a@b + c
Value linear(const Value& a, const Value& b, const Value& c); // fused multiply-add a@b + c

Value attention(const Value& a, const Value& b, const Value& c, const Value& d, int H);
Value mse_loss(const Value& pred, const Value& target);
Value mae_loss(const Value& pred, const Value& target);


Tensor forward_eval_node(Node* node);


} // namespace ag