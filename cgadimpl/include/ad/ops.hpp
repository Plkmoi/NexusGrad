// =====================
// file: cgadimpl/include/ag/ops.hpp (declarations only)
// =====================
#pragma once
#include "ad/graph.hpp"
#include "ad/nodeops.hpp"
#include "ad/checkpoint.hpp"


namespace ag {

struct CheckpointOptions;

Value checkpoint(const Value &v, const CheckpointOptions &opts);
Value inplace_checkpoint(const Value& v);
Value add (const Value& a, const Value& b);
Value sub (const Value& a, const Value& b);
Value mul (const Value& a, const Value& b);
Value div (const Value& a, const Value& b);

Value matmul(const Value& a, const Value& b);
Value sum (const Value& x);
Value flomul (const Value& a, float b);
Value floadd (const Value& a, float b);
Value flodiv (const Value& a, float b);

inline Value operator+(const Value& a, const Value& b){ return add(a,b);}
inline Value operator-(const Value& a, const Value& b){ return sub(a,b);}
inline Value operator*(const Value& a, const Value& b){ return mul(a,b);}
inline Value operator/(const Value& a, const Value& b){ return div(a,b);}
inline Value operator*(const Value& a, float b){ return flomul(a,b);}
inline Value operator*( float b, const Value& a){ return flomul(a,b);}
inline Value operator/( float b, const Value& a){ return flodiv(a, b);}
inline Value operator+( float b, const Value& a){ return floadd(a, b);}
inline Value operator+( const Value& a, float b){ return floadd(a, b);}

// unary elementwise
Value exp (const Value& x);
Value log (const Value& x);
Value tanh (const Value& x);

Value transpose(const Value& x);
Value sign (const Value& a, const Value& b);

// rowwise reductions / softmax family
Value rowsum (const Value& x); // [B,C] -> [B,1]
Value rowmax (const Value& x); // [B,C] -> [B,1]
Value mean_all(const Value& x); // scalar

// composite loss (one-hot targets)

Tensor forward_eval_node(Node* node);


} // namespace ag