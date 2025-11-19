#pragma once
#include "layer/affine.hpp"

#include "ad/graph.hpp"
#include "ad/ops.hpp"
#include <vector>

namespace ag::layer {


class RMSNorm : public Layer {
public:
    RMSNorm(Device dev = Device::CPU);
    Value operator()(Value input) override;
private:
    float gamma = 1.0;
};


// class Embedding : public ag::layer::Layer {
// public:
//     Embedding(int vocab_size, int d_model, Device dev = Device::CPU)
//         : vocab_size_(vocab_size), d_model_(d_model) 
//     {
//         auto opts = OwnTensor::TensorOptions()
//                         .with_device(dev)
//                         .with_req_grad(true);

//         Tensor w = OwnTensor::Tensor::randn(
//             Shape{{vocab_size, d_model}}, opts
//         );

//         W_ = make_tensor(w, "embedding_weight");
//         params_.push_back(W_);
//     }

//     Value operator()(Value token_ids) override {
//         // token_ids: [B], stored as float or int — doesn't matter

//         Tensor ids = token_ids.val();  // [B]
//         auto sh = ids.shape().dims;
//         int B = sh[0];

//         // -------- SAFE ONE-HOT CONSTRUCTION ----------
//         // one_hot[b, i] = (ids[b] == i) ? 1 : 0
//         Tensor one_hot = OwnTensor::Tensor::zeros(
//             Shape{{B, vocab_size_}},
//             OwnTensor::TensorOptions()
//                 .with_device(ids.device())
//                 .with_req_grad(false)
//         );

//         float* ohp = one_hot.data<float>();
//         Tensor ids_cpu = ids.to(OwnTensor::Device::CPU);
//         float* idp = ids_cpu.data<float>();

//         for (int b = 0; b < B; ++b) {
//             int tok = int(idp[b]);
//             ohp[b * vocab_size_ + tok] = 1.0f;
//         }

//         if (ids.device().device == OwnTensor::Device::CUDA) {
//             one_hot = one_hot.to(OwnTensor::Device::CUDA);
//         }

//         // ------------ EMBEDDING = one_hot @ W ------------
//         Value OH = make_tensor(one_hot);
//         Value out = matmul(OH, W_);

//         return out;  // [B, d_model]
//     }

// private:
//     int vocab_size_;
//     int d_model_;
//     Value W_;
// };




} // namespace ag::nn