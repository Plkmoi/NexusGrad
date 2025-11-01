// =============================================
// cgadimpl/src/core/autodiff.cpp
// =============================================
#include <unordered_map>
#include <stdexcept>
#include "ad/autodiff.hpp"
#include "ad/detail/autodiff_ops.hpp"
#include "ad/debug.hpp"
#include <ad/checkpoint.hpp>
#include "json.hpp"// or nlohmann::json
#include <iomanip>
#include <fstream>
#include <sstream>

namespace ag {

void zero_grad(const Value& root){
    auto order = topo_from(root.node.get());
    for (Node* n : order) {if (n->requires_grad){ n->grad = Tensor::zeros_like(n->value);

        if(n->op==Op::Sub)
        {
            n->op=Op::Add;
            n->value = ag::detail::add_nodeops(n->inputs[0],n->inputs[1])->value;


        }

        //         if(n->op==Op::Add)
        // {

        //     if(n->inputs[0]->op==Op::MatMul)
        // {
        //     n->op=Op::FMA;
        //     n->value = ag::detail::fmab_nodeops(n->inputs[0]->inputs[0], n->inputs[0]->inputs[1],n->inputs[1])->value;


        // }
        // else if(n->inputs[1]->op==Op::MatMul)
        //     {
        //     n->value = ag::detail::fmab_nodeops(n->inputs[1]->inputs[0], n->inputs[1]->inputs[1],n->inputs[0])->value;
        //     n->inputs = {}

        //     }

        // }
    
    std::cout << "Captured Node: " << op_name(n->op)
          << " | shape: [" << n->value.rows() << ", " << n->value.cols() << "]"
          << " | device: " << (n->value.is_cuda() ? "CUDA" : "CPU") << "\n";
    
    
    }};
    
    
}

void backward(const Value& root, const Tensor* grad_seed){
    auto order = topo_from(root.node.get());

    // seed
    if (root.node->requires_grad) {
        root.node->grad = grad_seed ? *grad_seed
                                    : (root.node->value.size()==1 ? Tensor::ones(1,1)
                                                                  : Tensor::ones_like(root.node->value));
    }

    // reverse topo
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        Node* n = *it;
if(n->value.is_cuda()) n->requires_cuda=true;

        if (!n->requires_grad) continue;
        const Tensor& gy = n->grad;
        ag::debug::on_backprop_step(n, gy); // (optional) prints one line per node

        if (n->is_checkpoint && n->value.size() == 0) {
        if (!ag::checkpoint_impl::recompute_subgraph(n->shared_from_this())) {
            throw std::runtime_error("autodiff: failed to recompute checkpointed node during backward");
        }
        }
        VjpFn fn = vjp_lookup(n->op);
        if (fn) fn(n, gy); // handler accumulates into parents

    }
}



void valsend(const Value& root) {
    auto order = topo_from(root.node.get());
    cudaDeviceSynchronize(); // Wait for all GPU ops to finish

    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        Node* n = *it;
if(n->value.is_cuda()) n->requires_cuda=true;
        if (!n->requires_grad || !(n->value.is_cuda())) continue;
        std::cout << "(" << n->value.numel() << ")\n";

        if (n->value.numel()> 0) {
            n->value = n->value.to(Device::CPU);
            
        }

        std::cout << "[CUDA VALSEND output]: ";
        for (int i = 0; i < 10; ++i)
          {  std::cout << n->value.data()[i] << " ";
        std::cout << "(" << n->debug_name << ")\n";}
                    std::cout<<n->value.is_cuda()<<"           "<<"sfbsb"<<"  ";

    }
}

void grasend(const Value& root) {
    auto order = topo_from(root.node.get());
    cudaDeviceSynchronize(); // Wait for all GPU ops to finish

    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        Node* n = *it;
        if(n->value.is_cuda()) n->requires_cuda=true;

        if (!n->requires_grad || !(n->requires_cuda)) continue;
        std::cout << "(" << n->value.numel() << ")\n";

        if (n->grad.numel()> 0) {
            n->grad = n->grad.to(Device::CPU);
        }

        std::cout << "[CUDA GRASEND output]: ";
        for (int i = 0; i < 10; ++i)
          {  std::cout << n->grad.data()[i] << " ";
        std::cout << "(" << n->debug_name << ")\n";}
    }
}

void unisend(const Value& root)
{
    auto order = topo_from(root.node.get());
    cudaDeviceSynchronize(); // Wait for all GPU ops to finish

    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        Node* n = *it;
        
        if (!n->requires_grad || !(n->grad.is_cuda()) || !(n->value.is_cuda())) continue;
        std::cout << "\n(" << n->value.numel() << ")\n";

        if (n->grad.numel()> 0 || n->value.numel()> 0) {
            n->requires_cuda = true;
            n->value =             n->value.to(Device::CPU);
            n->grad = n->grad.to(Device::CPU);

        }


    }
    
}

Tensor jvp(const Value& root, const std::unordered_map<Node*, Tensor>& seed){
    if (!root.node) return Tensor{};
    auto order = topo_from(root.node.get());
    std::unordered_map<Node*, Tensor> T;
    T.reserve(order.size());

    auto tangent_of = [&](Node* p) -> const Tensor& {
        auto it = T.find(p);
        if (it != T.end()) return it->second;
        static Tensor Z; // fallback; shouldn't be used for leaves without seeds
        return Z;
    };

    for (Node* n : order) {
        // seed tangent for this node (if provided), else zeros
        Tensor t = Tensor::zeros_like(n->value);
        if (auto it = seed.find(n); it != seed.end()) t = it->second;

        ag::debug::on_jvp_step(n); // (optional) prints forward-mode step

        JvpFn fn = jvp_lookup(n->op);
        if (fn) t = fn(n, tangent_of);

        T[n] = t;
    }
    return T[root.node.get()];
}

// Tensor forward_pass(const Value& root) {
//     if (!root.node) return Tensor{};
//     auto order = topo_from(root.node.get());
    
//     for (Node* n : order) {
//         if (n->op == Op::Leaf) continue; // already has value
//         auto fn = (n->op);
//         if (fn) {
//             n->value = fn(n);
//         } else {
//             throw std::runtime_error("No forward fn for op " + std::to_string((int)n->op));
//         }
//     }
//     return root.node->value;
// }




void save_safetensors(const std::unordered_map<std::string, Tensor>& tensors,
                      const std::string& filename)
{
    nlohmann::json header;
    size_t offset = 0;
    std::vector<char> binary;

    // Step 1: build header and binary data
    for (const auto& [name, tensor] : tensors) {
        size_t num_bytes = tensor.numel() * sizeof(float);
        auto cpu_data = tensor.to(Device::CPU);

        header[name] = {
            {"dtype", "F32"},
            {"shape", {tensor.rows(), tensor.cols()}},
            {"data_offsets", {offset, offset + num_bytes}}
        };

        binary.insert(binary.end(),
                      reinterpret_cast<char*>(cpu_data.data()),
                      reinterpret_cast<char*>(cpu_data.data()) + num_bytes);

        offset += num_bytes;
    }

    // Step 2: convert header to JSON
    std::string header_str = header.dump();
    uint64_t header_len = header_str.size();

    // Step 3: write file
    std::ofstream out(filename, std::ios::binary);
    out.write(reinterpret_cast<const char*>(&header_len), sizeof(uint64_t));
    out.write(header_str.data(), header_len);
    out.write(binary.data(), binary.size());
    out.close();
}

void save_all_values_and_grads(const Value& root) {
    auto order = topo_from(root.node.get());
    std::unordered_map<std::string, Tensor> tensors;

    for (Node* n : order) {
        if (n->requires_grad) {
            std::string name = op_name(n->op);
            tensors[name + ".value"] = n->value;
            tensors[name + ".grad"] = n->grad;
        }
    }

    save_safetensors(tensors, "cgadimpl/tests/graph_dump.safetensors");
    std::cout << "💾 Saved computation graph to graph_dump.safetensors ✅\n";
}
























} // namespace ag
