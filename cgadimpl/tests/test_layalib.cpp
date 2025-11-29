#include "ad/ag_all.hpp"
#include <iostream>
#include "layer/affine.hpp"
#include "layer/attention.hpp"



int main() {
    using namespace ag;
    using namespace OwnTensor;

    Device dev = Device::CUDA;
    const int seqlen = 64;
    const int d_model = 16;
    const int n_heads = 4;
    const int batch = 2;

    Tensor x_t = Tensor::randn(
        Shape{{batch, seqlen, d_model}},
        TensorOptions().with_device(dev).with_req_grad(false)
    );
    Value x = make_tensor(x_t, "x_in");

   auto qlayer = ag::layer::AlibiAttention(batch, seqlen, d_model, n_heads, dev);

    Value y = qlayer(x);   // [T, d_model]
    Tensor y_cpu = y.val().to(Device::CPU);

    // Check for NaNs
    const auto& dims = y_cpu.shape().dims;
    std::size_t n = 1;
    for (auto d : dims) n *= static_cast<std::size_t>(d);

    const float* data = y_cpu.data<float>();
    bool bad = false;
    float max_abs = 0.f;
    for (std::size_t i = 0; i < n; ++i) {
        float v = data[i];
        if (std::isnan(v) || std::isinf(v)) bad = true;
        max_abs = std::max(max_abs, std::fabs(v));
    }

    std::cout << "Output shape: [" << dims[0] << ", " << dims[1] << "]\n";
    std::cout << (bad ? "⚠️ has NaN/Inf\n" : "✅ finite\n");
    std::cout << "max |y| = " << max_abs << "\n";

    

    // Optionally test backward:
    Value sum_y = ag::sum(y);  // if you have a Value-level sum, else wrap a tensor sum
    backward(sum_y);
    // Inspect att_layer.params()[0].node->grad...

    return 0;
}
