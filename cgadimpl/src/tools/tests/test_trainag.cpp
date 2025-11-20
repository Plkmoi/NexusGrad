#include <iostream>
#include "ad/ag_all.hpp"
#include "optim.hpp"
#include <iomanip>
using namespace ag;

int main() {
    using std::cout; using std::fixed; using std::setprecision;

    // --- Params (learnable) ---
    Tensor A_tensor   = Tensor::randn(Shape{{2, 3}}, TensorOptions().with_req_grad(true));
    Tensor B_tensor   = Tensor::randn(Shape{{3, 2}}, TensorOptions().with_req_grad(true));
    Tensor Bias_tensor= Tensor::zeros(Shape{{1, 2}}, TensorOptions().with_req_grad(true));

    // --- Targets: two rows of class probs (must be >0 and sum to 1) ---
    Tensor Yt_tensor = Tensor::rand(Shape{{2, 2}}, TensorOptions());
    // one-hot: row0 -> class0, row1 -> class1


    // --- Graph handles ---
    auto a    = make_tensor(A_tensor,    "A");
    auto b    = make_tensor(B_tensor,    "B");
    auto bias = make_tensor(Bias_tensor, "bias");
    auto Yt   = make_tensor(Yt_tensor,   "Y_target");

    // Prediction: logits = A*B + bias
    auto logits = matmul(a, b) + bias;

    // IMPORTANT: bias was already added in logits; don't add it twice
    // Loss: KL(target || log_probs). Most libs expect log_probs:
    auto probs = softmax_row(logits);
auto y = kldivergence(log(probs), Yt); // if you have a safe log()


    // Optional: zero out forward caches if your engine needs it
    zero_val(y);

    cout << fixed << setprecision(6);

    const float lr = 0.10f;
    for (int it = 0; it < 200; ++it) {
        zero_grad(y);     // clear grads on the graph
        forward(y);       // compute loss
        backward(y);      // backprop

        if (it % 10 == 0) {
            cout << "it " << it << " | loss = ";
            debug::print_value("", y); // prints and newline
        }

        // Update all leaf params reachable from y (requires_grad=true)
        SGD(y, /*params*/nullptr, lr);

        // (Optional) mirror values back into raw tensors for inspection
        A_tensor.copy_(a.val());
        B_tensor.copy_(b.val());
        Bias_tensor.copy_(bias.val());
    }

    // Final inspection
    cout << "\nFinal params & preds:\n";
    debug::print_value("A", a);
    debug::print_value("B", b);
    debug::print_value("bias", bias);
    auto final_logits = matmul(a, b) + bias;
    auto final_logp   = kldivergence(log(softmax_row(final_logits)), Yt);
    debug::print_value("logits", final_logits);
    debug::print_value("log_probs", final_logp);
    debug::print_value("target", Yt);

    return 0;
}
