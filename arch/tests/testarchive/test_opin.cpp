
// #include <iostream>
// #include <fstream>

// #include "layer/archlist.hpp"
// #include <ops/optim.hpp>
// #include "layer/opinse.hpp"

// #include <iomanip>
// using namespace flow;


// class CusFunc : public Layer {
// public:
//     CusFunc(float gamma = 0.1f, Device dev = Device::CPU){
// _gamma = gamma;
// }

//     Value operator()(Value input) {   
//     return realrms(input, _gamma);
// }

// Value backwarda(Value input) ;


// private:
//     float _gamma = 1.0;
// };





//     namespace ag{


//          namespace detail{   

// // Value CusFunc::backwarda(Value input)
// // {   
// //     return realrms(input, _gamma);
// // }


// void vjp_CusFunc(Node* n, const Tensor& gy){
//     Node* x = n->inputs[0].get();
//     // if (!x->requires_grad()) return;

//     // const Tensor& rms = *n->tape[0]; // rsqrt(variance + epsilon)
//     // const Tensor& y_normalized = *n->tape[1]; // x * rsqrt(...)
    
//     // const float inv_N = 1.0f / static_cast<float>(x->value.shape().dims.back());

//     // Tensor dot = OwnTensor::reduce_sum(gy * y_normalized, {-1}, true);
    
//     // // grad_x = rsqrt * (gy - y * dot)
//     // Tensor grad_x = rms * (gy - y_normalized * dot);

//     x->grad += x->value;
//     std::cout<<"FRGHYEJ";
// }
//          }

// }




// void test_aliatt( int Heads, int B, int S, int D, int K, int num_layers)
// {
//     auto a = make_tensor(Tensor::randn(Shape{{D,D}}, TensorOptions().with_req_grad(true))).node;
//     auto b = make_tensor(Tensor::randn(Shape{{D,D}}, TensorOptions().with_req_grad(true))).node;
//     Tensor Y = a->value + b->value; 

//     // FIX: Use the new 3-argument Node constructor

//     enum class ExtendOriginalType : std::underlying_type_t<Op> {
//    CusFunc = static_cast<std::underlying_type_t<Op>>
//                                            (Op::ExpandHeads), // 2
//    CusFunca  // 3
// };
    
//     std::cout<<op_name((Op)ExtendOriginalType::CusFunc);
// std::ofstream myfile;
//   myfile.open ("/home/blubridge-034/Downloads/Newf/cgadimpl/arch/include/layer/newops.def");
//   myfile << "OP(CusFunc, 1, \"cusfunc\")\n";
//   myfile.close();
    
//     std::cout<<op_name((Op)ExtendOriginalType::CusFunc);


//     auto n = std::make_shared<Node>(Y, (Op)ExtendOriginalType::CusFunc, (a->requires_grad() || b->requires_grad()), "+");
    
//     n->inputs = {a, b};
//     ag::debug::on_node_created(n);
//     backward(Value(n));

// }



// int main(){

// test_aliatt(8, 4, 64, 256, 5, 11);
// // test_att(4, 4, 64, 128); // working
// // test_atte(2, 2, 8, 8, 2, 1);

// return 0;

// }