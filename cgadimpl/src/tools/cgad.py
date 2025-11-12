import sys, os

# Get absolute path to build folder
build_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../build"))
print("Adding:", build_dir)
sys.path.append(build_dir)

import cgadimpl  # should now work


cgadimpl.kernels.load_cuda_plugin("/home/blubridge-034/Downloads/Newf/cgadimpl/cgadimpl/build/libagkernels_cuda.so")
cgadimpl.kernels.load_cpu_plugin("/home/blubridge-034/Downloads/Newf/cgadimpl/cgadimpl/build/libagkernels_cpu.so")

y = cgadimpl.Tensor.ones([4, 4], device="cuda", requires_grad=True)
s = cgadimpl.Tensor.ones([4, 4], device="cuda", requires_grad=True)
r = cgadimpl.Tensor.ones([4, 4], device="cuda", requires_grad=True)
g = cgadimpl.Tensor.ones([4, 4], device="cuda", requires_grad=True)
print(y.shape, y.is_cuda(), y.requires_grad)
q = cgadimpl.make_tensor(y, "A")
w = cgadimpl.make_tensor(s, "B")
x = cgadimpl.make_tensor(r, "C")
p = cgadimpl.make_tensor(g, "D")
f = cgadimpl.ops.add(q,w)
m = cgadimpl.ops.linear(x,f,p)
for i in range(22):
    cgadimpl.forward(m)
    cgadimpl.zero_grad(m)
    cgadimpl.backward(m)
    cgadimpl.optim.SGD(m, 1)

    
try:
    cgadimpl.debug.print_grad("M",q)
except:
    print('YAY!')
finally:
    try:
        cgadimpl.debug.print_value("M",q)
    except:
        print('AGAIN YAY!')
    finally:
        print("DONE!")
