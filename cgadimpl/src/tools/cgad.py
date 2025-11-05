import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../../build"))
import cgadimpl 

cgadimpl.kernels.load_cuda_plugin("/home/blubridge-034/Downloads/Newf/cgadimpl/cgadimpl/build/libagkernels_cuda.so")
cgadimpl.kernels.load_cpu_plugin("/home/blubridge-034/Downloads/Newf/cgadimpl/cgadimpl/build/libagkernels_cpu.so")

print(cgadimpl.cuda_available())  # True/False depending on build

aa = cgadimpl.Tensor.randn(3,4,11,cgadimpl.Device.CUDA)
bb = cgadimpl.Tensor.randn(3,4,11,cgadimpl.Device.CUDA)
cc = cgadimpl.Tensor.randn(3,4,11,cgadimpl.Device.CUDA)
a = cgadimpl.make_tensor(aa, "A", True)
b = cgadimpl.make_tensor(bb, "B", True)
c = cgadimpl.make_tensor(cc, "C", True)


n = (cgadimpl.ops.linear(a, b,c))
cgadimpl.backward(n)
cgadimpl.unisend(n)

print(n.val()) 
print(a.grad()) 
print(b.grad()) 
print(c.grad()) 