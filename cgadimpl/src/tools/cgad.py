import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../../build"))
import cgadimpl 

cgadimpl.kernels.load_cuda_plugin("/home/blubridge-034/Downloads/Newf/cgadimpl/cgadimpl/build/libagkernels_cuda.so")
cgadimpl.kernels.load_cpu_plugin("/home/blubridge-034/Downloads/Newf/cgadimpl/cgadimpl/build/libagkernels_cpu.so")

print(cgadimpl.cuda_available())  # True/False depending on build

aa = cgadimpl.Tensor.vales(3,4,0.44,cgadimpl.Device.CUDA)
bb = cgadimpl.Tensor.vales(3,4,0.11,cgadimpl.Device.CUDA)
cc = cgadimpl.Tensor.vales(3,4,0.01,cgadimpl.Device.CUDA)
dd = cgadimpl.Tensor.vals_like(cc, 0.2)
ee = cgadimpl.Tensor.vales(3,4,0.03,cgadimpl.Device.CUDA)
ff = cgadimpl.Tensor.vales(3,4,0.44,cgadimpl.Device.CUDA)
gg = cgadimpl.Tensor.vales(3,4,0.44,cgadimpl.Device.CPU)
a = cgadimpl.constant(aa, "A")
b = cgadimpl.make_tensor(bb, "B", True)
c = cgadimpl.make_tensor(cc, "C", True) 
d = cgadimpl.make_tensor(dd, "D", True) 
e = cgadimpl.make_tensor(ee, "E", True) 
f = cgadimpl.constant(ff, "F") 
g = cgadimpl.make_tensor(gg, "G", True) 
n = cgadimpl.make_tensor(cgadimpl.Tensor.zeros(3,4,cgadimpl.Device.CUDA), "N", True) 
q = cgadimpl.make_tensor(cgadimpl.Tensor.zeros(3,4,cgadimpl.Device.CUDA), "Q", True) 

for i in range(20000): 
    q = cgadimpl.ops.sigmoid(cgadimpl.ops.linear(cgadimpl.ops.relu((cgadimpl.ops.linear(a, b,c))), d, e))
    cgadimpl.zero_grad(n) 
    n = cgadimpl.ops.sum(cgadimpl.ops.mse_loss(q, f))
    cgadimpl.backward(n) 
    cgadimpl.optim.SGD(n, 0.1)
    
cgadimpl.unisend(n) 
print(n.val()) 
print(q.val()) 
print(f.val()) 
print(g.val()) 
print(a.val()) 
