import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../../build"))
import cgadimpl 

print(cgadimpl.cuda_available())  # True/False depending on build

aa = cgadimpl.Tensor.randn(3,4,11,cgadimpl.Device.CPU)
bb = cgadimpl.Tensor.randn(3,4,11,cgadimpl.Device.CPU)
cc = cgadimpl.Tensor.randn(3,4,11,cgadimpl.Device.CPU)
a = cgadimpl.Node(aa, True, cgadimpl.Op.Leaf, 'leaf')
b = cgadimpl.Node(bb, True, cgadimpl.Op.Leaf, 'leaf')
c = cgadimpl.Node(cc, True, cgadimpl.Op.Leaf, 'leaf')


n = cgadimpl.add_nodeops(cgadimpl.sub_nodeops(a,c), b)

print(n.value) 
print(a.value) 
print(b.value) 
print(c.value) 