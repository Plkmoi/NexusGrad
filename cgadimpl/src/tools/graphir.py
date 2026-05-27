import sys, os

# Get absolute path to build folder
build_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../build"))
print("Adding:", build_dir)
sys.path.append(build_dir)

import cgadimpl  # should now work

cgadimpl.kernels.load_cuda_plugin("/home/blubridge-034/Downloads/Newf/cgadimpl/cgadimpl/build/libagkernels_cuda.so")
cgadimpl.kernels.load_cpu_plugin("/home/blubridge-034/Downloads/Newf/cgadimpl/cgadimpl/build/libagkernels_cpu.so")


def qae(a, b):
    c = a+b


