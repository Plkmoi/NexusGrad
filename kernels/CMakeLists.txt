# ================================
# kernels/CMakeLists.txt
# ================================
cmake_minimum_required(VERSION 3.20)
project(flowkernels LANGUAGES CXX)

set(ARCH_INCLUDE_DIR "/home/blubridge-034/Downloads/Newf/cgadimpl/arch/include" CACHE PATH "Path to arch/include (contains kernels/kernels.hpp)")
if(NOT ARCH_INCLUDE_DIR)
  message(FATAL_ERROR "Set -DARCH_INCLUDE_DIR=/abs/path/to/arch/include")
endif()

add_subdirectory(cpu)
add_subdirectory(gpu)