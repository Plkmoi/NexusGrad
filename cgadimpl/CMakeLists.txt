
cmake_minimum_required(VERSION 3.20)

# --- FIX #2: Set the CUDA architecture BEFORE the project() command ---
set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA SMs" FORCE)

project(cgadimpl LANGUAGES CXX CUDA)

# --- Compiler, CUDA, and OpenMP Setup ---
find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++20")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)



# ---- Python & Pybind11 setup ----
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Locate pybind11 even if installed via pip
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
  OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")

find_package(pybind11 CONFIG REQUIRED)



# --- Find and Link the Pre-built Tensor Library ---
set(OWNTENSOR_DIR ${CMAKE_SOURCE_DIR}/../tensor)
find_path(OWNTENSOR_INCLUDE_DIR NAMES TensorLib.h HINTS ${OWNTENSOR_DIR}/include)
find_library(OWNTENSOR_LIBRARY NAMES tensor HINTS ${OWNTENSOR_DIR}/lib)


if(OWNTENSOR_INCLUDE_DIR AND OWNTENSOR_LIBRARY)
    add_library(OwnTensor::tensor SHARED IMPORTED)
    set_target_properties(OwnTensor::tensor PROPERTIES
        IMPORTED_LOCATION "${OWNTENSOR_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${OWNTENSOR_INCLUDE_DIR}"
    )
    message(STATUS "Found OwnTensor headers: ${OWNTENSOR_INCLUDE_DIR}")
    message(STATUS "Found OwnTensor library: ${OWNTENSOR_LIBRARY}")
else()
    message(FATAL_ERROR "Could not find the OwnTensor library. Searched in ${OWNTENSOR_DIR}")
endif()
# ---- Core C++ library ----
file(GLOB_RECURSE CGADIMPL_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
list(REMOVE_ITEM CGADIMPL_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/tools/binder.cpp")

add_library(cgadimpl_core STATIC ${CGADIMPL_SRC})
add_library(cgadimpl::cgadimpl ALIAS cgadimpl_core)

target_compile_definitions(cgadimpl_core PUBLIC WITH_CUDA)
target_include_directories(cgadimpl_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(cgadimpl_core PUBLIC OwnTensor::tensor dl)


# ---- Pybind11 module ----
pybind11_add_module(cgadimpl src/tools/binder.cpp)  # 👈 this creates cgadimpl.so for Python

target_link_libraries(cgadimpl PRIVATE
    cgadimpl_core
    OwnTensor::tensor
    pybind11::module
    Python3::Python
    OpenMP::OpenMP_CXX
)

target_include_directories(cgadimpl PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${Python3_INCLUDE_DIRS}
)

set_target_properties(cgadimpl PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/build"
)

# ---- Tests ----
if(true) # or use if(AG_BUILD_TESTS)
  enable_testing()

  # --- FIX 1: Clean up the helper functions (Define only once) ---

  # Helper for CPU Tests
  function(add_ag_test name src)
    add_executable(${name} ${src})
    
    # --- FIX 2: Link against cgadimpl::cgadimpl (the Core static lib), NOT cgadimpl (the Python module)
    # --- FIX 3: Explicitly link dl (needed for CUDA driver) and CUDA::cudart
    target_link_libraries(${name} PRIVATE 
        cgadimpl::cgadimpl 
        CUDA::cudart 
        OpenMP::OpenMP_CXX 
        dl
    )

    add_test(NAME ${name} COMMAND ${name})
  endfunction()

  # Helper for CUDA Tests
  function(add_ag_testcuda name src)
    add_executable(${name} ${src})
    
    # Force NVCC compilation for the test file
    set_source_files_properties(${src} PROPERTIES LANGUAGE CUDA)
    
    target_link_libraries(${name} PRIVATE 
        cgadimpl::cgadimpl 
        CUDA::cudart 
        dl
    )

    add_test(NAME ${name} COMMAND ${name})
  endfunction()
  
  # --- Define Test Sources ---
  
  # CPU Tests
  # add_ag_test(test_layalib     tests/test_layalib.cpp)
  add_ag_test(test_eh       tests/test_eh.cpp)
  add_ag_test(test_neh       tests/test_neh.cpp)
  add_ag_test(test_aliop       tests/test_aliop.cpp)
 add_ag_test(test_keretu       tests/test_keretu.cpp)
  # add_ag_test(test_layer       tests/test_layer.cpp)
  # add_ag_test(test_aligen       tests/test_aligen.cpp)
  # Add your other CPU tests back here...

  # GPU/CUDA Tests
  # Ensure these files actually have .cpp or .cu extension and contain CUDA code


endif()


message(STATUS "cgadimpl build mode: ${CMAKE_BUILD_TYPE}")
















  # add_ag_test(test_ag                    tests/test_ag.cpp)
  # add_ag_test(test_trainag                    tests/test_trainag.cpp)
  # add_ag_test(test_forgpu                    tests/test_forgpu.cpp)
  # add_ag_test(test_mlp                    tests/test_mlp.cpp)
  # add_ag_test(test_complex_mlp                    tests/test_complex_mlp.cpp)
  # # add_ag_test(test_bench_relu                    tests/bench_relu.cpp)
  # add_ag_test(test_tiny_handcalc                    tests/tiny_handcalc.cpp)
  # add_ag_test(test_emb                    tests/test_emb.cpp)
  # add_ag_test(test_aline                    tests/test_aline.cpp)
  # add_ag_test(test_alinef                    tests/test_alinef.cpp)
  # add_ag_test(test_graph_compile                    tests/test_graph_compile.cpp)
  # add_ag_test(test_newcompile                    tests/test_newcompile.cpp)
  # add_ag_test(ag_core_test                    tests/test.cpp)
  # # add_ag_test(test_checkpoint                    tests/test_checkpoint.cpp)
  # # add_ag_test(test_inplace                    tests/test_inplace.cpp)
  # # add_ag_test(test_careful_deletion                    tests/test_careful_deletion.cpp)
  # add_ag_test(test_nn                    tests/test_nn.cpp)
  # add_ag_test(test_ann                    tests/test_ann.cpp)
  # add_ag_test(test_uniali                    tests/test_uniali.cpp)
  # add_ag_test(test_realann                    tests/test_realann.cpp)
  # add_ag_test(test_wann                    tests/test_wann.cpp)
  # add_ag_test(test_cann                    tests/test_cann.cpp)
  # add_ag_test(test_alinfe                    tests/test_alinfe.cpp)
  # add_ag_test(test_floal                    tests/test_floal.cpp)
  # add_ag_test(test_singat                    tests/test_singat.cpp)
  # add_ag_test(test_crosat                    tests/test_crosat.cpp)
  # add_ag_test(test_tenat                    tests/test_tenat.cpp)
  # add_ag_test(test_alibat                    tests/test_alibat.cpp)
  # add_ag_test(test_alicat                    tests/test_alicat.cpp)
  # add_ag_test(test_version                    tests/test_version.cpp)
  # add_ag_test(test_nodeops                    tests/test_nodeops.cpp)
  # add_ag_test(test_kernels_cpu                    tests/test_kernels_cpu.cpp)
  # add_ag_test(test_vjp_kernels                    tests/test_end_to_end_gpu.cpp)
  # add_ag_test(test_nn_mlp                    tests/test_mlp_training.cpp)
  # add_ag_test(test_kernels_gpu                    tests/test_kernels_gpu.cpp)
  # # add_ag_test(test_graph_cuda                    tests/test_graph_gpu.cpp)
  # add_ag_test(test_core_ops_gpu                    tests/test_core_ops_gpu.cpp)