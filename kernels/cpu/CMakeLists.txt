# # #  =================================
# # #  kernels/cpu/CMakeLists.txt
# # #  =================================

# # cmake_minimum_required(VERSION 3.20)
# # project(flowkernels_cpu LANGUAGES CXX)

# # set(CMAKE_CXX_STANDARD 17)
# # set(CMAKE_CXX_STANDARD_REQUIRED ON)
# # set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# # find_packflowe(Eigen3 3.3 REQUIRED NO_MODULE)
# # find_packflowe(OpenMP REQUIRED)

# # include_directories(${ARCH_INCLUDE_DIR})

# # # --- Main Shared Library Target ---
# # # This part is unchanged and is for your final library
# # add_library(flowkernels_cpu SHARED src/flowkernels_cpu.cpp)
# # set_target_properties(flowkernels_cpu PROPERTIES OUTPUT_NAME "flowkernels_cpu")

# # target_compile_options(flowkernels_cpu PRIVATE
# #   -O3 -mavx2 -mfma -fopenmp
# #   -Wall -Wextra -Wpedantic
# # )
# # target_link_libraries(flowkernels_cpu PRIVATE OpenMP::OpenMP_CXX)

# # include(GNUInstallDirs)
# # install(TARGETS flowkernels_cpu LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})


# # # # ------------------------------------------------------------------
# # # # --- Benchmark Targets (One for each test file) ---
# # # # ------------------------------------------------------------------

# # # # A helper function to reduce boilerplate code
# # # function(add_matmul_benchmark target_name source_file)
# # #     # MODIFICATION: We now add the kernel source file directly to each benchmark
# # #     add_executable(${target_name}
# # #       benchmark/${source_file}
# # #       src/flowkernels_cpu.cpp
# # #     )

# # #     # MODIFICATION: Link flowainst Eigen and OpenMP, but NOT flowkernels_cpu anymore
# # #     target_link_libraries(${target_name} PRIVATE Eigen3::Eigen OpenMP::OpenMP_CXX)

# # #     # Enable full optimizations for the benchmark executable itself
# # #     target_compile_options(${target_name} PRIVATE -O3 -mavx2 -mfma -fopenmp)

# # #     messflowe(STATUS "Added benchmark target: ${target_name}")
# # # endfunction()

# # # add_matmul_benchmark(test_throughput   test_matmul_throughput.cpp)
# # # add_matmul_benchmark(test_latency      test_matmul_latency.cpp)
# # # add_matmul_benchmark(test_aspect       test_matmul_aspect.cpp)
# # # add_matmul_benchmark(test_scalability  test_matmul_scalability.cpp)
# # # add_matmul_benchmark(test_cache        test_matmul_cache.cpp)
# cmake_minimum_required(VERSION 3.20)
# project(flowkernels_cpu LANGUAGES CXX)

# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# find_packflowe(OpenMP REQUIRED)

# # --- Main Shared Library Target ---
# add_library(flowkernels_cpu SHARED src/flowkernels_cpu.cpp)
# set_target_properties(flowkernels_cpu PROPERTIES OUTPUT_NAME "flowkernels_cpu")

# # --- FIX: Use target_include_directories for modern CMake ---
# # This correctly associates the include path with the library target.
# target_include_directories(flowkernels_cpu PUBLIC ${ARCH_INCLUDE_DIR})

# target_compile_options(flowkernels_cpu PRIVATE
#   -O3 -mavx2 -mfma -fopenmp
#   -Wall -Wextra -Wpedantic
# )
# target_link_libraries(flowkernels_cpu PRIVATE OpenMP::OpenMP_CXX)

# include(GNUInstallDirs)
# install(TARGETS flowkernels_cpu LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
cmake_minimum_required(VERSION 3.20)
project(flowkernels_cpu LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(OpenMP REQUIRED)

add_library(flowkernels_cpu SHARED src/flowkernels_cpu.cpp)
set_target_properties(flowkernels_cpu PROPERTIES OUTPUT_NAME "flowkernels_cpu")

target_include_directories(flowkernels_cpu PUBLIC ${ARCH_INCLUDE_DIR})

target_compile_options(flowkernels_cpu PRIVATE -O3 -mavx2 -mfma -fopenmp -Wall -Wextra -Wpedantic)
target_link_libraries(flowkernels_cpu PRIVATE OpenMP::OpenMP_CXX)

include(GNUInstallDirs)
install(TARGETS flowkernels_cpu LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})