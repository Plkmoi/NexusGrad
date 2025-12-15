
cmake_minimum_required(VERSION 3.20)

# --- FIX #2: Set the CUDA architecture BEFORE the project() command ---
set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA SMs" FORCE)

project(arch LANGUAGES CXX CUDA)

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


# --- Find and Link the Pre-built Tensor Library ---
set(OWNCGAD_DIR ${CMAKE_SOURCE_DIR}/../cgadimpl)
find_path(OWNCGAD_INCLUDE_DIR NAMES ad/ag_all.hpp HINTS ${OWNCGAD_DIR}/include)
find_library(OWNCGAD_LIBRARY NAMES cgadimpl_core HINTS ${OWNCGAD_DIR}/build)


if(OWNCGAD_INCLUDE_DIR AND OWNCGAD_LIBRARY)
    add_library(ag::cgadimpl_core SHARED IMPORTED)
    set_target_properties(ag::cgadimpl_core PROPERTIES
        IMPORTED_LOCATION "${OWNCGAD_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${OWNCGAD_INCLUDE_DIR}"
    )
    message(STATUS "Found CGAD headers: ${OWNCGAD_INCLUDE_DIR}")
    message(STATUS "Found CGAD library: ${OWNCGAD_LIBRARY}")
else()
    message(FATAL_ERROR "Could not find the CGAD library. Searched in ${OWNCGAD_DIR}")
endif()


# ---- Core C++ library ----
file(GLOB_RECURSE ARCH_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

add_library(arch SHARED ${ARCH_SRC})
# add_library(arch::arch ALIAS arch)

target_compile_definitions(arch PUBLIC WITH_CUDA)
target_include_directories(arch PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(arch PUBLIC OwnTensor::tensor dl)
target_link_libraries(arch PUBLIC ag::cgadimpl_core dl)



target_link_libraries(arch PRIVATE
    ag::cgadimpl_core
    OwnTensor::tensor
    pybind11::module
    Python3::Python
    OpenMP::OpenMP_CXX
)

target_include_directories(arch PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${Python3_INCLUDE_DIRS}
)

set_target_properties(arch PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/build"
)




# ---- Tests ----
if(true) # or use if(AG_BUILD_TESTS)
  enable_testing()

  # --- FIX 1: Clean up the helper functions (Define only once) ---

  # Helper for CPU Tests
  function(add_arch_test name src)
    add_executable(${name} ${src})
    
    # --- FIX 2: Link against cgadimpl::cgadimpl (the Core static lib), NOT cgadimpl (the Python module)
    # --- FIX 3: Explicitly link dl (needed for CUDA driver) and CUDA::cudart
    target_link_libraries(${name} PRIVATE 
        arch
        CUDA::cudart 
        OpenMP::OpenMP_CXX 
        dl
    )

    add_test(NAME ${name} COMMAND ${name})
  endfunction()

  # Helper for CUDA Tests
  function(add_arch_testcuda name src)
    add_executable(${name} ${src})
    
    # Force NVCC compilation for the test file
    set_source_files_properties(${src} PROPERTIES LANGUAGE CUDA)
    
    target_link_libraries(${name} PRIVATE 
        arch
        CUDA::cudart 
        dl
    )

    add_test(NAME ${name} COMMAND ${name})
  endfunction()
  
  # --- Define Test Sources ---
  
  # CPU Tests
  # add_arch_test(test_layalib     tests/test_layalib.cpp)
  add_arch_test(test_aliop       tests/test_aliop.cpp)
  add_arch_test(test_eh       tests/test_eh.cpp)
  add_arch_test(test_neh       tests/test_neh.cpp)
 add_arch_test(test_keretu       tests/test_keretu.cpp)
 add_arch_test(test_aligen       tests/test_aligen.cpp)
#  add_arch_test(test_alimass       tests/test_alimass.cpp) WORKING MASSIVE LAYER!!!!!!
  # add_arch_test(test_layer       tests/test_layer.cpp)
  # add_arch_test(test_aligen       tests/test_aligen.cpp)
  # Add your other CPU tests back here...

  # GPU/CUDA Tests
  # Ensure these files actually have .cpp or .cu extension and contain CUDA code


endif()


message(STATUS "arch build mode: ${CMAKE_BUILD_TYPE}")



