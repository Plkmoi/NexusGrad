cmake_minimum_required(VERSION 3.12)
project(cgad_pybind)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python and pybind11
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Include directories for main library
include_directories(${CMAKE_SOURCE_DIR}/../cgadimpl/include)
include_directories(${CMAKE_SOURCE_DIR}/../tensor/include)

# Find the main cgadimpl library
find_library(CGADIMPL_LIB 
    NAMES cgadimpl
    PATHS ${CMAKE_SOURCE_DIR}/../cgadimpl/build ${CMAKE_SOURCE_DIR}/../cgadimpl/build/Release
    NO_DEFAULT_PATH
)

if(NOT CGADIMPL_LIB)
    message(FATAL_ERROR "cgadimpl library not found. Please build cgadimpl first.")
endif()

# Collect all binding source files
file(GLOB_RECURSE BINDING_SOURCES 
    "${CMAKE_CURRENT_SOURCE_DIR}/bindings/*.cpp"
)

# Create the Python module
pybind11_add_module(cgad 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cgad_pybind.cpp
    ${BINDING_SOURCES}
)

# Link against main library
target_link_libraries(cgad PRIVATE ${CGADIMPL_LIB})

# Output to site-packages or current directory
set_target_properties(cgad PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/dist"
)

# Installation
install(TARGETS cgad LIBRARY DESTINATION "${Python3_SITELIB}")

# Build options
option(BUILD_TESTS "Build Python binding tests" ON)
if(BUILD_TESTS)
    add_subdirectory(tests)
endif()
