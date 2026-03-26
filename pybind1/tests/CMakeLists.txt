cmake_minimum_required(VERSION 3.12)

# Python tests are run directly, no compilation needed
# This file serves as a reference for potential C++ test integration

# Enable testing
enable_testing()

# Add Python-based tests
if(Python3_FOUND)
    add_test(
        NAME python_bindings_test
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_bindings.py
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
    
    set_tests_properties(python_bindings_test PROPERTIES
        ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/../dist:$ENV{PYTHONPATH}"
    )
endif()

# Note: Additional test configurations can be added here as the project grows
