cmake_minimum_required(VERSION 3.16)
project(pathfinder LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Decoder library (pure CPU C++ — builds everywhere)
add_library(decoder SHARED
    src/types.cpp
    src/union_find.cpp
)
target_include_directories(decoder PUBLIC include)

# Python bindings
find_package(pybind11 QUIET)
if(pybind11_FOUND)
    pybind11_add_module(pydecoder python/pydecoder.cpp)
    target_link_libraries(pydecoder PRIVATE decoder)
    message(STATUS "pybind11 found — building pydecoder Python module")
else()
    message(STATUS "pybind11 not found — skipping pydecoder (pip install pybind11)")
endif()
