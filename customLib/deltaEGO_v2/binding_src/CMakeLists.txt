cmake_minimum_required(VERSION 3.18)
project(delta_ego_core LANGUAGES CXX)

# C++20, AVX-512
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# compile flag
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  add_compile_options(-O3 -march=native -funroll-loops)
endif()

#PYBIND11
find_package(pybind11 REQUIRED)

pybind11_add_module(delta_ego_core
  deltaEGO.cpp
  json.hpp
)

target_include_directories(delta_ego_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
