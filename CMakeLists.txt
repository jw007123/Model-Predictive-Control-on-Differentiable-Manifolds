# CMake setup
cmake_minimum_required(VERSION 3.11)

# Project setup
project(MPConManifolds VERSION 1.0)

# C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Libs and exe
add_subdirectory(Source)
add_subdirectory(Source/Tests)
