cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(MyProj)

set(CMAKE_CXX_STANDARD 17)

include_directories(third_party)
add_subdirectory(third_party/gflags EXCLUDE_FROM_ALL)

add_subdirectory(common)
add_subdirectory(src)
