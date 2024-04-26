cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(MyProj)

set(CMAKE_CXX_STANDARD 20)

include_directories(third_party)
include_directories(include)

add_subdirectory(src)
