cmake_minimum_required(VERSION 2.8)

project(libL2)

add_library(L2 STATIC L2/tensor.cpp)
target_include_directories(L2 PUBLIC include)

install(TARGETS L2 DESTINATION lib)