cmake_minimum_required(VERSION 2.8)

project(libL2)

add_library(L2 STATIC L2/L2.cpp L2/temp.cpp)
target_include_directories(L2 PUBLIC include)

install(TARGETS L2 DESTINATION lib)