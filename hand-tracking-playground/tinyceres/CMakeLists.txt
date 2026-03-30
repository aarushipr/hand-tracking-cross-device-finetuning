# Wait, what do I put in here?
project(tinyceres)

cmake_minimum_required(VERSION 3.10)

find_package(Eigen3 3 REQUIRED)

add_library(tinyceres INTERFACE)
target_include_directories(tinyceres INTERFACE include)

# install(TARGETS tinyceres)