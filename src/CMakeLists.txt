cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(fd2d LANGUAGES CXX CUDA)

add_executable(fd2d fd2d.cu)

add_executable(hostfd2d hostfd2d.cu)