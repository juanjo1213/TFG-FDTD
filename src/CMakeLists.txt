cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(fd2d LANGUAGES CXX CUDA)

add_executable(fd2d_simp fd2d_simp.cu)

add_executable(HOST_FD2D_SIMP HOST_FD2D_SIMP.cu)


