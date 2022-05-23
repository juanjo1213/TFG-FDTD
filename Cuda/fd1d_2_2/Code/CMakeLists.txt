
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(fd1d_2_2 LANGUAGES CXX CUDA)





# Definir programa ejecutable: ejemplo C++ básico
add_executable(
	# Nombre del ejecutable
	fd1d_2_2
	# Lista de ficheros de código fuente a compilar en este "target":
	fd1d_2_2.cu
)

