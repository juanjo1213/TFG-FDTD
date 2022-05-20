
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(fd3d_1_1 LANGUAGES CXX CUDA)





# Definir programa ejecutable: ejemplo C++ básico
add_executable(
	# Nombre del ejecutable
	fd3d_1_1
	# Lista de ficheros de código fuente a compilar en este "target":
	fd3d_1_1.cu
)

