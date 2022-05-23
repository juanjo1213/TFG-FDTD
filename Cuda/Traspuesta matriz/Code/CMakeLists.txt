
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(Traspuesta LANGUAGES CXX CUDA)





# Definir programa ejecutable: ejemplo C++ básico
add_executable(
	# Nombre del ejecutable
	Traspuesta
	# Lista de ficheros de código fuente a compilar en este "target":
	Traspuesta.cu
)

