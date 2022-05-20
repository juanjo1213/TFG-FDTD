
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(Suma-Vectores-Con-Bloques LANGUAGES CXX CUDA)





# Definir programa ejecutable: ejemplo C++ básico
add_executable(
	# Nombre del ejecutable
	Suma-Vectores-Con-Bloques
	# Lista de ficheros de código fuente a compilar en este "target":
	Suma-Vectores-Con-Bloques.cu
)

