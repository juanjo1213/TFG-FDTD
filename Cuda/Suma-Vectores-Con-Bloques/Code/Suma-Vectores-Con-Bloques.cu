//Cálculo de suma de vectores usando bloques de varios hilos cada uno

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
using namespace std;

#define BLOCK 10   //Defino los hilos por bloque


__global__ void suma(int* vector1, int* vector2, int* vector_suma, int n);

int main(int argc, char** argv)
{
	//Archivo donde guardar el resultado
	FILE* out_file = fopen("Resultados.txt", "w"); // write only 
			  // test for files not existing. 
	if (out_file == NULL)
	{
		printf("Error! Could not open file\n");
		exit(-1); // must include stdlib.h 
	}

	//Declaro vectores
	int* hst_v1, * hst_v2, * hst_suma;
	int* dev_v1, * dev_v2, * dev_suma;
	int n = 25;

	ofstream myfile;
	myfile.open("example.txt");


	//Reservo memoria
	//Host
	hst_v1 = (int*)malloc(n * sizeof(int));
	hst_v2 = (int*)malloc(n * sizeof(int));
	hst_suma = (int*)malloc(n * sizeof(int));
	//Device
	cudaMalloc((void**)&dev_v1, n * sizeof(int));
	cudaMalloc((void**)&dev_v2, n * sizeof(int));
	cudaMalloc((void**)&dev_suma, n * sizeof(int));

	//Inicializo el vector 1
	for (int i = 0;i < n;i++)
	{
		hst_v1[i] = (int)rand();
	}

	//Paso el vector 1 del host al device
	cudaMemcpy(dev_v1, hst_v1, n * sizeof(int), cudaMemcpyHostToDevice);

	//Calculo el numero de bloques
	int bloques = n / BLOCK;
	//Si la división no es exacta, añado un bloque mas 
	if (n % BLOCK != 0)
	{
		bloques = bloques + 1;
	}

	printf("> Vector de %d elementos\n", n);
	printf("> Lanzamiento con %d bloques de %d hilos (%d hilos)\n", bloques, BLOCK, bloques * BLOCK);


	//Calculo la suma
	suma << <bloques, BLOCK >> > (dev_v1, dev_v2, dev_suma, n);

	//Paso los vectores v2 y vsuma del device al host
	cudaMemcpy(hst_v2, dev_v2, n * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(hst_suma, dev_suma, n * sizeof(int), cudaMemcpyDeviceToHost);




	// impresion de resultados
	printf("VECTOR 1:\n");
	for (int i = 0; i < n; i++)
	{
		printf("%2d ", hst_v1[i]);
	}
	printf("\n");
	printf("VECTOR 2:\n");
	for (int i = 0; i < n; i++)
	{
		printf("%2d ", hst_v2[i]);
	}
	printf("\n");
	printf("SUMA:\n");
	for (int i = 0; i < n; i++)
	{
		printf("%2d ", hst_suma[i]);
		//Escribo en el fichero ejemplo
		myfile << hst_suma[i] << " ";
	}
	printf("\n");
	// salida
	printf("\n<pulsa [INTRO] para finalizar>\n");

	// write to file
	fwrite(hst_suma, sizeof(int), sizeof(hst_suma), out_file); // write to file 


	myfile.close();

	// Liberacion de recursos
	cudaFree(dev_v1);
	cudaFree(dev_v2);
	cudaFree(dev_suma);

	free(hst_v1);
	free(hst_v2);
	free(hst_suma);

	//Cierro el fichero
	fclose(out_file);


	return 0;
}


//Funcion suma usando bloques, generando el vector 2 de forma aleatoria
__global__ void suma(int* vector1, int* vector2, int* vector_suma, int n)
{
	//Creo el identificador para cada proceso
	int id = threadIdx.x + blockDim.x * blockIdx.x;


	if (id < n)
	{
		//Genero el vector 2
		vector2[id] = (n - 1) + id;

		vector_suma[id] = vector1[id] + vector2[id];


	}

}