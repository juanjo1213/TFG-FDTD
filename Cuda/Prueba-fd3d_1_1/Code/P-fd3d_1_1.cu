//C�lculo del campo electromagn�tico usando el m�todo FDTD con Cuda
//Juan Jos� Salazar

//Librerias utilizadas
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>

using namespace std;

//Defino la dimensi�n de los vectores campo el�ctrico y magn�tico
#define dim 200



//Funci�n principal
int main(int argc, char** argv)
{
	//Declaraci�n de punteros usados en host y device
	float* hst_e, * hst_h;
	float* dev_e, * dev_h;
	//Declaro el n�mero de bloques a utilizar
	int bloques = 2;

	//Declaro dos ficheros para guardar los resultados
	ofstream ex;
	ofstream hy;

	//Abro los ficheros
	ex.open("ex.txt");
	hy.open("hy.txt");

	//Reserva de memoria en host y device
	hst_e = (float*)malloc(dim * sizeof(float));
	hst_h = (float*)malloc(dim * sizeof(float));

	cudaMalloc((void**)&dev_e, dim * sizeof(float));
	cudaMalloc((void**)&dev_h, dim * sizeof(float));

	//Inicializo los vectores
	for (int i = 0;i < dim;i++)
	{
		hst_e[i] = 0;
		hst_h[i] = 0;
	}




	//Muevo la informaci�n de los vectores del host al device
	//cudaMemcpy(dev_e, hst_e, dim * sizeof(float), cudaMemcpyHostToDevice);
	//cudaMemcpy(dev_h, hst_h, dim * sizeof(float), cudaMemcpyHostToDevice);


	//Llamo a la funci�n para ejecutar el m�todo con un �nico bloque de 200 hilos
	//fdtd << <1, dim >> > (dev_e, dev_h,dim);




	//Muevo la informaci�n de los vectores del device al host
	//cudaMemcpy(hst_e, dev_e, dim * sizeof(float), cudaMemcpyDeviceToHost);
	//cudaMemcpy(hst_h, dev_h, dim * sizeof(float), cudaMemcpyDeviceToHost);



	//Declaro los parametros del pulso
	float pulse;
	int kc = (int)dim / 2;
	float t = 40.0;
	float spread = 12.0;
	float a;
	//Declaro el n�mero de pasos
	int nsteps = 100;

	//Realizo el m�todo
	for (int i = 1;i < (nsteps + 1);i++)
	{

		for (int j = 1;j < dim;j++)
		{
			//C�lculo del vector Ex
			hst_e[j] = hst_e[j] + 0.5 * (hst_h[j - 1] - hst_h[j]);
		}
		//Genero el pulso
		a = ((t - i) / spread) * ((t - i) / spread);
		a = -0.5 * a;
		pulse = expf(a);
		hst_e[kc] = pulse;
		//cout << pulse << endl;

		for (int j = 0;j < dim - 1;j++)
		{
			//C�lculo del vector Hy
			hst_h[j] = hst_h[j] + 0.5 * (hst_e[j] - hst_e[j + 1]);
		}
	}



	//Escribo los resultados en los ficheros ex.txt (campo electrico) y hy.txt (campo magnetico)
	for (int i = 0;i < dim;i++)
	{
		ex << hst_e[i] << endl;
		hy << hst_h[i] << endl;
		cout << hst_e[i] << endl;
	}


	// Liberacion de recursos tanto en host como en device
	cudaFree(dev_e);
	cudaFree(dev_h);

	free(hst_e);
	free(hst_h);

	//Cierro los archivos
	ex.close();
	hy.close();


	return 0;
}


