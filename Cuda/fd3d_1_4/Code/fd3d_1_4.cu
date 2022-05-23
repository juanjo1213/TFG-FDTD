//Calculus of the electromagnetic field using the FDTD method in Cuda with boundary condition so we have no reflected wave.
// In this case it is set a dielectric medium at the half right of the space, with epsilon=4.
// We use a sinusoidal function as pulse
// 
//Juan José Salazar

//Libraries
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <math.h>       /* sin */

#define PI 3.14159265


using namespace std;

//Electromagnetic vectors dimension
#define dim 200

//Kernel function to calculate the fields
__global__ void fdtd(float* ex, float* hy, float* aux, float* epsilon, int dimension, float puls)
{
	//Wire id generation
	int id = threadIdx.x;





	if (id > 0)
	{
		//Ex vector calculus
		ex[id] = ex[id] + epsilon[id] * (hy[id - 1] - hy[id]);
	}



	//Pulse generation
	ex[5] = puls + ex[5];

	//Boundary conditions application
	ex[0] = aux[0];
	aux[0] = ex[1];
	ex[dimension - 1] = aux[1];
	aux[1] = ex[dimension - 2];



	if (id < dimension - 1)
	{
		//Hy vector calculus
		hy[id] = hy[id] + 0.5 * (ex[id] - ex[id + 1]);

	}




}


//MAin function
int main(int argc, char** argv)
{
	//Declaration of host and device pointers
	float* hst_e, * hst_h, * hst_aux, * hst_epsilon;
	float* dev_e, * dev_h, * dev_aux, * dev_epsilon;

	//Block declaration
	int bloques = 1;

	//Pulse parameter declaration
	float pulse;
	int kc = (int)dim / 2;
	float ddx = 0.01;        //Cell size
	float dt = ddx / 6e8;    //Time step size
	float freq_in = 700e6;

	//Dielectric constant
	float epsilon = 4;

	//Number of steps done
	int nsteps = 425;

	//Output files declaration (each one to store the fields values)
	ofstream ex;
	ofstream hy;

	//File opening
	ex.open("ex.txt");
	hy.open("hy.txt");

	//Host and device memory reserve
	hst_e = (float*)malloc(dim * sizeof(float));
	hst_h = (float*)malloc(dim * sizeof(float));
	hst_aux = (float*)malloc(2 * sizeof(float));
	hst_epsilon = (float*)malloc(dim * sizeof(float));

	cudaMalloc(&dev_e, dim * sizeof(float));
	cudaMalloc(&dev_h, dim * sizeof(float));
	cudaMalloc(&dev_aux, 2 * sizeof(float));
	cudaMalloc(&dev_epsilon, dim * sizeof(float));

	//Vectors initialization
	for (int i = 0;i < dim;i++)
	{
		hst_e[i] = 0;
		hst_h[i] = 0;

		//Inicializo epsilon
		if (i < kc) hst_epsilon[i] = 0.5;
		else hst_epsilon[i] = 0.5 / epsilon;

	}

	//Auxiliar vector for boundary conditions initialization
	hst_aux[0] = 0;
	hst_aux[1] = 0;

	//Host to device information movement
	cudaMemcpy(dev_e, hst_e, dim * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_h, hst_h, dim * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_aux, hst_aux, 2 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_epsilon, hst_epsilon, dim * sizeof(float), cudaMemcpyHostToDevice);

	for (int i = 1;i < nsteps + 1;i++)
	{
		//Pulse generation
		pulse = sin(2*PI*freq_in*dt*i);
		//cout << pulse << endl;
		//Kernel function call to use 200 wires (vectors dimension) to calculate the fields
		fdtd << <bloques, dim >> > (dev_e, dev_h, dev_aux, dev_epsilon, dim, pulse);
	}



	//Device to host information movement
	cudaMemcpy(hst_e, dev_e, dim * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(hst_h, dev_h, dim * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(hst_aux, dev_aux, 2 * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(hst_epsilon, dev_epsilon, dim * sizeof(float), cudaMemcpyDeviceToHost);




	//Results writing 
	for (int i = 0;i < dim;i++)
	{
		ex << hst_e[i] << endl;
		hy << hst_h[i] << endl;
		cout << hst_e[i] << endl;
	}


	// Device and host memory release
	cudaFree(dev_e);
	cudaFree(dev_h);
	cudaFree(dev_aux);
	cudaFree(dev_epsilon);

	free(hst_e);
	free(hst_h);
	free(hst_aux);
	free(hst_epsilon);

	//File closing
	ex.close();
	hy.close();


	return 0;
}


