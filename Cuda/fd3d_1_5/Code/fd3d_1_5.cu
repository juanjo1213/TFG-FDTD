//Calculus of the electromagnetic field using the FDTD method in Cuda with boundary condition so we have no reflected wave.
// In this case it is set a dielectric medium at the half right of the space, with epsilon=4.
// We use a sinusoidal function as pulse
// It is considered a lossy dielectric medium
//Juan José Salazar

//Libraries
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <math.h>       /* sin */

#define PI 3.14159265358979323


using namespace std;

//Electromagnetic vectors dimension
#define dim 200

//Kernel function to calculate the fields
__global__ void fdtd(double* ex, double* hy, double* aux, double* ca,double* cb, int dimension, double puls)
{
	//Wire id generation
	int id = threadIdx.x;





	if (id > 0)
	{
		//Ex vector calculus
		ex[id] = ca[id]*ex[id] + cb[id] * (hy[id - 1] - hy[id]);
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
	double* hst_e, * hst_h, * hst_aux, *hst_ca, * hst_cb;
	double* dev_e, * dev_h, * dev_aux, *dev_ca, * dev_cb;

	//Block declaration
	int bloques = 1;

	//Pulse parameter declaration
	double pulse;
	int kc = (int)dim / 2;
	double ddx = 0.01;        //Cell size
	double dt = ddx / 6e8;    //Time step size
	double freq_in = 700e6;

	//Dielectric Profile
	double epsz = 8.854e-12;
	double epsilon = 4;
	double sigma = 0.04;
	double eaf = dt * sigma / (2 * epsz * epsilon);

	//Number of steps done
	int nsteps =500;

	//Output files declaration (each one to store the fields values)
	ofstream ex;
	ofstream hy;

	//File opening
	ex.open("ex.txt");
	hy.open("hy.txt");

	//Host and device memory reserve
	hst_e = (double*)malloc(dim * sizeof(double));
	hst_h = (double*)malloc(dim * sizeof(double));
	hst_aux = (double*)malloc(2 * sizeof(double));
	hst_ca = (double*)malloc(dim * sizeof(double));
	hst_cb = (double*)malloc(dim * sizeof(double));

	cudaMalloc(&dev_e, dim * sizeof(double));
	cudaMalloc(&dev_h, dim * sizeof(double));
	cudaMalloc(&dev_aux, 2 * sizeof(double));
	cudaMalloc(&dev_ca, dim * sizeof(double));
	cudaMalloc(&dev_cb, dim * sizeof(double));

	//Vectors initialization
	for (int i = 0;i < dim;i++)
	{
		hst_e[i] = 0;
		hst_h[i] = 0;

		//Inicializo epsilon
		if (i < kc)
		{
			hst_ca[i] = 1;
			hst_cb[i] = 0.5;
		}
		else
		{
			hst_ca[i] = (1 - eaf) / (1 + eaf);
			hst_cb[i] = 0.5 / (epsilon * (1 + eaf));
			
		}
		//cout << hst_cb[i] << endl;
	}

	//Auxiliar vector for boundary conditions initialization
	hst_aux[0] = 0;
	hst_aux[1] = 0;

	//Host to device information movement
	cudaMemcpy(dev_e, hst_e, dim * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_h, hst_h, dim * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_aux, hst_aux, 2 * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_ca, hst_ca, dim * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_cb, hst_cb, dim * sizeof(double), cudaMemcpyHostToDevice);

	for (int i = 1;i < nsteps + 1;i++)
	{
		//Pulse generation
		pulse = sin(2 * PI * freq_in * dt * i);
		//cout << pulse << endl;
		//Kernel function call to use 200 wires (vectors dimension) to calculate the fields
		fdtd << <bloques, dim >> > (dev_e, dev_h, dev_aux, dev_ca, dev_cb, dim, pulse);
	}



	//Device to host information movement
	cudaMemcpy(hst_e, dev_e, dim * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(hst_h, dev_h, dim * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(hst_aux, dev_aux, 2 * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(hst_ca, dev_ca, dim * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(hst_cb, dev_cb, dim * sizeof(double), cudaMemcpyDeviceToHost);




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
	cudaFree(dev_ca);
	cudaFree(dev_cb);

	free(hst_e);
	free(hst_h);
	free(hst_aux);
	free(hst_ca);
	free(hst_cb);

	//File closing
	ex.close();
	hy.close();


	return 0;
}


