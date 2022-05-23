//Calculus of the electromagnetic field using the FDTD method in Cuda with boundary condition so we have no reflected wave.
// In this case it is set a dielectric medium at the half right of the space, with epsilon=4.
// We use a sinusoidal function as pulse
// It is considered a lossy dielectric medium
// Now we generalize the method including the vector D
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
__global__ void fdtd(double* ex, double* hy, double* aux, double* gax, double* gbx, double* dx, double* ix, int dimension, double puls)
{
	//Wire id generation
	int id = threadIdx.x;





	if (id > 0)
	{
		//dx vector calculus
		dx[id] = dx[id] + 0.5 * (hy[id - 1] - hy[id]);
	}



	//Pulse generation
	dx[5] = puls + dx[5];

	if (id > 0)
	{
		//Ex calculus from Dx field
		ex[id] = gax[id] * (dx[id] - ix[id]);
		ix[id] = ix[id] + gbx[id] * ex[id];
	}

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
	double* hst_e, * hst_h, * hst_aux, * hst_gax, * hst_gbx, *hst_dx, *hst_ix;
	double* dev_e, * dev_h, * dev_aux, * dev_gax, * dev_gbx, *dev_dx, *dev_ix;

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
	double epsr = 4;
	double epsilon = 4;
	double sigma = 0.04;
	double eaf = dt * sigma / (2 * epsz * epsilon);

	//Number of steps done
	int nsteps = 500;

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
	hst_gax = (double*)malloc(dim * sizeof(double));
	hst_gbx = (double*)malloc(dim * sizeof(double));
	hst_dx = (double*)malloc(dim * sizeof(double));
	hst_ix = (double*)malloc(dim * sizeof(double));

	cudaMalloc(&dev_e, dim * sizeof(double));
	cudaMalloc(&dev_h, dim * sizeof(double));
	cudaMalloc(&dev_aux, 2 * sizeof(double));
	cudaMalloc(&dev_gax, dim * sizeof(double));
	cudaMalloc(&dev_gbx, dim * sizeof(double));
	cudaMalloc(&dev_dx, dim * sizeof(double));
	cudaMalloc(&dev_ix, dim * sizeof(double));

	//Vectors initialization
	for (int i = 0;i < dim;i++)
	{
		hst_e[i] = 0;
		hst_h[i] = 0;
		hst_dx[i] = 0;
		hst_ix[i] = 0;

		//Inicializo epsilon
		if (i < kc)
		{
			hst_gax[i] = 1;
			hst_gbx[i] = 0;
		}
		else
		{
			hst_gax[i] = 1 / (epsr + (sigma * dt / epsz));
			hst_gbx[i] = sigma * dt / epsz;

		}
		//cout << hst_gbx[i] << endl;
	}

	//Auxiliar vector for boundary conditions initialization
	hst_aux[0] = 0;
	hst_aux[1] = 0;

	//Host to device information movement
	cudaMemcpy(dev_e, hst_e, dim * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_h, hst_h, dim * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_aux, hst_aux, 2 * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_gax, hst_gax, dim * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_gbx, hst_gbx, dim * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_dx, hst_dx, dim * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_ix, hst_ix, dim * sizeof(double), cudaMemcpyHostToDevice);

	for (int i = 1;i < nsteps + 1;i++)
	{
		//Pulse generation
		pulse = sin(2 * PI * freq_in * dt * i);
		//cout << pulse << endl;
		//Kernel function call to use 200 wires (vectors dimension) to calculate the fields
		fdtd << <bloques, dim >> > (dev_e, dev_h, dev_aux, dev_gax, dev_gbx,dev_dx,dev_ix, dim, pulse);
	}



	//Device to host information movement
	cudaMemcpy(hst_e, dev_e, dim * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(hst_h, dev_h, dim * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(hst_aux, dev_aux, 2 * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(hst_gax, dev_gax, dim * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(hst_gbx, dev_gbx, dim * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(hst_dx, dev_dx, dim * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(hst_ix, dev_ix, dim * sizeof(double), cudaMemcpyDeviceToHost);




	//Results writing 
	for (int i = 0;i < dim;i++)
	{
		ex << hst_e[i] << endl;
		hy << hst_h[i] << endl;
		//cout << hst_e[i] << endl;
	}


	// Device and host memory release
	cudaFree(dev_e);
	cudaFree(dev_h);
	cudaFree(dev_aux);
	cudaFree(dev_gax);
	cudaFree(dev_gbx);
	cudaFree(dev_dx);
	cudaFree(dev_ix);

	free(hst_e);
	free(hst_h);
	free(hst_aux);
	free(hst_gax);
	free(hst_gbx);
	free(hst_dx);
	free(hst_ix);

	//File closing
	ex.close();
	hy.close();


	return 0;
}


