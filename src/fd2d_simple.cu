//FD2D in vioid

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <iomanip>
#include <fstream>

#include <cuda_runtime.h>

using namespace std;

__host__ void check_CUDA_Error(const char* mensaje)
{
	cudaError_t error;
	cudaDeviceSynchronize();
	error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		printf("ERROR %d: %s (%s)\n", error, cudaGetErrorString(error), mensaje);
		printf("\npulsa INTRO para finalizar...");
		fflush(stdin);
		char tecla = getchar();
		exit(-1);
	}
}

typedef struct
{
	double ez,
		dz,
		hx,
		hy,
		gaz;

}FDTDData;

void ini(FDTDData* field, int xdim, int ydim)
{
	for (int i = 0;i < xdim; i++) {
		for (int j = 0;j < ydim;j++) {

			field[j * xdim + i].ez = 0.0;
			field[j * xdim + i].dz = 0.0;
			field[j * xdim + i].hx = 0.0;
			field[j * xdim + i].hy = 0.0;
			field[j * xdim + i].gaz = 1.0;
		}
		//cout << field[i].ez_inc  << ",  ";
	}

}

__global__ void FD2D(FDTDData* field, int xdim, int ydim, int ts)
{
	int fil = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;





	int ic = int(xdim / 2),
		jc = int(ydim / 2);

	double ddx = 0.01,
		dt = ddx / 6e8,
		epsz = 8.854e-12;

	//pulse parameters
	double t0 = 20.0,
		spread = 6.0;

	//Calculate Dz
	if (0 < fil && fil < ydim && 0 < col && col < xdim) {
		field[fil * xdim + col].dz += 0.5 * (field[fil * xdim + col].hy - field[fil * xdim + col - 1].hy -
			field[fil * xdim + col].hx + field[(fil - 1) * xdim + col].hx);
	}
	__syncthreads();

	//Gaussian pulse in the middle
	double pulse = exp(-0.5 * ((t0 - ts) / spread) * ((t0 - ts) / spread));
	field[jc * xdim + ic].dz = pulse;
	__syncthreads();

	//Calculate the Ez field from Dz
	if (0 < fil && fil < ydim && 0 < col && col < xdim) {
		field[fil * xdim + col].ez = field[fil * xdim + col].gaz * field[fil * xdim + col].dz;
	}
	__syncthreads();

	//Calculate the Hx field
	if (fil < ydim - 1 && col < xdim - 1) {
		field[fil * xdim + col].hx += 0.5 * (field[fil * xdim + col].ez - field[(fil + 1) * xdim + col].ez);
	}
	__syncthreads();

	//Calculate the Hy field
	if (fil < ydim - 1 && col < xdim - 1) {
		field[fil * xdim + col].hy += 0.5 * (field[fil * xdim + col + 1].ez - field[fil * xdim + col].ez);
	}

}

int main(void)
{
	int dimx = 1000,
		dimy = 1000,
		dimension = dimx * dimy,
		FDTDDatasize = sizeof(FDTDData),
		datanumbytes = dimension * FDTDDatasize;

	//Initialization
	FDTDData* hst_field;
	hst_field = (FDTDData*)malloc(datanumbytes);

	ini(hst_field, dimx, dimy);

	FDTDData* dev_field;
	cudaMalloc((void**)&dev_field, datanumbytes);
	check_CUDA_Error("Error in statement 1");

	cudaMemcpy(dev_field, hst_field, datanumbytes, cudaMemcpyHostToDevice);
	check_CUDA_Error("Error in statement 2");

	dim3 blocks(32, 32);
	dim3 threads(32, 32);
	int nsteps = 1000;

	for (int time_step = 1;time_step < nsteps + 1;time_step++)
	{
		FD2D << <blocks, threads>> > (dev_field, dimx, dimy, time_step);
		check_CUDA_Error("Error in statement 3");
	}

	cudaMemcpy(hst_field, dev_field, datanumbytes, cudaMemcpyDeviceToHost);
	check_CUDA_Error("error 1");

	//Matrix writing 
	ofstream ez;
	ez.open("ez.txt");

	for (int j = 0;j < dimx; j++) {
		for (int i = 0;i < dimy;i++) {

			ez << hst_field[j * dimx + i].ez << ",  ";
		}
		ez << endl;
	}

	ez.close();

	//Free memory
	free(hst_field);

	cudaFree(dev_field);

	return 0;
}