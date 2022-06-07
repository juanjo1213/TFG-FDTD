//Calculus of the electromagnetic field using FDTD 2D 

//Libraries
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <iomanip>
#include <fstream>
#include <ctime>

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
		iz,
		ihx,
		ihy,
		gaz,
		gbz,
		curl_e;

}FDTDData;

typedef struct
{
	double gi2,
		gi3,
		fi1,
		fi2,
		fi3,

		gj2,
		gj3,
		fj1,
		fj2,
		fj3;
}PML;

typedef struct
{
	double ez_inc,
		hx_inc;

}INCFIELD;

__host__ void ini(FDTDData* data, INCFIELD* field, int xdim, int ydim)
{
	for (int i = 0;i < xdim; i++) {
		for (int j = 0;j < ydim;j++) {

			data[j * xdim + i].ez = 0.0;
			data[j * xdim + i].dz = 0.0;
			data[j * xdim + i].hx = 0.0;
			data[j * xdim + i].hy = 0.0;
			data[j * xdim + i].iz = 0.0;
			data[j * xdim + i].ihx = 0.0;
			data[j * xdim + i].ihy = 0.0;
			data[j * xdim + i].curl_e = 0.0;
		}
		field[i].ez_inc = 0.0;
		field[i].hx_inc = 0.0;
		//cout << field[i].ez_inc  << ",  ";
	}

}

//PML generation, it receives PML class object and x,y dimension
__host__ void PMLGEN(PML* parameter, int x, int y)
{
	//Tamaño
	int npml = 8;

	for (int n = 0; n < x;n++)
	{
		parameter[n].gi2 = 1.0;
		parameter[n].gi3 = 1.0;
		parameter[n].fi1 = 0.0;
		parameter[n].fi2 = 1.0;
		parameter[n].fi3 = 1.0;

	}

	for (int n = 0; n < y;n++)
	{
		parameter[n].gj2 = 1.0;
		parameter[n].gj3 = 1.0;
		parameter[n].fj1 = 0.0;
		parameter[n].fj2 = 1.0;
		parameter[n].fj3 = 1.0;

	}

	for (int n = 0; n < npml;n++)
	{
		int xnum = npml - n;
		double xd = npml;
		double xxn = (double)xnum / xd;
		double xn = 0.33 * xxn * xxn * xxn;

		parameter[n].gi2 = 1 / (1 + xn);
		parameter[x - 1 - n].gi2 = 1 / (1 + xn);
		parameter[n].gi3 = (1 - xn) / (1 + xn);
		parameter[x - 1 - n].gi3 = (1 - xn) / (1 + xn);

		parameter[n].gj2 = 1 / (1 + xn);
		parameter[y - 1 - n].gj2 = 1 / (1 + xn);
		parameter[n].gj3 = (1 - xn) / (1 + xn);
		parameter[y - 1 - n].gj3 = (1 - xn) / (1 + xn);

		xxn = (xnum - 0.5) / xd;
		xn = 0.33 * xxn * xxn * xxn;

		parameter[n].fi1 = xn;
		parameter[x - 2 - n].fi1 = xn;
		parameter[n].fi2 = 1 / (1 + xn);
		parameter[x - 2 - n].fi2 = 1 / (1 + xn);
		parameter[n].fi3 = (1 - xn) / (1 + xn);
		parameter[x - 2 - n].fi3 = (1 - xn) / (1 + xn);

		parameter[n].fj1 = xn;
		parameter[y - 2 - n].fj1 = xn;
		parameter[n].fj2 = 1 / (1 + xn);
		parameter[y - 2 - n].fj2 = 1 / (1 + xn);
		parameter[n].fj3 = (1 - xn) / (1 + xn);
		parameter[y - 2 - n].fj3 = (1 - xn) / (1 + xn);

	}

}

__host__ void cylinder(FDTDData* data, int xdim, int ydim)
{
	double epsr = 30.0,
		sigma = 0.3,
		radius = 200.0,
		epsz = 8.854e-12,
		ddx = 0.01,      //Cell size
		dt = ddx / 6e8,  //Time step size
		xdist = 0.0,
		ydist = 0.0,
		dist = 0.0;


	int xc = int(xdim / 2 - 1),
		yc = int(ydim / 2 - 1);

	for (int i = 0;i < xdim;i++)
	{
		for (int j = 0;j < ydim;j++)
		{
			xdist = (xc - i);
			ydist = (yc - j);
			dist = sqrt(xdist * xdist + ydist * ydist);
			if (dist <= radius) {
				data[j * xdim + i].gaz = 1 / (epsr + (sigma * dt / epsz));
				data[j * xdim + i].gbz = (sigma * dt / epsz);
			}
			else {
				data[j * xdim + i].gaz = 1.0;
				data[j * xdim + i].gbz = 0.0;
			}

		}
	}
}

__global__ void FDTD(FDTDData* field, PML* pml, INCFIELD* inc_field, double* aux, int xdim, int ydim, int ts)
{
	int fil = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	int ia = 7,
		ib = xdim - ia - 1,
		ja = 7,
		jb = ydim - ja - 1;

	//Pulse parameters and boundary conditions
	double t0 = 25.0,
		spread = 8.0;

	//Incident Ez values
	if (0 < col && col < xdim && fil == 0) {
		inc_field[fil * xdim + col].ez_inc += 0.5 * (inc_field[fil * xdim + col - 1].hx_inc - inc_field[fil * xdim + col].hx_inc);
	}
	__syncthreads();

	//Abdorbing boundary conditions
	inc_field[0].ez_inc = aux[0];
	aux[0] = aux[1];
	aux[1] = inc_field[1].ez_inc;

	inc_field[ydim - 1].ez_inc = aux[2];
	aux[2] = aux[3];
	aux[3] = inc_field[ydim - 2].ez_inc;

	//Calculate the Dz field
	if (0 < fil && fil < ydim && 0 < col && col < xdim) {
		field[fil * xdim + col].dz *= pml[col].gi3 * pml[fil].gj3;
	}

	__syncthreads();
	if (0 < fil && fil < ydim && 0 < col && col < xdim) {
		field[fil * xdim + col].dz += pml[col].gi2 * pml[fil].gj2 * 0.5 * (field[fil * xdim + col].hy - field[fil * xdim + col - 1].hy -
			field[fil * xdim + col].hx + field[(fil - 1) * xdim + col].hx);
	}
	__syncthreads();

	//Source
	double pulse = exp(-0.5 * ((t0 - ts) / spread) * ((t0 - ts) / spread));
	inc_field[3].ez_inc = pulse;


	//Incident Dz values
	if (ia < col && col < ib + 1 && fil == ja) {
		field[fil * xdim + col].dz += +0.5 * inc_field[fil - 1].hx_inc;
	}
	__syncthreads();
	if (ia < col && col < ib + 1 && fil == jb) {
		field[fil * xdim + col].dz += -0.5 * inc_field[fil].hx_inc;
	}
	__syncthreads();

	//Calculate the Ez field
	if (fil < ydim && col < xdim) {
		field[fil * xdim + col].ez = field[fil * xdim + col].gaz * (field[fil * xdim + col].dz - field[fil * xdim + col].iz);
	}
	__syncthreads();
	if (fil < ydim && col < xdim) {
		field[fil * xdim + col].iz += field[fil * xdim + col].gbz * field[fil * xdim + col].ez;
	}
	__syncthreads();

	//Calculate the incident Hx
	if (col < xdim - 1 && fil == 0) {
		inc_field[fil * xdim + col].hx_inc += 0.5 * (inc_field[fil * xdim + col].ez_inc - inc_field[fil * xdim + col + 1].ez_inc);
	}
	__syncthreads();

	//Calculate the Hx field
	if (fil < ydim - 1 && col < xdim - 1) {
		field[fil * xdim + col].curl_e = field[fil * xdim + col].ez - field[(fil + 1) * xdim + col].ez;
	}
	__syncthreads();
	if (fil < ydim - 1 && col < xdim - 1) {
		field[fil * xdim + col].ihx += +field[fil * xdim + col].curl_e;
	}
	__syncthreads();
	if (fil < ydim - 1 && col < xdim - 1) {
		field[fil * xdim + col].hx *= pml[fil].fj3;
	}
	__syncthreads();
	if (fil < ydim - 1 && col < xdim - 1) {
		field[fil * xdim + col].hx += pml[fil].fj2 * (0.5 * field[fil * xdim + col].curl_e + pml[col].fi1 * field[fil * xdim + col].ihx);
	}
	__syncthreads();

	//Incident Hx values
	if (ia < col && col < ib + 1 && fil == ja - 1) {
		field[fil * xdim + col].hx += 0.5 * inc_field[fil + 1].ez_inc;
	}
	if (ia < col && col < ib + 1 && fil == jb) {
		field[fil * xdim + col].hx += -0.5 * inc_field[fil].ez_inc;
	}
	__syncthreads();

	//Calculate the Hy field
	if (fil < ydim && col < xdim - 1) {
		field[fil * xdim + col].curl_e = field[fil * xdim + col].ez - field[fil * xdim + col + 1].ez;
	}
	__syncthreads();
	if (fil < ydim && col < xdim - 1) {
		field[fil * xdim + col].ihy += field[fil * xdim + col].curl_e;
	}
	__syncthreads();
	if (fil < ydim && col < xdim - 1) {
		field[fil * xdim + col].hy *= pml[col].fi3;
	}
	__syncthreads();
	if (fil < ydim && col < xdim - 1) {
		field[fil * xdim + col].hy += -pml[col].fi2 * (0.5 * field[fil * xdim + col].curl_e + pml[fil].fj1 * field[fil * xdim + col].ihy);
	}
	__syncthreads();

	//Incident Hy values
	if (ja < fil && fil < jb + 1 && col == ia - 1) {
		field[fil * xdim + col].hy += -0.5 * inc_field[fil].ez_inc;
	}
	if (ja < fil && fil < jb + 1 && col == ib) {
		field[fil * xdim + col].hy += +0.5 * inc_field[fil].ez_inc;
	}
	__syncthreads();

}


int main(void)
{
	int dimx = 2000,
		dimy = 2000,
		dimension = dimx * dimy,
		FDTDDatasize = sizeof(FDTDData),
		INCFIELDsize = sizeof(INCFIELD),
		PMLsize = sizeof(PML),
		FDTDDatabytes = FDTDDatasize * dimension,
		INCFIELDbytes = INCFIELDsize * dimx,
		PMLbytes = PMLsize * dimx;

	//Initialization
	FDTDData* hst_field;
	hst_field = (FDTDData*)malloc(FDTDDatabytes);

	INCFIELD* hst_incfield;
	hst_incfield = (INCFIELD*)malloc(INCFIELDbytes);

	ini(hst_field, hst_incfield, dimx, dimy);

	//Cylinder
	cylinder(hst_field, dimx, dimy);

	//Create PML
	PML* hst_pml;
	hst_pml = (PML*)malloc(PMLbytes);

	PMLGEN(hst_pml, dimx, dimy);

	//FDTD
	FDTDData* dev_field;
	PML* dev_pml;
	INCFIELD* dev_incfield;

	cudaMalloc((void**)&dev_field, FDTDDatabytes);
	check_CUDA_Error("Error in statement 1");
	cudaMalloc((void**)&dev_pml, PMLbytes);
	check_CUDA_Error("Error in statement 2");
	cudaMalloc((void**)&dev_incfield, INCFIELDbytes);
	check_CUDA_Error("Error in statement 3");

	cudaMemcpy(dev_field, hst_field, FDTDDatabytes, cudaMemcpyHostToDevice);
	check_CUDA_Error("Error in statement 4");
	cudaMemcpy(dev_pml, hst_pml, PMLbytes, cudaMemcpyHostToDevice);
	check_CUDA_Error("Error in statement 5");
	cudaMemcpy(dev_incfield, hst_incfield, INCFIELDbytes, cudaMemcpyHostToDevice);
	check_CUDA_Error("Error in statement 6");

	// Time recording
	cudaEvent_t start;
	cudaEvent_t stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//auxiliar vector for calculus
	double* hst_aux, * dev_aux;

	hst_aux = (double*)malloc(4 * sizeof(double));
	cudaMalloc((void**)&dev_aux, 4 * sizeof(double));
	check_CUDA_Error("Error in statement 7");
	for (int i = 0;i < 4;i++) hst_aux[i] = 0.0;
	cudaMemcpy(dev_aux, hst_aux, 4 * sizeof(double), cudaMemcpyHostToDevice);
	check_CUDA_Error("Error in statement 4");


	int n_steps = 10000;

	dim3 blocks(100, 100), threads(25, 25);

	//Main loop
	cudaEventRecord(start, 0);

	for (int time_step = 1;time_step < n_steps + 1; time_step++)
	{
		FDTD << <blocks, threads >> > (dev_field, dev_pml, dev_incfield, dev_aux, dimx, dimy, time_step);
		check_CUDA_Error("Error in main loop");
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	// Time calculus
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout << elapsedTime;

	cudaMemcpy(hst_field, dev_field, FDTDDatabytes, cudaMemcpyDeviceToHost);
	check_CUDA_Error("Error in statement 8");
	cudaMemcpy(hst_pml, dev_pml, PMLbytes, cudaMemcpyDeviceToHost);
	check_CUDA_Error("Error in statement 9");
	cudaMemcpy(hst_incfield, dev_incfield, INCFIELDbytes, cudaMemcpyDeviceToHost);
	check_CUDA_Error("Error in statement 10");


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
	free(hst_pml);
	free(hst_incfield);
	free(hst_aux);

	cudaFree(dev_field);
	cudaFree(dev_pml);
	cudaFree(dev_incfield);
	cudaFree(dev_aux);

	return 0;
}