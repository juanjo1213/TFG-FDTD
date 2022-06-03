//Calculus of the electromagnetic field using FDTD in 2D

//Libraries
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <iomanip>

#include <cuda_runtime.h>

using namespace std;

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

void ini(FDTDData* data, INCFIELD* field, int xdim, int ydim)
{
	for (int i = 0;i < xdim; i++) {
		for (int j = 0;j < ydim;j++) {

			    data[j * xdim + i].ez  = 0.0; 
				data[j * xdim + i].dz = 0.0;
				data[j * xdim + i].hx  = 0.0; 
				data[j * xdim + i].hy  = 0.0; 
				data[j * xdim + i].iz  = 0.0; 
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
void PMLGEN(PML* parameter, int x, int y)
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

	for (int n=0; n < npml;n++)
	{
		int xnum = npml - n;
		double xd = npml;
		double xxn =(double) xnum / xd;
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

void cilinder(FDTDData* data, int xdim, int ydim)
{
	double epsr = 30.0,
		sigma = 0.3,
		radius = 10.0,
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

__global__ void FDTD(FDTDData* data, PML* pml, INCFIELD* field, int xdim, int ydim)
{
	
	int fil = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int ia = 7,
		ib = xdim - ia - 1,
		ja = 7,
		jb = xdim - ja - 1;

	//Boundary Conditions parameters
	double low1 = 0.0,
		   low2 = 0.0,
		   high1 = 0.0,
		   high2 = 0.0;
	//Pulse parameters and steps
	double pulse,
		   t0 = 20.0,
		   spread = 8.0;
	int nsteps = 100;

	
	for (int time_step = 1;time_step < nsteps + 1;time_step++)
	{
		//Incident Ez values
		if (0 < fil && fil<ydim) {
			field[fil].ez_inc = field[fil].ez_inc + 0.5 * (field[fil - 1].hx_inc - field[fil].hx_inc);
		}

		//Absorbing Boundary Conditions
		field[0].ez_inc = low1;
		low1 = low2;
		low2 = field[1].ez_inc;

		field[ydim - 1].ez_inc = high1;
		high1 = high2;
		high2 = field[ydim - 2].ez_inc;
		
		//Calculate the Dz field
		if (0 < fil && fil < ydim && 0 < col && col < xdim) {
			data[fil * xdim + col].dz = pml[col].gi3 * pml[fil].gj3 * data[fil * xdim + col].dz
				+ pml[col].gi2 * pml[fil].gj2 * 0.5 * (data[fil * xdim + col].hy - data[fil  * xdim + col-1].hy
					- data[fil * xdim + col].hx + data[(fil-1) * xdim + col ].hx);
		}
		
		//Source
		pulse = exp(-0.5 * ((t0 - time_step) / spread) * ((t0 - time_step) / spread));
		field[3].ez_inc = pulse;
		
		//Incident Dz values
		if (ia <= col && col < ib + 1) {
			data[ja * xdim + col].dz = data[ja * xdim + col].dz + 0.5 * field[ja - 1].hx_inc;
			data[jb * xdim + col].dz = data[jb * xdim + col].dz - 0.5 * field[jb].hx_inc;
		}
		
		//Calculate the Ez field
		if (fil < ydim && col < xdim) {
			data[fil * xdim + col].ez = data[fil * xdim + col].gaz * (data[fil * xdim + col].dz - data[fil * xdim + col].iz);
			data[fil * xdim + col].iz = data[fil * xdim + col].iz + data[fil * xdim + col].gbz * data[fil * xdim + col].ez;
		}
		
		//Calculate the Incident Hx
		if (fil < ydim - 1) {
			field[fil].hx_inc = field[fil].hx_inc + 0.5 * (field[fil].ez_inc - field[fil + 1].ez_inc);
		}
		
		//Calculate the Hx field
		if(fil<ydim-1 && col<xdim-1){
			data[fil * xdim + col].curl_e = data[fil * xdim + col].ez - data[(fil + 1) * xdim + col].ez;
			data[fil * xdim + col].ihx = data[fil * xdim + col].ihx + data[fil * xdim + col].curl_e;
			data[fil * xdim + col].hx = pml[fil].fj3 * data[fil * xdim + col].hx + pml[fil].fj2 *
				(0.5 * data[fil * xdim + col].curl_e + pml[col].fi1 * data[fil * xdim + col].ihx);
		}
		
		//Incident Hx values
		if (ia < col && col < ib + 1) {
			data[(ja-1) * xdim + col].hx = data[(ja - 1) * xdim + col].hx + 0.5 * field[ja].ez_inc;
			data[jb * xdim + col].hx = data[jb * xdim + col].hx - 0.5 * field[jb].ez_inc;
		}
		
		//Calculate the Hy field
		if (col < xdim - 1 && fil < ydim) {
			data[fil * xdim + col].curl_e = data[fil * xdim + col].ez - data[fil * xdim + col + 1].ez;
			data[fil * xdim + col].ihy = data[fil * xdim + col].ihy + data[fil * xdim + col].curl_e;
			data[fil * xdim + col].hy = pml[col].fi3 * data[fil * xdim + col].hy - pml[col].fi2 *
				(0.5 * data[fil * xdim + col].curl_e + pml[fil].fj1 * data[fil * xdim + col].ihy);
		}
		
		//Incident Hy values
		if (ja < fil && fil < jb + 1) {
			data[fil * xdim + ia - 1].hy = data[fil * xdim + ia - 1].hy - 0.5 * field[fil].ez_inc;
			data[fil * xdim + ib].hy = data[fil * xdim + ib].hy + 0.5 * field[fil].ez_inc;
		}
		
	}

	/*
	int fil = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (fil<xdim && col<ydim) {

		data[fil * xdim + col].ez = 6;
	}
	*/
	
}

int main(void)
{
	int dimx = 50,
		dimy = 50,
		dimension = dimx * dimy,
		blocksize = dimension,
		FDTDDatasize = sizeof(FDTDData),
		PMLsize = sizeof(PML),
		INCFIELDsize = sizeof(INCFIELD),
		INCFIELDbytes = dimx * INCFIELDsize,
		PMLnumbytes = dimx * PMLsize,
		datanumbytes = dimension * FDTDDatasize;
	

	//Create PML
	PML* hst_pml;	
	hst_pml = (PML*)malloc(PMLnumbytes);
		
	PMLGEN(hst_pml, dimx, dimy);

	//Create the cylinder
	FDTDData* hst_data;
	hst_data = (FDTDData*)malloc(datanumbytes);

	cilinder(hst_data, dimx, dimy);

	//Initialization
	INCFIELD* hst_field;
	hst_field= (INCFIELD*)malloc(INCFIELDbytes);

	ini(hst_data, hst_field, dimx, dimy);

	//FDTD
	FDTDData* dev_data;
	PML* dev_pml;
	INCFIELD* dev_field;

	cudaMalloc((void**)&dev_data, datanumbytes);
	cudaMalloc((void**)&dev_pml, PMLnumbytes);
	cudaMalloc((void**)&dev_field, INCFIELDbytes);

	cudaMemcpy(dev_data, hst_data, datanumbytes, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_pml, hst_pml, PMLnumbytes, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_field, hst_field, INCFIELDbytes, cudaMemcpyHostToDevice);

	dim3 blocks(2,2);
	dim3 threads(25, 25);

	//main loop
	FDTD <<<blocks, threads >>> (dev_data, dev_pml, dev_field, dimx, dimy);

	cudaMemcpy(hst_data, dev_data, datanumbytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(hst_pml, dev_pml, PMLnumbytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(hst_field, dev_field, INCFIELDbytes, cudaMemcpyDeviceToHost);

	
	for (int i = 0;i < dimx; i++) {
		for (int j = 0;j <dimy;j++) {

			cout <<setprecision(6) << hst_data[j * dimx + i].curl_e << ",  ";
		}
		cout << endl;
	}
	

	return 0;
}