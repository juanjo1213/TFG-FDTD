//Calculus of the electromagnetic field using FDTD 2D in host

//Libraries
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <iomanip>
#include <fstream>
#include <ctime>


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

void FDTD(FDTDData* data, PML* pml, INCFIELD* field, int xdim, int ydim, int nsteps)
{
	int ia = 7,
		ib = xdim - ia - 1,
		ja = 7,
		jb = ydim - ja - 1;

	//Pulse parameters and steps
	double pulse,
		t0 = 25.0,
		spread = 8.0;


	double low1 = 0.0,
		low2 = 0.0,
		high1 = 0.0,
		high2 = 0.0;



	for (int time_step = 1;time_step < nsteps + 1;time_step++)
	{
		
		//Incident Ez values
		for (int j = 1;j < ydim;j++) {
			field[j].ez_inc = field[j].ez_inc + 0.5 * (field[j - 1].hx_inc - field[j].hx_inc);
		}
		
		//Absorbing boundary conditions
		field[0].ez_inc = low1;
		low1 = low2;
		low2 = field[1].ez_inc;

		field[ydim - 1].ez_inc = high1;
		high1 = high2;
		high2 = field[ydim - 2].ez_inc;

		//Calculate the Dz field
		for (int j = 1;j < ydim;j++) {
			for (int i = 1; i < xdim;i++) {
				data[j * xdim + i].dz = pml[i].gi3 * pml[j].gj3 * data[j * xdim + i].dz
					+ pml[i].gi2 * pml[j].gj2 * 0.5 * (data[j * xdim + i].hy - data[j * xdim + i - 1].hy
						- data[j * xdim + i].hx + data[(j - 1) * xdim + i].hx);
			}
		}

		//Source
		pulse = exp(-0.5 * ((t0 - time_step) / spread) * ((t0 - time_step) / spread));
		field[3].ez_inc = pulse;
		
		//Incident Dz values
		for (int i = ia; i < ib + 1;i++) {
			data[ja * xdim + i].dz = data[ja * xdim + i].dz + 0.5 * field[ja - 1].hx_inc;
			data[jb * xdim + i].dz = data[jb * xdim + i].dz - 0.5 * field[jb].hx_inc;
		}

		//Calculate the Ez field
		for (int j = 0; j < ydim;j++) {
			for (int i = 0;i < xdim;i++) {
				data[j * xdim + i].ez = data[j * xdim + i].gaz * (data[j * xdim + i].dz - data[j * xdim + i].iz);
				data[j * xdim + i].iz = data[j * xdim + i].iz + data[j * xdim + i].gbz * data[j * xdim + i].ez;
			}
		}

		//Calculate the incident Hx
		for (int j = 0;j < ydim - 1;j++) {
			field[j].hx_inc = field[j].hx_inc + 0.5 * (field[j].ez_inc - field[j + 1].ez_inc);
		}

		//Calculate the Hx field
		for (int j = 0;j < ydim - 1;j++) {
			for (int i = 0;i < xdim - 1;i++) {
				data[j * xdim + i].curl_e = data[j * xdim + i].ez - data[(j + 1) * xdim + i].ez;
				data[j * xdim + i].ihx = data[j * xdim + i].ihx + data[j * xdim + i].curl_e;
				data[j * xdim + i].hx = pml[j].fj3 * data[j * xdim + i].hx + pml[j].fj2 *
					(0.5 * data[j * xdim + i].curl_e + pml[i].fi1 * data[j * xdim + i].ihx);
			}
		}

		//Incident Hx values
		for (int i = ia;i < ib + 1;i++) {
			data[(ja - 1) * xdim + i].hx = data[(ja - 1) * xdim + i].hx + 0.5 * field[ja].ez_inc;
			data[jb * xdim + i].hx = data[jb * xdim + i].hx - 0.5 * field[jb].ez_inc;
		}

		//Calculate the Hy field
		for (int j = 0;j < ydim;j++) {
			for (int i = 0;i < xdim - 1;i++) {
				data[j * xdim + i].curl_e = data[j * xdim + i].ez - data[j * xdim + i + 1].ez;
				data[j * xdim + i].ihy = data[j * xdim + i].ihy + data[j * xdim + i].curl_e;
				data[j * xdim + i].hy = pml[i].fi3 * data[j * xdim + i].hy - pml[i].fi2 *
					(0.5 * data[j * xdim + i].curl_e + pml[j].fj1 * data[j * xdim + i].ihy);
			}
		}

		//Incident Hy values
		for (int j = ja;j < jb + 1;j++) {
			data[j * xdim + ia - 1].hy = data[j * xdim + ia - 1].hy - 0.5 * field[j].ez_inc;
			data[j * xdim + ib].hy = data[j * xdim + ib].hy + 0.5 * field[j].ez_inc;
		}
		//cout << setprecision(17) << data[14 * xdim + 36].ez << endl;
	}

}



int main(void)
{
	int dimx = 50,
		dimy = 50,
		dimension = dimx * dimy,
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
	hst_field = (INCFIELD*)malloc(INCFIELDbytes);

	ini(hst_data, hst_field, dimx, dimy);

	//Host FDTD
	clock_t time;
	int n_steps =100;

	time = clock();
	FDTD(hst_data, hst_pml, hst_field, dimx, dimy, n_steps);
	time = clock() - time;
	
	//Calculated

	cout << "Time working: " << time/CLOCKS_PER_SEC<<"seconds";

	//Matrix writing 
	ofstream ez;
	ez.open("hst_ez.txt");

	for (int i = 0;i < dimx; i++) {
		for (int j = 0;j < dimy;j++) {

			ez << setprecision(17) << hst_data[j * dimx + i].ez << "     ";

		}
		ez << endl;
	}

	ez.close();

	//Free memory
	free(hst_data);
	free(hst_pml);
	free(hst_field);

	return 0;
}