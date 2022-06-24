//JUAN JOSE SALAZAR LOPEZ
// 
//FD2D host implementation

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <iomanip>
#include <fstream>

using namespace std;

typedef struct
{
	double ez,
		dz,
		hx,
		hy;

}FDTDData;

void ini(FDTDData* field, int xdim, int ydim)
{
	for (int i = 0;i < xdim; i++) {
		for (int j = 0;j < ydim;j++) {

			field[j * xdim + i].ez = 0.0;
			field[j * xdim + i].dz = 0.0;
			field[j * xdim + i].hx = 0.0;
			field[j * xdim + i].hy = 0.0;
		}
		//cout << field[i].ez_inc  << ",  ";
	}

}

void FD2D(FDTDData* field, int xdim, int ydim, int nsteps)
{
	int ic = int(xdim / 2),
		jc = int(ydim / 2);

	//Medium conditions
	double* gaz;

	gaz = (double*)malloc(xdim * ydim * sizeof(double));

	for (int j = 0;j < ydim;j++) {
		for (int i = 0;i < xdim;i++) {
			gaz[j * xdim + i] = 1.0;
		}
	}


	double ddx = 0.01,         //cell size
		   dt = ddx / 6e8,     //Time step size
		   epsz = 8.854e-12;   //Dielectric profile



	//Pulse parameters
	double t0 = 20.0,
		spread = 6.0;

	//Main loop
	for (int time_step = 1;time_step < nsteps + 1;time_step++)
	{
		//Calculate Dz
		for (int j = 1;j < ydim;j++) {
			for (int i = 1;i < xdim;i++) {
				field[j*xdim+i].dz= field[j * xdim + i].dz+0.5*(field[j*xdim+i].hy- field[j * xdim + i-1].hy-
					                                            field[j*xdim+i].hx+ field[(j-1) * xdim +i].hx);
			}
		}

		//Gaussian pulse in the middle
		double pulse = exp(-0.5 * ((t0 - time_step) / spread) * ((t0 - time_step) / spread));
		field[jc * xdim + ic].dz = pulse;

		//Calculate the Ez field from Dz
		for (int j = 1;j < ydim;j++) {
			for (int i = 1;i < xdim;i++) {
				field[j * xdim + i].ez = gaz[j * xdim + i] * field[j * xdim + i].dz;
			}
		}

		//Calculate the Hx field
		for (int j = 0;j < ydim-1;j++) {
			for (int i = 0;i < xdim-1;i++) {
				field[j * xdim + i].hx = field[j * xdim + i].hx + 0.5 * (field[j * xdim + i].ez - field[(j + 1) * xdim + i].ez);
			}
		}

		//Calculate the Hy field
			//Calculate the Hx field
		for (int j = 0;j < ydim - 1;j++) {
			for (int i = 0;i < xdim - 1;i++) {
				field[j * xdim + i].hy = field[j * xdim + i].hy + 0.5 * (field[j * xdim + i + 1].ez - field[j * xdim + i].ez);
			}
		}

	}




}


int main(void)
{
	int dimx = 60,
		dimy = 60,
		dimension = dimx * dimy,
		FDTDDatasize = sizeof(FDTDData),
		datanumbytes = dimension * FDTDDatasize;

	//Initialization
	FDTDData* hst_field;
	hst_field = (FDTDData*)malloc(datanumbytes);

	ini(hst_field, dimx, dimy);

	//Pasos teporales
	clock_t time;
	int n_steps = 50;

	time = clock();
	//Main Loop
	FD2D(hst_field, dimx, dimy, n_steps);
	time = clock() - time;

	cout  << "Time working: " << (float)time / CLOCKS_PER_SEC << "seconds";

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

	return 0;
}