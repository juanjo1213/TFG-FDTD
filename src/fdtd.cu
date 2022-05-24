
#include <cuda_runtime.h>

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
