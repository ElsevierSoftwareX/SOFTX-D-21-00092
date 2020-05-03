#ifndef H_MV_CLASS
#define H_MV_CLASS

#include <iostream>
#include <stdlib.h>
#include <complex>
#include <complex.h>
#include "config.h"

#include <omp.h>

#include <mpi.h>

#include "mpi_class.h"

#include <random>


class MV_class{

	public:

	double g_parameter;
	double mu_parameter;
	int Ny_parameter;

	MV_class(double g, double mu, int N){

		g_parameter = g;
		mu_parameter = mu;
		Ny_parameter = N;
	}
};

#endif
