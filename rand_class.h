#ifndef H_RAND_CLASS
#define H_RAND_CLASS

#include <iostream>
#include <stdlib.h>
#include <complex>
#include <complex.h>
#include "config.h"

#include <omp.h>

#include <mpi.h>

#include "mpi_class.h"

#include <random>


class rand_class{

	public:

	std::ranlux48 rgenerator;
	std::uniform_real_distribution<double> distribution{0.0,1.0};

	rand_class(mpi_class *mpi, config *cnfg){

	rgenerator.seed(cnfg->seed + mpi->getRank());

	}

	double get(){
	
		return distribution(rgenerator);
	
	}
};

#endif
