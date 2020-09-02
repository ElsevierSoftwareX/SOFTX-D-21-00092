#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <ctime>

#include <complex.h>

#include "../su3_matrix.h"

#include "../config.h"

#include <fftw3.h>
#include <fftw3-mpi.h>

#include "../field.h"

#include "../mpi_fftw_class.h"

#include <omp.h>

#include <math.h>

#include "../mpi_class.h"

#include "../momenta.h"

#include <time.h>

#include "../rand_class.h"

#include "../MV_class.h"

#include <numeric>

//#include "single_field.h"

int main(int argc, char *argv[]) {

    printf("INITIALIZATION\n");

//initialization

    config* cnfg = new config;

    cnfg->stat = 800;

    mpi_class* mpi = new mpi_class(argc, argv);

    mpi->mpi_init(cnfg);

    mpi->mpi_exchange_grid();

    rand_class* random_generator = new rand_class(mpi,cnfg);

    fftw2D* fourier2 = new fftw2D(cnfg);

    fourier2->init2D();    

    std::complex<double> data[Nx*Ny];
    std::complex<double> sum[Nx*Ny];
    std::complex<double> err[Nx*Ny];


    for(int i = 0; i < Nx*Ny; i++){
	sum[i] = 0.0;
	err[i] = 0.0;
    }

    static __thread std::ranlux24* generator = nullptr;
    if (!generator){
    	std::hash<std::thread::id> hasher;
        generator = new std::ranlux24(clock() + hasher(std::this_thread::get_id()));
    }
    std::normal_distribution<double> distribution{0.0,1.0};

const int nn = 1000000;
const double inv = (1.0/nn);

for(int t = 0; t < nn; t++){

    for(int i = 0; i < Nx*Ny; i++)
	data[i] = std::complex<double>(distribution(*generator)/sqrt(1.0*Nx*Ny),0.0);

//  for(int i = 0; i < Nx*Ny; i++)
//rintf("before i = %i, data = %f %f\n", i, data[i].real(), data[i].imag());

    fourier2->execute2D(data, 0);

    for(int i = 0; i < Nx*Ny; i++){
	sum[i] += data[i]*inv;
	err[i] += data[i]*std::conj(data[i])*inv;
    }
}

    for(int i = 0; i < Nx*Ny; i++)
	printf("i = %i, sum = %f %f, err = %f %f\n", i, sum[i].real(), sum[i].imag(), err[i].real(), err[i].imag());

    delete cnfg;
  
    delete random_generator;

    delete fourier2;

    delete mpi;

    MPI_Finalize();


return 1;
}
 
