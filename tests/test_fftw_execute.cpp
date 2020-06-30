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

    for(int i = 0; i < Nx*Ny; i++)
	data[i] = std::complex<double>(0.0,0.0);

    data[0] = std::complex<double>(1.0,0.0);

    for(int i = 0; i < Nx*Ny; i++)
	printf("before i = %i, data = %f %f\n", i, data[i].real(), data[i].imag());

    fourier2->execute2D(data, 0);

    for(int i = 0; i < Nx*Ny; i++)
	printf("after i = %i, data = %f %f\n", i, data[i].real(), data[i].imag());

    delete cnfg;
  
    delete random_generator;

    delete fourier2;

    delete mpi;

    MPI_Finalize();


return 1;
}
 
