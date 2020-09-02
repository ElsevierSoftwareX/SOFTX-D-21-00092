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

    cnfg->stat = 80000;

    mpi_class* mpi = new mpi_class(argc, argv);

    mpi->mpi_init(cnfg);

    mpi->mpi_exchange_grid();

    printf("MOMTABLE\n");

    momenta* momtable = new momenta(cnfg, mpi);

    printf("MOMTABLE->SET\n");

    momtable->set();

    printf("RAND_CLASS\n");

    rand_class* random_generator = new rand_class(mpi,cnfg);

    printf("MVModel\n");

    MV_class* MVmodel = new MV_class(1.0, 30.72/Nx, 50);

//    fftw1D* fourier = new fftw1D(cnfg);

    printf("FFTW2\n");

    fftw2D* fourier2 = new fftw2D(cnfg);

//    fourier->init1D(mpi->getRowComm(), mpi->getColComm());    

    printf("FFTW INIT\n");

    fourier2->init2D();    

    printf("ALLOCATION\n");

    const double tmp = pow(15.0*15.0/6.0/6.0,1.0/0.2);
    const double tmp2 = 4.0*M_PI/ (11.0-2.0*3.0/3.0);

    for(int x = 0; x < Nx; x++){
	for(int y = 0; y < Ny; y++){

		int i = x*Ny+y;

		printf("%i %i %f\n", x, y, sqrt( tmp2 / log( pow( tmp + pow((momtable->phat2(i)*Nx*Ny)/6.0/6.0,1.0/0.2) , 0.2) ) ));

	}

    }

//-------------------------------------------------------
//------DEALLOCATE AND CLEAN UP--------------------------
//-------------------------------------------------------

    delete cnfg;

    delete momtable;

    delete random_generator;

    delete MVmodel;

    delete fourier2;

    delete mpi;

    MPI_Finalize();


return 1;
}
 
