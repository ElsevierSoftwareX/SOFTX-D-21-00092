#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include <complex.h>

#include "su3_complex.h"
#include "su3_matrix.h"

#include "config.h"
#include "mpi_pos.h"
#include "mpi_split.h"
#include "mpi_gather.h"
#include "utils.h"

#include <fftw3.h>
#include <fftw3-mpi.h>

#include "field.h"

#include "mpi_fftw_class.h"

//#include "zheevh3.h"

#include <omp.h>

#include <math.h>

#include "mpi_class.h"

int main(int argc, char *argv[]) {


    printf("STARTING CLASS PROGRAM\n");

    mpi_class* mpi = new mpi_class(argc, argv);

    mpi->mpi_init();

    mpi->mpi_exchange_grid();

    mpi->mpi_exchange_groups();

    printf("TESTING PROGRAM\n");

    double XX[Nx*Ny];
    double YY[Nx*Ny];

    int ii;
    for(ii = 0; ii < Nx*Ny; ii++){
	XX[ii] = 0;
    }
    for(ii = 0; ii < Nx*Ny; ii++){
	YY[ii] = 0;
    }

    int xx, yy;

    for(xx = 0; xx < Nxl; xx++){
    	for(yy = 0; yy < Nyl; yy++){

		XX[buf_pos(xx, yy)] = mpi->getRank()*100+loc_pos(xx, yy) + 1;
	}
    }

    for(xx = 0; xx < Nx; xx++){
    	for(yy = 0; yy < Ny; yy++){

		XX[yy + Ny*xx] = yy + Ny*xx + 1;
	}
    }


//    mpi_split(XX, p);

//    mpi_gather(YY, p);

    printf("FIELD ALLOCATION AND BOUNDARY EXCHANGE\n");

    gfield<double> gf(Nx,Ny);

    lfield<double> f(Nxl,Nyl);

    f.mpi_exchange_boundaries(mpi);


    printf("FFTW TEST\n");

    fftw1D* fourier = new fftw1D();

    fourier->init1D(mpi->getRowComm(), mpi->getColComm());    

    fourier->execute1D(&f);

    printf("FINALIZE\n");

    MPI_Barrier(MPI_COMM_WORLD);

su3_matrix<double> AA;

double w[3];


AA.m[0] = 2.0;
AA.m[4] = 3.0;
AA.m[8] = 4.0;

AA.zheevh3(&w[0]);

printf("eigenvalues: %e %e %e\n", w[0], w[1], w[2]);


//    delete &f;

//    delete &gf;

    delete fourier;

    delete mpi;

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();

return 1;
}

