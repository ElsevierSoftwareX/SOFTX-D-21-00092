#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include <complex.h>

#include "su3_matrix.h"

#include "config.h"

#include <fftw3.h>
#include <fftw3-mpi.h>

#include "field.h"

#include "mpi_fftw_class.h"

#include <omp.h>

#include <math.h>

#include "mpi_class.h"

#include "momenta.h"

#include "rand_class.h"

#include "MV_class.h"

int main(int argc, char *argv[]) {

    printf("STARTING CLASS PROGRAM\n");

    config* cnfg = new config;

    mpi_class* mpi = new mpi_class(argc, argv);

    mpi->mpi_init(cnfg);

    mpi->mpi_exchange_grid();

    mpi->mpi_exchange_groups();

    momenta* momtable = new momenta(cnfg, mpi);

    momtable->set();

    rand_class* random_generator = new rand_class(mpi,cnfg);

    MV_class* MVmodel = new MV_class(1.0, 0.12, 5);

    fftw1D* fourier = new fftw1D(cnfg);

    fourier->init1D(mpi->getRowComm(), mpi->getColComm());    


    //construct initial state
    lfield<double> f(cnfg->Nxl,cnfg->Nyl);
    lfield<double> uf(cnfg->Nxl,cnfg->Nyl);

    for(int i = 0; i < MVmodel->Ny_parameter; i++){
	
	f.setMVModel(MVmodel, random_generator);

	fourier->execute1D(&f, 0);

	f.solvePoisson(0.00001 * pow(MVmodel->g_parameter,2.0) * MVmodel->mu_parameter, MVmodel->g_parameter, momtable);

    	fourier->execute1D(&f, 1);

	f.exponentiate();

	uf *= f;
    }

    gfield<double> gf(Nx, Ny);

    gf.allgather(&uf);

    //perform evolution
    lfield<double> xi_local_x(cnfg->Nxl,cnfg->Nyl);
    lfield<double> xi_local_y(cnfg->Nxl,cnfg->Nyl);

    for(int langevin = 0; langevin < 10; langevin++){

	xi_local_x.setGaussian(random_generator);
	xi_local_y.setGaussian(random_generator);

	fourier->execute1D(&xi_local_x, 0);
	fourier->execute1D(&xi_local_y, 0);

    }

    delete fourier;

    delete mpi;

    MPI_Finalize();

return 1;
}

/*
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

    for(xx = 0; xx < cnfg->Nxl; xx++){
    	for(yy = 0; yy < cnfg->Nyl; yy++){

		XX[xx*cnfg->Nyl + yy] = mpi->getRank()*100+(xx*cnfg->Nyl+yy) + 1;
	}
    }

    for(xx = 0; xx < Nx; xx++){
    	for(yy = 0; yy < Ny; yy++){

		XX[yy + Ny*xx] = yy + Ny*xx + 1;
	}
    }

    printf("FIELD ALLOCATION AND BOUNDARY EXCHANGE\n");

    gfield<double> gf(Nx,Ny);

//    mpi_split(XX, p);

//    mpi_gather(YY, p);

//    f.mpi_exchange_boundaries(mpi);

    printf("FFTW TEST\n");



*/ 

