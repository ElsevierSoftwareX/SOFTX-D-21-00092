#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <ctime>

#include <complex.h>

#include "../su3_matrix.h"
#include "../matrix.h"

#include "../config.h"

#include <fftw3.h>
#include <fftw3-mpi.h>

#include "../field.h"

#include "../mpi_fftw_class.h"

#include <omp.h>

#include <math.h>

#include "../mpi_class.h"

#include "../momenta.h"
#include "../positions.h"

#include "../rand_class.h"

#include "../MV_class.h"

#include <numeric>

//#include "single_field.h"

int main(int argc, char *argv[]) {

    printf("INITIALIZATION\n");

//initialization

    config* cnfg = new config;

    cnfg->stat = 1000;

    mpi_class* mpi = new mpi_class(argc, argv);

    mpi->mpi_init(cnfg);

    mpi->mpi_exchange_grid();

    momenta* momtable = new momenta(cnfg, mpi);

    momtable->set();

    positions postable(cnfg, mpi);

    postable.set();

    rand_class* random_generator = new rand_class(mpi,cnfg);

    MV_class* MVmodel = new MV_class(1.0, 30.72/Nx, 50);

    //fftw1D* fourier = new fftw1D(cnfg);

    fftw2D* fourier2 = new fftw2D(cnfg);

    //fourier->init1D(mpi->getRowComm(), mpi->getColComm());    

    fourier2->init2D();    

    printf("ALLOCATION\n");

//allocation
//-------------------------------------------------------
    //construct initial state
    lfield<double,9> f(cnfg->Nxl, cnfg->Nyl);
    lfield<double,9> uf(cnfg->Nxl, cnfg->Nyl);

    gfield<double,9> uf_global(Nx, Ny);

    //evolution
    lfield<double,9> xi_local_x(cnfg->Nxl, cnfg->Nyl);
    lfield<double,9> xi_local_y(cnfg->Nxl, cnfg->Nyl);

    gfield<double,9> xi_global_x(Nx, Ny);
    gfield<double,9> xi_global_y(Nx, Ny);

    lfield<double,9> A_local(cnfg->Nxl, cnfg->Nyl);
    lfield<double,9> B_local(cnfg->Nxl, cnfg->Nyl);

//-------------------------------------------------------

    //correlation function
    lfield<double,1>* corr = new lfield<double,1>(cnfg->Nxl, cnfg->Nyl);
    gfield<double,1>* corr_global = new gfield<double,1>(Nx, Ny);

//-------------------------------------------------------
//------ACCUMULATE STATISTICS----------------------------
//-------------------------------------------------------

    //std::vector<lfield<double,1>*> accumulator;

//    lfield<double,1> zero(cnfg->Nxl, cnfg->Nyl);
//    zero.setToZero();

    lfield<double,81> zero(cnfg->Nxl, cnfg->Nyl);
//    zero.setToZero();

//    std::vector<lfield<double,81>> sum(Nx, zero);
//    std::vector<lfield<double,81>> err(Nx, zero);

    lfield<double,81>* sum[Nx*Ny];
    lfield<double,81>* err[Nx*Ny];

    for(int i = 0; i < Nx*Ny; i++){

        sum[i] = new lfield<double,81>(Nx,Ny);
        err[i] = new lfield<double,81>(Nx,Ny);

    }


//    int langevin_steps = 50;
//    int measurements = 50;

//    std::vector<lfield<double,1>> sum(measurements, zero);
//    std::vector<lfield<double,1>> err(measurements, zero);

//-------------------------------------------------------
//-------------------------------------------------------

        gmatrix<double>* cholesky;


//create sigma correlation matrix for the noise vectors in position space
//perform cholesky decomposition to get the square root of the correlation matrix

        int position_space = 0;
        int momentum_space = 1;

        if( position_space == 1 ){

                corr_global->setCorrelationsForCouplingConstant();

        }
        if( momentum_space == 1 ){

                corr->setCorrelationsForCouplingConstant(momtable);
	
		//constructed in P space, transformed to X space
                fourier2->execute2D(corr, 0);

                corr_global->allgather(corr, mpi);

        }
        //each rank is constructing his own global cholesky matrix

//      gmatrix<double,1> sigma(Nx*Ny,Nx*Ny);
        cholesky = new gmatrix<double>(Nx*Ny,Nx*Ny);

	corr_global->printDebug();

//	exit(0);

        //cholesky decomposition of the implicit matrix sigma(x,y) = corr_global(x-y) 

        cholesky->decompose(corr_global);
        printf("cholesky decomposition finished\n");

//	cholesky->print();


for(int stat = 0; stat < cnfg->stat; stat++){

	generate_gaussian(&xi_local_x, &xi_local_y, mpi, cnfg);

//        printf("gathering local xi to global\n");
        xi_global_x.allgather(&xi_local_x, mpi);
//        xi_global_y.allgather(&xi_local_y, mpi);

//	xi_global_x.printDebug();

        xi_global_x.multiplyByCholesky(cholesky);
//        xi_global_y.multiplyByCholesky(cholesky);

//	xi_global_x.printDebug();

        for(int i = 0; i < Nx*Ny; i++){

                for(int ii = 0; ii < Nx*Ny; ii++){

                        for(int t = 0; t < 9; t++){

                                for(int tt = 0; tt < 9; tt++){

                                sum[i]->u[ii*81+t*9+tt] += (1.0/cnfg->stat) * xi_global_x.u[i*9+t] * std::conj(xi_global_x.u[ii*9+tt]);
                                //err[i]->u[ii*81+t*9+tt] += (xi_local_x.u[i*9+t] * std::conj(xi_local_y.u[ii*9+tt])) * (xi_local_x.u[i*9+t] * std::conj(xi_local_y.u[ii*9+tt]));

                                }
                        }
                }
        }
}

for(int i = 0; i < Nx*Ny; i++){
        sum[i]->printDebug(i); //, &err[i], momtable, 1.0/3.0/cnfg->stat, mpi);
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

    delete corr;

    delete corr_global;

    MPI_Finalize();


return 1;
}
 
