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

    cnfg->stat = 2500;

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

//allocation
//-------------------------------------------------------
    //evolution
    lfield<double,9> xi_local_x(cnfg->Nxl, cnfg->Nyl);
    lfield<double,9> xi_local_y(cnfg->Nxl, cnfg->Nyl);

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


//    lfield<double,9> kernel_pbarx(cnfg->Nxl, cnfg->Nyl);
//    kernel_pbarx.setToZero();
//    kernel_pbarx.setKernelPbarX(momtable);

//    lfield<double,9> kernel_pbary(cnfg->Nxl, cnfg->Nyl);
//    kernel_pbary.setToZero();
//    kernel_pbary.setKernelPbarY(momtable);

    lfield<double,9> A_local(cnfg->Nxl, cnfg->Nyl);


for(int stat = 0; stat < cnfg->stat; stat++){

	generate_gaussian(&xi_local_x, &xi_local_y, mpi, cnfg);
	//generate_gaussian_with_noise_coupling_constant(&xi_local_x, &xi_local_y, momtable, mpi, cnfg);

        fourier2->execute2D(&xi_local_x, 1);
     	fourier2->execute2D(&xi_local_y, 1);

        prepare_A_local(&A_local, &xi_local_x, &xi_local_y, momtable);

	for(int i = 0; i < Nx*Ny; i++){

		for(int ii = 0; ii < Nx*Ny; ii++){

			for(int t = 0; t < 9; t++){

				for(int tt = 0; tt < 9; tt++){

				sum[i]->u[ii*81+t*9+tt] += (1.0/Nx/Ny)*A_local.u[i*9+t] * std::conj(A_local.u[ii*9+tt]);
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

    MPI_Finalize();


return 1;
}
 
