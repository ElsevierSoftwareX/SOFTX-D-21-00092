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
//    lfield<double> uf_next(cnfg->Nxl,cnfg->Nyl);

    for(int i = 0; i < MVmodel->Ny_parameter; i++){
	
	f.setMVModel(MVmodel, random_generator);

	fourier->execute1D(&f, 0);

	f.solvePoisson(0.00001 * pow(MVmodel->g_parameter,2.0) * MVmodel->mu_parameter, MVmodel->g_parameter, momtable);

    	fourier->execute1D(&f, 1);

	f.exponentiate();

	uf *= f;
    }

    //exchange and store uf in the global array gf
    gfield<double> gf(Nx, Ny);
//    gfield<double> gf_next(Nx, Ny);

    gf.allgather(&uf);

    //perform evolution
    lfield<double> xi_local_x(cnfg->Nxl,cnfg->Nyl);
    lfield<double> xi_local_y(cnfg->Nxl,cnfg->Nyl);

    lfield<double> kernel_pbarx(cnfg->Nxl,cnfg->Nyl);
    kernel_pbarx.setKernelPbarX(momtable);

    lfield<double> kernel_pbary(cnfg->Nxl,cnfg->Nyl);
    kernel_pbary.setKernelPbarY(momtable);

    lfield<double> A_local(cnfg->Nxl, cnfg->Nyl);
    lfield<double> B_local(cnfg->Nxl, cnfg->Nyl);

    lfield<double> uxiulocal_x(cnfg->Nxl, cnfg->Nyl);
    lfield<double> uxiulocal_y(cnfg->Nxl, cnfg->Nyl);

    lfield<double> uf_hermitian(cnfg->Nxl, cnfg->Nyl);

    double step = 0.0001;

    for(int langevin = 0; langevin < 10; langevin++){

	xi_local_x.setGaussian(random_generator);
	xi_local_y.setGaussian(random_generator);

	//should be X2K
	fourier->execute1D(&xi_local_x, 0);
	fourier->execute1D(&xi_local_y, 0);

	//construcing A
	xi_local_x = kernel_pbarx * xi_local_x;
	xi_local_y = kernel_pbary * xi_local_y;

	A_local = xi_local_x + xi_local_y;

	//should be K2X
 	fourier->execute1D(&A_local, 1);

 	fourier->execute1D(&xi_local_x, 1);

 	fourier->execute1D(&xi_local_y, 1);

	//constructng B
                   //tmpunitc%su3 = uglobal(me()*volume_half()+ind,eo)%su3

                   //tmpunitd%su3 = transpose(conjg(tmpunitc%su3))

                   //uxiulocal(ind,eo,1)%su3 = matmul(tmpunitc%su3, matmul(xi_local(ind,eo,1)%su3, tmpunitd%su3))
                   //uxiulocal(ind,eo,2)%su3 = matmul(tmpunitc%su3, matmul(xi_local(ind,eo,2)%su3, tmpunitd%su3))

    	uf_hermitian = uf.hermitian();

	uxiulocal_x = uf * xi_local_x * uf_hermitian;

	uxiulocal_y = uf * xi_local_y * uf_hermitian;

	//should be X2K
	fourier->execute1D(&uxiulocal_x, 0);
	fourier->execute1D(&uxiulocal_y, 0);

	uxiulocal_x = kernel_pbarx * uxiulocal_x;
	uxiulocal_y = kernel_pbary * uxiulocal_y;

	B_local = uxiulocal_x + uxiulocal_y;

	//should be K2X
	fourier->execute1D(&B_local, 1);

	A_local.exponentiate(sqrt(step));

	B_local.exponentiate(-sqrt(step));

	uf = B_local * uf * A_local;

	gf.allgather(&uf);
    }

    //compute correlation function
    fourier->execute1D(&uf);

    uf_hermitian = uf.hermitian();

    trace(uf, uf_hermitian);

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

