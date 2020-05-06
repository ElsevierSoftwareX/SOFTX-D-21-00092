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

//#include "single_field.h"

int main(int argc, char *argv[]) {

    printf("INITIALIZATION\n");

//initialization

    config* cnfg = new config;

	cnfg->stat = 10;

    mpi_class* mpi = new mpi_class(argc, argv);

    mpi->mpi_init(cnfg);

    mpi->mpi_exchange_grid();

    mpi->mpi_exchange_groups();

    momenta* momtable = new momenta(cnfg, mpi);

    momtable->set();

    rand_class* random_generator = new rand_class(mpi,cnfg);

    MV_class* MVmodel = new MV_class(1.0, 0.48, 5);

    fftw1D* fourier = new fftw1D(cnfg);

    fourier->init1D(mpi->getRowComm(), mpi->getColComm());    

    printf("ALLOCATION\n");

//allocation
//-------------------------------------------------------
    //construct initial state
    lfield<double,9> f(cnfg->Nxl, cnfg->Nyl);
    lfield<double,9> uf(cnfg->Nxl, cnfg->Nyl);
//-------------------------------------------------------

    //exchange and store uf in the global array gf
    //gfield<double,9> gf(Nx, Ny);
    //is this realy needed? for momentum evolution is not

//-------------------------------------------------------
    //evolution
    lfield<double,9> xi_local_x(cnfg->Nxl, cnfg->Nyl);
    lfield<double,9> xi_local_y(cnfg->Nxl, cnfg->Nyl);

    //initiaization of kernel fields
    lfield<double,9> kernel_pbarx(cnfg->Nxl, cnfg->Nyl);
    kernel_pbarx.setKernelPbarX(momtable);

    lfield<double,9> kernel_pbary(cnfg->Nxl, cnfg->Nyl);
    kernel_pbary.setKernelPbarY(momtable);

    lfield<double,9> A_local(cnfg->Nxl, cnfg->Nyl);
    lfield<double,9> B_local(cnfg->Nxl, cnfg->Nyl);

    lfield<double,9> uxiulocal_x(cnfg->Nxl, cnfg->Nyl);
    lfield<double,9> uxiulocal_y(cnfg->Nxl, cnfg->Nyl);

    lfield<double,9> uf_hermitian(cnfg->Nxl, cnfg->Nyl);

//-------------------------------------------------------
    //correlation function
    lfield<double,1> corr(cnfg->Nxl, cnfg->Nyl);
    gfield<double,1> corr_global(Nx, Ny);

//-------------------------------------------------------
//------ACCUMULATE STATISTICS----------------------------
//-------------------------------------------------------

    std::vector<lfield<double,1>> accumulator;

for(int stat = 0; stat < cnfg->stat; stat++){

	printf("Gatherting stat sample no. %i\n", stat);

	//-------------------------------------------------------
	//------INITIAL STATE------------------------------------
	//-------------------------------------------------------

	printf("Constructing initial state from the MV model\n");

    	for(int i = 0; i < MVmodel->Ny_parameter; i++){
	
		f.setMVModel(MVmodel, random_generator);

		fourier->execute1D(&f, 0);

		f.solvePoisson(0.00001 * pow(MVmodel->g_parameter,2.0) * MVmodel->mu_parameter, MVmodel->g_parameter, momtable);

	    	fourier->execute1D(&f, 1);

		f.exponentiate();

		uf *= f;
    	}

	printf("Initial state constructed\n");

    	//gf.allgather(&uf);

	//-------------------------------------------------------
	//------EVOLUTION----------------------------------------
	//-------------------------------------------------------


	//perform evolution
    	double step = 0.0001;

    	for(int langevin = 0; langevin < 10; langevin++){

		printf("Performing evolution step no. %i\n", langevin);

		xi_local_x.setGaussian(random_generator);
		xi_local_y.setGaussian(random_generator);

		printf("A\n");

		//should be X2K
		fourier->execute1D(&xi_local_x, 0);
		fourier->execute1D(&xi_local_y, 0);

		printf("B\n");

		//construcing A
		xi_local_x = kernel_pbarx * xi_local_x;
		xi_local_y = kernel_pbary * xi_local_y;

		printf("C\n");

		A_local = xi_local_x + xi_local_y;

		printf("D\n");

		//should be K2X
 		fourier->execute1D(&A_local, 1);

		printf("D.a\n");

 		fourier->execute1D(&xi_local_x, 1);

		printf("D.b\n");

 		fourier->execute1D(&xi_local_y, 1);

		printf("D.c\n");

		printf("E\n");

		//constructng B
        	           //tmpunitc%su3 = uglobal(me()*volume_half()+ind,eo)%su3

                	   //tmpunitd%su3 = transpose(conjg(tmpunitc%su3))

	                   //uxiulocal(ind,eo,1)%su3 = matmul(tmpunitc%su3, matmul(xi_local(ind,eo,1)%su3, tmpunitd%su3))
        	           //uxiulocal(ind,eo,2)%su3 = matmul(tmpunitc%su3, matmul(xi_local(ind,eo,2)%su3, tmpunitd%su3))

	    	uf_hermitian = uf.hermitian();

		printf("F\n");

		uxiulocal_x = uf * xi_local_x * uf_hermitian;

		uxiulocal_y = uf * xi_local_y * uf_hermitian;

		printf("G\n");

		//should be X2K
		fourier->execute1D(&uxiulocal_x, 0);
		fourier->execute1D(&uxiulocal_y, 0);

		printf("H\n");

		uxiulocal_x = kernel_pbarx * uxiulocal_x;
		uxiulocal_y = kernel_pbary * uxiulocal_y;

		printf("I\n");

		B_local = uxiulocal_x + uxiulocal_y;

		printf("J\n");

		//should be K2X
		fourier->execute1D(&B_local, 1);

		printf("K\n");

		A_local.exponentiate(sqrt(step));

		B_local.exponentiate(-sqrt(step));

		printf("L\n");

		uf = B_local * uf * A_local;

		printf("M\n");

		//gf.allgather(&uf);
    	}

	//-------------------------------------------------------
	//------CORRELATION FUNCTION-----------------------------
	//-------------------------------------------------------

	//compute correlation function
	//should be X2K
    	fourier->execute1D(&uf, 0);

    	uf.trace(&corr);

    	corr_global.allgather(&corr);	

    	corr_global.average_and_symmetrize();

	//store stat in the accumulator

	accumulator.push_back(corr_global.reduce(cnfg->Nxl, cnfg->Nyl, mpi));

    }

    printf("accumulator size = %i\n", accumulator.size());

    delete fourier;

    delete mpi;

    MPI_Finalize();

return 1;
}
 
