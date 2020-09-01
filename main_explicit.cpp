#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include <complex.h>

#include "su3_matrix.h"

#include "config.h"

#include <fftw3.h>
#include <fftw3-mpi.h>

#include "field.h"
#include "matrix.h"

#include "mpi_fftw_class.h"

#include <omp.h>

#include <math.h>

#include "mpi_class.h"

#include "momenta.h"
#include "positions.h"

#include "rand_class.h"

#include "MV_class.h"

#include <numeric>

//#include "single_field.h"

int main(int argc, char *argv[]) {

    printf("INITIALIZATION\n");

//initialization

    config* cnfg = new config;

    cnfg->stat = 4;

    mpi_class* mpi = new mpi_class(argc, argv);

    mpi->mpi_init(cnfg);

    mpi->mpi_exchange_grid();

    momenta* momtable = new momenta(cnfg, mpi);

    positions* postable = new positions(cnfg, mpi);

    momtable->set();

    postable->set();

    rand_class* random_generator = new rand_class(mpi,cnfg);

    MV_class* MVmodel = new MV_class(1.0, 30.72/Nx, 50);

    fftw2D* fourier2 = new fftw2D(cnfg);

    fourier2->init2D();

    printf("ALLOCATION\n");

//allocation
//-------------------------------------------------------
    //construct initial state
    lfield<double,9> f(cnfg->Nxl, cnfg->Nyl);
    lfield<double,9> uf(cnfg->Nxl, cnfg->Nyl);

//    lfield<double,9> tmp(cnfg->Nxl, cnfg->Nyl);

//-------------------------------------------------------

    //exchange and store uf in the global array gf
    //gfield<double,9> gf(Nx, Ny);
    //is this realy needed? for momentum evolution is not

    gfield<double,9> uf_global(Nx, Ny);

//-------------------------------------------------------
    //evolution
    lfield<double,9> xi_local_x(cnfg->Nxl, cnfg->Nyl);
    lfield<double,9> xi_local_y(cnfg->Nxl, cnfg->Nyl);

    lfield<double,9> xi_local_x_tmp(cnfg->Nxl, cnfg->Nyl);
    lfield<double,9> xi_local_y_tmp(cnfg->Nxl, cnfg->Nyl);



    //for evolution in position space
    gfield<double,9> xi_global_x(Nx, Ny);
    gfield<double,9> xi_global_y(Nx, Ny);

    gfield<double,9> xi_global_x_tmp(Nx, Ny);
    gfield<double,9> xi_global_y_tmp(Nx, Ny);
    gfield<double,9> xi_global_tmp(Nx, Ny);

    //initiaization of kernel fields
    lfield<double,9> kernel_pbarx(cnfg->Nxl, cnfg->Nyl);
    kernel_pbarx.setKernelPbarX(momtable);

    lfield<double,9> kernel_pbary(cnfg->Nxl, cnfg->Nyl);
    kernel_pbary.setKernelPbarY(momtable);

    //initiaization of kernel fields
    gfield<double,9> kernel_xbarx(Nx, Ny);
    gfield<double,9> kernel_xbary(Nx, Ny);

    //initiaization of kernel fields
    lfield<double,9> kernel_pbarx_with_sqrt_coupling_constant(cnfg->Nxl, cnfg->Nyl);
    kernel_pbarx_with_sqrt_coupling_constant.setKernelPbarXWithCouplingConstant(momtable);

    lfield<double,9> kernel_pbary_with_sqrt_coupling_constant(cnfg->Nxl, cnfg->Nyl);
    kernel_pbary_with_sqrt_coupling_constant.setKernelPbarYWithCouplingConstant(momtable);


    lfield<double,9> A_local(cnfg->Nxl, cnfg->Nyl);
    lfield<double,9> B_local(cnfg->Nxl, cnfg->Nyl);

    lfield<double,9> uxiulocal_x(cnfg->Nxl, cnfg->Nyl);
    lfield<double,9> uxiulocal_y(cnfg->Nxl, cnfg->Nyl);

    gfield<double,9> uxiu_global_tmp(Nx, Ny);

    lfield<double,9>* uf_hermitian;
    gfield<double,9>* uf_global_hermitian;

//-------------------------------------------------------

    //correlation function
    lfield<double,1>* corr = new lfield<double,1>(cnfg->Nxl, cnfg->Nyl);
    gfield<double,1>* corr_global = new gfield<double,1>(Nx, Ny);

//-------------------------------------------------------
//------ACCUMULATE STATISTICS----------------------------
//-------------------------------------------------------


    lfield<double,1> sum(cnfg->Nxl, cnfg->Nyl);

    sum.setToZero();

//-------------------------------------------------------
//-------------------------------------------------------

//global control variables, should be exported to the config structure and set up from the input file
	int sqrt_coupling_constant = 0;
	int noise_coupling_constant = 1;

	int momentum_evolution = 0;
	int position_evolution = 1;

//-------------------------------------------------------
//-------------------------------------------------------

	gmatrix<double>* cholesky;


//create sigma correlation matrix for the noise vectors in position space
//perform cholesky decomposition to get the square root of the correlation matrix
if(position_evolution == 1 && noise_coupling_constant == 1 ){

	corr->setCorrelationsForCouplingConstant(momtable);

	fourier2->execute2D(corr, 0);

	corr_global->allgather(corr, mpi);
		
	cholesky = new gmatrix<double>(Nx*Ny,Nx*Ny);

	cholesky->decompose(corr_global);
	printf("cholesky decomposition finished\n");
}

for(int stat = 0; stat < cnfg->stat; stat++){


	printf("Gatherting stat sample no. %i\n", stat);

	//-------------------------------------------------------
	//------INITIAL STATE------------------------------------
	//-------------------------------------------------------

	uf.setToUnit();

	printf("Constructing initial state from the MV model\n");

    	for(int i = 0; i < MVmodel->Ny_parameter; i++){
	
		f.setMVModel(MVmodel, random_generator);

		fourier2->execute2D(&f, 1);

		f.solvePoisson(0.0001 * pow(MVmodel->g_parameter,2.0) * MVmodel->mu_parameter, MVmodel->g_parameter, momtable);

	    	fourier2->execute2D(&f, 0);

		//f.exponentiate();

		uf *= f;
    	}

	printf("Initial state constructed\n");

	//-------------------------------------------------------
	//------EVOLUTION----------------------------------------
	//-------------------------------------------------------


	//perform evolution
    	double step = 0.0004;

	//evolution
    	for(int langevin = 0; langevin < 100; langevin++){

		printf("Performing evolution step no. %i\n", langevin);

		xi_local_x.setGaussian(mpi, cnfg);
		xi_local_y.setGaussian(mpi, cnfg);

		if( momentum_evolution == 1 ){

			if(noise_coupling_constant == 0){
				//should be X2K
				fourier2->execute2D(&xi_local_x, 1);
				fourier2->execute2D(&xi_local_y, 1);
			}

			if(sqrt_coupling_constant == 1 || noise_coupling_constant == 1 ){
				//construcing A
				xi_local_x_tmp = kernel_pbarx_with_sqrt_coupling_constant * xi_local_x;
				xi_local_y_tmp = kernel_pbary_with_sqrt_coupling_constant * xi_local_y;
			}else{
				xi_local_x_tmp = kernel_pbarx * xi_local_x;
				xi_local_y_tmp = kernel_pbary * xi_local_y;
			}

			A_local = xi_local_x_tmp + xi_local_y_tmp;

			//should be K2X
	 		fourier2->execute2D(&A_local, 0);
			fourier2->execute2D(&xi_local_x, 0);
 			fourier2->execute2D(&xi_local_y, 0);

			//constructng B
		    	uf_hermitian = uf.hermitian();

			uxiulocal_x = uf * xi_local_x * (*uf_hermitian);

			uxiulocal_y = uf * xi_local_y * (*uf_hermitian);

			delete uf_hermitian;

			//should be X2K
			fourier2->execute2D(&uxiulocal_x, 1);
			fourier2->execute2D(&uxiulocal_y, 1);

			if(sqrt_coupling_constant == 1 || noise_coupling_constant == 1){
				//construcing B
				uxiulocal_x = kernel_pbarx_with_sqrt_coupling_constant * uxiulocal_x;
				uxiulocal_y = kernel_pbary_with_sqrt_coupling_constant * uxiulocal_y;
			}else{
				uxiulocal_x = kernel_pbarx * uxiulocal_x;
				uxiulocal_y = kernel_pbary * uxiulocal_y;
			}

			B_local = uxiulocal_x + uxiulocal_y;

			//should be K2X
			fourier2->execute2D(&B_local, 0);
		}

		if( position_evolution == 1 ){

			//if(noise_coupling_constant == 1 ){
			//	//should be X2K
			//	fourier2->execute2D(&xi_local_x, 0);
			//	fourier2->execute2D(&xi_local_y, 0);
			//}

			printf("gathering local xi to global\n");
			xi_global_x.allgather(&xi_local_x, mpi);	
    			xi_global_y.allgather(&xi_local_y, mpi);	

			if(noise_coupling_constant == 1 ){
				//should be X2K
				xi_global_x.multiplyByCholesky(cholesky);
				xi_global_y.multiplyByCholesky(cholesky);
	
			}

			printf("gathering local uf to global\n");
    			uf_global.allgather(&uf, mpi);

			printf("starting iteration over global lattice\n");
			//for(int i = 0; i < cnfg->Nxl*cnfg->Nyl; i++){
			for(int x = 0; x < cnfg->Nxl; x++){
				for(int y = 0; y < cnfg->Nyl; y++){

					int x_global = x + mpi->getPosX()*cnfg->Nxl;
					int y_global = y + mpi->getPosY()*cnfg->Nyl;

					if( sqrt_coupling_constant == 1 ){
						kernel_xbary.setKernelXbarYWithCouplingConstant(x_global, y_global, postable);
						kernel_xbarx.setKernelXbarXWithCouplingConstant(x_global, y_global, postable);
					
						xi_global_x_tmp = kernel_xbarx * xi_global_x; 
						xi_global_y_tmp = kernel_xbary * xi_global_y; 
					}else{
						kernel_xbary.setKernelXbarY(x_global, y_global, postable);
						kernel_xbarx.setKernelXbarX(x_global, y_global, postable);
					
						xi_global_x_tmp = kernel_xbarx * xi_global_x; 
						xi_global_y_tmp = kernel_xbary * xi_global_y; 
					}

					xi_global_tmp = xi_global_x_tmp + xi_global_y_tmp;

				    	uf_global_hermitian = uf_global.hermitian();

					uxiu_global_tmp = uf_global * xi_global_tmp * (*uf_global_hermitian);

					delete uf_global_hermitian;
			
					A_local.reduceAndSet(x, y, &xi_global_tmp); 

					B_local.reduceAndSet(x, y, &uxiu_global_tmp); 

				}
			}

		}
		
		A_local.exponentiate(sqrt(step));

		B_local.exponentiate(-sqrt(step));

		uf = B_local * uf * A_local;
			
    	}

	//-------------------------------------------------------
	//------CORRELATION FUNCTION-----------------------------
	//-------------------------------------------------------


	//compute correlation function
	//should be X2K

   	fourier2->execute2D(&uf, 1);
    
	uf.trace(corr);

    	corr_global->allgather(corr, mpi);	

   	corr_global->average_and_symmetrize();

        //store stat in the accumulator
        lfield<double,1>* corr_ptr = corr_global->reduce(cnfg->Nxl, cnfg->Nyl, mpi);

        sum += *corr_ptr;

        delete corr_ptr;

    }

//    printf("accumulator size = %i\n", accumulator.size());

//    for (std::vector<lfield<double,1>*>::iterator it = accumulator.begin() ; it != accumulator.end(); ++it)
//      sum += **it;

//    sum.print(momtable, 1.0/3.0/accumulator.size(), mpi);
    sum.print(momtable, 1.0/3.0/cnfg->stat, mpi);


//-------------------------------------------------------
//------DEALLOCATE AND CLEAN UP--------------------------
//-------------------------------------------------------



    if(position_evolution == 1 && noise_coupling_constant == 1 )
	delete cholesky;

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
 
