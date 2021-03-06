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

#include "rand_class.h"

#include "MV_class.h"

#include <numeric>

//#include "single_field.h"

int main(int argc, char *argv[]) {

    printf("INITIALIZATION\n");

//initialization

    config* cnfg = new config;

    cnfg->stat = 2;

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

    std::vector<lfield<double,1>*> accumulator;

//-------------------------------------------------------
//-------------------------------------------------------

//global control variables, should be exported to the config structure and set up from the input file
	int sqrt_coupling_constant = 0;
	int noise_coupling_constant = 0;

	int momentum_evolution = 1;
	int position_evolution = 0;

//-------------------------------------------------------
//-------------------------------------------------------

	gmatrix<double>* cholesky;


//create sigma correlation matrix for the noise vectors in position space
//perform cholesky decomposition to get the square root of the correlation matrix
if(position_evolution == 1 && noise_coupling_constant == 1 ){

	int position_space = 0;
	int momentum_space = 1;

	if( position_space == 1 ){
	
		corr_global->setCorrelationsForCouplingConstant();

	}
	if( momentum_space == 1 ){

		corr->setCorrelationsForCouplingConstant(momtable);

		fourier->execute1D(corr, 0);

		corr_global->allgather(corr);
		
	}
	//each rank is constructing his own global cholesky matrix

//	gmatrix<double,1> sigma(Nx*Ny,Nx*Ny);
	cholesky = new gmatrix<double>(Nx*Ny,Nx*Ny);

	//cholesky decomposition of the implicit matrix sigma(x,y) = corr_global(x-y) 

	cholesky->decompose(corr_global);
	printf("cholesky decomposition finished\n");
}

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

//	return 0;

	printf("Initial state constructed\n");

	//-------------------------------------------------------
	//------EVOLUTION----------------------------------------
	//-------------------------------------------------------


	//perform evolution
    	double step = 0.0001;

	//evolution
    	for(int langevin = 0; langevin < 2; langevin++){

		printf("Performing evolution step no. %i\n", langevin);

		xi_local_x.setGaussian(random_generator);
		xi_local_y.setGaussian(random_generator);

		if( momentum_evolution == 1 ){

			if(noise_coupling_constant == 0){
				//should be X2K
				fourier->execute1D(&xi_local_x, 0);
				fourier->execute1D(&xi_local_y, 0);
			}

			if(sqrt_coupling_constant == 1 || noise_coupling_constant == 1){
				//construcing A
				xi_local_x = kernel_pbarx_with_sqrt_coupling_constant * xi_local_x;
				xi_local_y = kernel_pbary_with_sqrt_coupling_constant * xi_local_y;
			}else{
				xi_local_x = kernel_pbarx * xi_local_x;
				xi_local_y = kernel_pbary * xi_local_y;
			}
	
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

			uxiulocal_x = uf * xi_local_x * (*uf_hermitian);

			uxiulocal_y = uf * xi_local_y * (*uf_hermitian);

			delete uf_hermitian;


			//should be X2K
			fourier->execute1D(&uxiulocal_x, 0);
			fourier->execute1D(&uxiulocal_y, 0);

			uxiulocal_x = kernel_pbarx * uxiulocal_x;
			uxiulocal_y = kernel_pbary * uxiulocal_y;

			B_local = uxiulocal_x + uxiulocal_y;

			//should be K2X
			fourier->execute1D(&B_local, 1);
		}

		if( position_evolution == 1 ){

			if(noise_coupling_constant == 1 ){
				//should be X2K
				fourier->execute1D(&xi_local_x, 0);
				fourier->execute1D(&xi_local_y, 0);
			}

			printf("gathering local xi to global\n");
			xi_global_x.allgather(&xi_local_x);	
    			xi_global_y.allgather(&xi_local_y);	

			if(noise_coupling_constant == 1 ){
				//should be X2K
				xi_global_x.multiplyByCholesky(cholesky);
				xi_global_y.multiplyByCholesky(cholesky);
	
			}

			printf("gathering local uf to global\n");
    			uf_global.allgather(&uf);

			printf("starting iteration over global lattice\n");
			//for(int i = 0; i < cnfg->Nxl*cnfg->Nyl; i++){
			for(int x = 0; x < cnfg->Nxl; x++){
				for(int y = 0; y < cnfg->Nyl; y++){

					int x_global = x + mpi->getPosX()*cnfg->Nxl;
					int y_global = y + mpi->getPosY()*cnfg->Nyl;

					if( sqrt_coupling_constant == 1 ){
						kernel_xbary.setKernelXbarYWithCouplingConstant(x_global, y_global);
						kernel_xbarx.setKernelXbarXWithCouplingConstant(x_global, y_global);
					
						xi_global_x_tmp = kernel_xbarx * xi_global_x; 
						xi_global_y_tmp = kernel_xbary * xi_global_y; 
					}else{
						kernel_xbary.setKernelXbarY(x_global, y_global);
						kernel_xbarx.setKernelXbarX(x_global, y_global);
					
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

   	fourier->execute1D(&uf, 0);
    
	uf.trace(corr);

    	corr_global->allgather(corr);	

   	corr_global->average_and_symmetrize();

	//store stat in the accumulator
	lfield<double,1>* corr_ptr = corr_global->reduce(cnfg->Nxl, cnfg->Nyl, mpi);

	//accumulator.push_back(corr_global->reduce(cnfg->Nxl, cnfg->Nyl, mpi));
	accumulator.push_back(corr_ptr);

    }

    printf("accumulator size = %i\n", accumulator.size());

    lfield<double,1> sum(cnfg->Nxl, cnfg->Nyl);

    sum.setToZero();

    for (std::vector<lfield<double,1>*>::iterator it = accumulator.begin() ; it != accumulator.end(); ++it)
	sum += **it;

    sum.print(momtable);


//-------------------------------------------------------
//------DEALLOCATE AND CLEAN UP--------------------------
//-------------------------------------------------------



    if(position_evolution == 1 && noise_coupling_constant == 1 )
	delete cholesky;

    delete cnfg;

    delete momtable;

    delete random_generator;

    delete MVmodel;

    delete fourier;

    delete mpi;

    delete corr;

    delete corr_global;

    MPI_Finalize();


return 1;
}
 
