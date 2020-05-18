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

#include "../rand_class.h"

#include "../MV_class.h"

#include <numeric>

//#include "single_field.h"

int main(int argc, char *argv[]) {

    printf("INITIALIZATION\n");

//initialization

    config* cnfg = new config;

    cnfg->stat = 64;

    mpi_class* mpi = new mpi_class(argc, argv);

    mpi->mpi_init(cnfg);

    mpi->mpi_exchange_grid();

    mpi->mpi_exchange_groups();

    momenta* momtable = new momenta(cnfg, mpi);

    momtable->set();

    rand_class* random_generator = new rand_class(mpi,cnfg);

    MV_class* MVmodel = new MV_class(1.0, 0.24, 50);

    fftw1D* fourier = new fftw1D(cnfg);

    fftw2D* fourier2 = new fftw2D(cnfg);

    fourier->init1D(mpi->getRowComm(), mpi->getColComm());    

    fourier2->init2D();    

    printf("ALLOCATION\n");

//allocation
//-------------------------------------------------------
    //construct initial state
    lfield<double,9> f(cnfg->Nxl, cnfg->Nyl);
    lfield<double,9> uf(cnfg->Nxl, cnfg->Nyl);

    //evolution
    lfield<double,9> xi_local_x(cnfg->Nxl, cnfg->Nyl);
    lfield<double,9> xi_local_y(cnfg->Nxl, cnfg->Nyl);


    //initiaization of kernel fields
    lfield<double,9> kernel_pbarx(cnfg->Nxl, cnfg->Nyl);
    kernel_pbarx.setToZero();
    kernel_pbarx.setKernelPbarX(momtable);

    lfield<double,9> kernel_pbary(cnfg->Nxl, cnfg->Nyl);
    kernel_pbary.setToZero();
    kernel_pbary.setKernelPbarY(momtable);

    lfield<double,9> A_local(cnfg->Nxl, cnfg->Nyl);
    lfield<double,9> B_local(cnfg->Nxl, cnfg->Nyl);

    lfield<double,9> uxiulocal_x(cnfg->Nxl, cnfg->Nyl);
    lfield<double,9> uxiulocal_y(cnfg->Nxl, cnfg->Nyl);

    lfield<double,9>* uf_hermitian;


//    lfield<double,9> uf_tmp(cnfg->Nxl, cnfg->Nyl);
    lfield<double,9> xi_local_x_tmp(cnfg->Nxl, cnfg->Nyl);
    lfield<double,9> xi_local_y_tmp(cnfg->Nxl, cnfg->Nyl);
//    lfield<double,9> uxiulocal_x_tmp(cnfg->Nxl, cnfg->Nyl);
//    lfield<double,9> uxiulocal_y_tmp(cnfg->Nxl, cnfg->Nyl);



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

for(int stat = 0; stat < cnfg->stat; stat++){

	const clock_t begin_time_stat = std::clock();

	printf("Gatherting stat sample no. %i\n", stat);

	//-------------------------------------------------------
	//------INITIAL STATE------------------------------------
	//-------------------------------------------------------

	uf.setToUnit();

    	for(int i = 0; i < MVmodel->Ny_parameter; i++){
	
		f.setToZero();

		f.setMVModel(MVmodel, random_generator);

//		f.print(momtable);
//		printf("Fourier transform\n");
//		fourier->execute1D(&f, 0);
		fourier2->execute2D(&f,1);
//		f.print(momtable);

//		printf("solvePoisson\n");
		f.solvePoisson(0.0001 * pow(MVmodel->g_parameter,2.0) * MVmodel->mu_parameter, MVmodel->g_parameter, momtable);
//		f.print(momtable);

//		printf("Fourier transform\n");
//	    	fourier->execute1D(&f, 1);
		fourier2->execute2D(&f,0);
//		f.print(momtable);

		//printf("exponential\n");
		f.exponentiate();
		//f.print(momtable);

		//f.print(momtable);

		uf *= f;
		//uf = f;
    	}

        double step = 0.00005;

        //evolution
        for(int langevin = 0; langevin < 800; langevin++){

//		const clock_t begin_time = std::clock();

//                printf("Performing evolution step no. %i\n", langevin);

		xi_local_x.setToZero();
		xi_local_y.setToZero();

                xi_local_x.setGaussian(random_generator,1);
                xi_local_y.setGaussian(random_generator,2);

//		printf("xi_local_x\n");
//		xi_local_x.print(momtable);
//		printf("xi_local_y\n");
//		xi_local_y.print(momtable);

                //should be X2K
                fourier2->execute2D(&xi_local_x, 1);
                fourier2->execute2D(&xi_local_y, 1);

//		printf("after xi_local_x\n");
//		xi_local_x.print(momtable);
//		printf("after xi_local_y\n");
//		xi_local_y.print(momtable);

//		xi_local_x_tmp.setToZero();
//		xi_local_y_tmp.setToZero();
 
                xi_local_x_tmp = kernel_pbarx * xi_local_x;
                xi_local_y_tmp = kernel_pbary * xi_local_y;

//		A_local.setToZero();

                A_local = xi_local_x_tmp + xi_local_y_tmp;

//		printf("A before\n");
//		A_local.print(momtable);

                //should be K2X
                fourier2->execute2D(&A_local, 0);
                fourier2->execute2D(&xi_local_x, 0);
                fourier2->execute2D(&xi_local_y, 0);

//		printf("A after\n");
//		A_local.print(momtable);


                        //constructng B
                                   //tmpunitc%su3 = uglobal(me()*volume_half()+ind,eo)%su3

                                   //tmpunitd%su3 = transpose(conjg(tmpunitc%su3))

                                   //uxiulocal(ind,eo,1)%su3 = matmul(tmpunitc%su3, matmul(xi_local(ind,eo,1)%su3, tmpunitd%su3))
                                   //uxiulocal(ind,eo,2)%su3 = matmul(tmpunitc%su3, matmul(xi_local(ind,eo,2)%su3, tmpunitd%su3))


                uf_hermitian = uf.hermitian();

//		uxiulocal_x.setToZero();
//		uxiulocal_y.setToZero();

                uxiulocal_x = uf * xi_local_x * (*uf_hermitian);

		uxiulocal_y = uf * xi_local_y * (*uf_hermitian);

                delete uf_hermitian;

//		printf("uxiulocal_x before\n");
//		uxiulocal_x.print(momtable);
//		printf("uxiulocal_y before\n");
//		uxiulocal_y.print(momtable);

                //should be X2K
                fourier2->execute2D(&uxiulocal_x, 1);
                fourier2->execute2D(&uxiulocal_y, 1);

//		printf("uxiulocal_x after\n");
//		uxiulocal_x.print(momtable);
//		printf("uxiulocal_y after\n");
//		uxiulocal_y.print(momtable);


//		uxiulocal_x_tmp.setToZero();
//		uxiulocal_y_tmp.setToZero();

                uxiulocal_x = kernel_pbarx * uxiulocal_x;
                uxiulocal_y = kernel_pbary * uxiulocal_y;

//		B_local.setToZero();

                B_local = uxiulocal_x + uxiulocal_y;

//		printf("B before\n");
//		B_local.print(momtable);

                //should be K2X
                fourier2->execute2D(&B_local, 0);

//		printf("B after\n");
//		B_local.print(momtable);
		        
 		A_local.exponentiate(sqrt(step));

        	B_local.exponentiate(-sqrt(step));

        	uf = B_local * uf * A_local;
		
//		uf = uf_tmp;

//		std::cout << float( std::clock () - begin_time ) /  CLOCKS_PER_SEC << std::endl;
	}

    	//-------------------------------------------------------
	//------CORRELATION FUNCTION-----------------------------
	//-------------------------------------------------------

	//compute correlation function
	//should be X2K
//   	fourier->execute1D(&uf, 0);
	fourier2->execute2D(&uf,1);
    
	uf.trace(corr);

    	corr_global->allgather(corr);	

   	corr_global->average_and_symmetrize();

	//store stat in the accumulator
	lfield<double,1>* corr_ptr = corr_global->reduce(cnfg->Nxl, cnfg->Nyl, mpi);

	//accumulator.push_back(corr_global->reduce(cnfg->Nxl, cnfg->Nyl, mpi));
	accumulator.push_back(corr_ptr);

	std::cout << "ONE STAT TIME: " << float( std::clock () - begin_time_stat ) /  CLOCKS_PER_SEC << std::endl;

    }

    printf("accumulator size = %i\n", accumulator.size());

    lfield<double,1> sum(cnfg->Nxl, cnfg->Nyl);

    sum.setToZero();

    for (std::vector<lfield<double,1>*>::iterator it = accumulator.begin() ; it != accumulator.end(); ++it)
	sum += **it;

    sum.print(momtable, 1.0/3.0/accumulator.size());


//-------------------------------------------------------
//------DEALLOCATE AND CLEAN UP--------------------------
//-------------------------------------------------------

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
 
