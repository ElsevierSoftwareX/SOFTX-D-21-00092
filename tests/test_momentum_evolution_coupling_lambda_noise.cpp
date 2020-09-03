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

    cnfg->stat = 4;

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
    //construct initial state
    lfield<double,9> f(cnfg->Nxl, cnfg->Nyl);
    lfield<double,9> uf(cnfg->Nxl, cnfg->Nyl);

    //evolution
    lfield<double,9> xi_local_x(cnfg->Nxl, cnfg->Nyl);
    lfield<double,9> xi_local_y(cnfg->Nxl, cnfg->Nyl);


    //initiaization of kernel fields
    lfield<double,9> kernel_pbarx(cnfg->Nxl, cnfg->Nyl);
    kernel_pbarx.setToZero();
//    kernel_pbarx.setKernelPbarXWithCouplingConstant(momtable); //setKernelPbarX(momtable);
    kernel_pbarx.setKernelPbarX(momtable);

    lfield<double,9> kernel_pbary(cnfg->Nxl, cnfg->Nyl);
    kernel_pbary.setToZero();
//    kernel_pbary.setKernelPbarYWithCouplingConstant(momtable); //setKernelPbarY(momtable);
    kernel_pbary.setKernelPbarY(momtable);

    lfield<double,9> A_local(cnfg->Nxl, cnfg->Nyl);
    lfield<double,9> B_local(cnfg->Nxl, cnfg->Nyl);

    lfield<double,9> uxiulocal_x(cnfg->Nxl, cnfg->Nyl);
    lfield<double,9> uxiulocal_y(cnfg->Nxl, cnfg->Nyl);

//    lfield<double,9>* uf_hermitian;


//    lfield<double,9> uf_tmp(cnfg->Nxl, cnfg->Nyl);
//    lfield<double,9> xi_local_x_tmp(cnfg->Nxl, cnfg->Nyl);
//    lfield<double,9> xi_local_y_tmp(cnfg->Nxl, cnfg->Nyl);
//    lfield<double,9> uxiulocal_x_tmp(cnfg->Nxl, cnfg->Nyl);
//    lfield<double,9> uxiulocal_y_tmp(cnfg->Nxl, cnfg->Nyl);



//-------------------------------------------------------

    //correlation function
    lfield<double,1>* corr = new lfield<double,1>(cnfg->Nxl, cnfg->Nyl);
    gfield<double,1>* corr_global = new gfield<double,1>(Nx, Ny);

//-------------------------------------------------------
//------ACCUMULATE STATISTICS----------------------------
//-------------------------------------------------------

//    std::vector<lfield<double,1>*> accumulator;

//-------------------------------------------------------
//-------------------------------------------------------

    lfield<double,1> zero(cnfg->Nxl, cnfg->Nyl);
//    zero.setToZero();

    int langevin_steps = 100;

    std::vector<lfield<double,1>> sum(langevin_steps, zero);
    std::vector<lfield<double,1>> err(langevin_steps, zero);


    //lfield<double,9> uf_copy(cnfg->Nxl, cnfg->Nyl);

    lfield<double,9> uf_copy(cnfg->Nxl, cnfg->Nyl);


for(int stat = 0; stat < cnfg->stat; stat++){

//	const clock_t begin_time_stat = std::clock();
	struct timespec start, finish;
	double elapsed;

	clock_gettime(CLOCK_MONOTONIC, &start);	

	printf("Gatherting stat sample no. %i\n", stat);

	//-------------------------------------------------------
	//------INITIAL STATE------------------------------------
	//-------------------------------------------------------

	struct timespec starti, finishi;
	double elapsedi;

	clock_gettime(CLOCK_MONOTONIC, &starti);		

	uf.setToUnit();

    	for(int i = 0; i < MVmodel->Ny_parameter; i++){
	
		//f.setToZero();

		f.setMVModel(MVmodel);

		fourier2->execute2D(&f,1);

		f.solvePoisson(0.0001 * pow(MVmodel->g_parameter,2.0) * MVmodel->mu_parameter, MVmodel->g_parameter, momtable);

		fourier2->execute2D(&f,0);

		//f.exponentiate();

		uf *= f;
    	}

	clock_gettime(CLOCK_MONOTONIC, &finishi);
		
	elapsedi = (finishi.tv_sec - starti.tv_sec);
	elapsedi += (finishi.tv_nsec - starti.tv_nsec) / 1000000000.0;		

	std::cout<<"Initial condition time: " << elapsedi << std::endl;

        double step = 0.0004;

        //evolution
        for(int langevin = 0; langevin < langevin_steps; langevin++){

//		const clock_t begin_time = std::clock();
		struct timespec starte, finishe;
		double elapsede;

		clock_gettime(CLOCK_MONOTONIC, &starte);		


                printf("Performing evolution step no. %i out of %i\n", langevin, langevin_steps);

//		xi_local_x.setToZero();
//		xi_local_y.setToZero();

//              xi_local_x.setGaussian(mpi, cnfg);
//              xi_local_y.setGaussian(mpi, cnfg);

		generate_gaussian_with_noise_coupling_constant(&xi_local_x, &xi_local_y, momtable, mpi, cnfg);

//		generate_gaussian(&xi_local_x, &xi_local_y, mpi, cnfg);

//                fourier2->execute2D(&xi_local_x, 1);
//                fourier2->execute2D(&xi_local_y, 1);

//              xi_local_x_tmp = kernel_pbarx * xi_local_x;
//              xi_local_y_tmp = kernel_pbary * xi_local_y;

//              A_local = xi_local_x_tmp + xi_local_y_tmp;

		//prepare_A_local(&A_local, &xi_local_x, &xi_local_y, &kernel_pbarx, &kernel_pbary);
		//prepare_A_local(&A_local, &xi_local_x, &xi_local_y, momtable);
		prepare_A_local(&A_local, &xi_local_x, &xi_local_y, momtable, mpi, cnfg->CouplingChoice, cnfg->KernelChoice);


                fourier2->execute2D(&A_local, 0);
                fourier2->execute2D(&xi_local_x, 0);
                fourier2->execute2D(&xi_local_y, 0);

//              uf_hermitian = uf.hermitian();
//
//              uxiulocal_x = uf * xi_local_x * (*uf_hermitian);
//		uxiulocal_y = uf * xi_local_y * (*uf_hermitian);
//
//              delete uf_hermitian;

		uxiulocal(&uxiulocal_x, &uxiulocal_y, &uf, &xi_local_x, &xi_local_y);

                fourier2->execute2D(&uxiulocal_x, 1);
                fourier2->execute2D(&uxiulocal_y, 1);

                //uxiulocal_x = kernel_pbarx * uxiulocal_x;
                //uxiulocal_y = kernel_pbary * uxiulocal_y;

                //B_local = uxiulocal_x + uxiulocal_y;

		//prepare_B_local(&B_local, &uxiulocal_x, &uxiulocal_y, &kernel_pbarx, &kernel_pbary);
		//prepare_A_local(&B_local, &uxiulocal_x, &uxiulocal_y, momtable);
		prepare_A_local(&B_local, &uxiulocal_x, &uxiulocal_y, momtable, mpi, cnfg->CouplingChoice, cnfg->KernelChoice);

                fourier2->execute2D(&B_local, 0);
	        
// 		A_local.exponentiate(sqrt(step));

//        	B_local.exponentiate(-sqrt(step));

//        	uf = B_local * uf * A_local;
	
		update_uf(&uf, &B_local, &A_local, step);
	
//		std::cout << float( std::clock () - begin_time ) /  CLOCKS_PER_SEC << std::endl;
		clock_gettime(CLOCK_MONOTONIC, &finishe);
		
		elapsede = (finishe.tv_sec - starte.tv_sec);
		elapsede += (finishe.tv_nsec - starte.tv_nsec) / 1000000000.0;		

		std::cout<<"Evolution time: " << elapsede << std::endl;

	    	//-------------------------------------------------------
		//------CORRELATION FUNCTION-----------------------------
		//-------------------------------------------------------

//		static lfield<double,9> uf_copy(uf);
		uf_copy = uf;

		//compute correlation function
		fourier2->execute2D(&uf_copy,1);
    
		uf_copy.trace(corr);
	
	    	corr_global->allgather(corr, mpi);	

   		corr_global->average_and_symmetrize();

		std::cout<<"Storing partial result at step = "<<langevin<<std::endl;

		corr_global->reduce(&sum[langevin], &err[langevin], mpi);

		std::cout<<"One full evolution, iterating further"<<std::endl;

	}

	clock_gettime(CLOCK_MONOTONIC, &finish);
		
	elapsed = (finish.tv_sec - start.tv_sec);
	elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;		

	std::cout<<"Statistics time: " << elapsed << std::endl;

    }

    const std::string file_name = "test_momentum_evolution_coupling_noise";

    for(int i = langevin_steps - 1; i < langevin_steps; i++){
	    print(i, &sum[i], &err[i], momtable, 1.0/3.0/cnfg->stat, mpi, file_name);
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
 
