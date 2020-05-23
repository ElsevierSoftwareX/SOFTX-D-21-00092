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
#include "../positions.h"

#include "../rand_class.h"

#include "../MV_class.h"

#include <numeric>

//#include "single_field.h"

int main(int argc, char *argv[]) {

    printf("INITIALIZATION\n");

//initialization

    config* cnfg = new config;

    cnfg->stat = 16;

    mpi_class* mpi = new mpi_class(argc, argv);

    mpi->mpi_init(cnfg);

    mpi->mpi_exchange_grid();

    mpi->mpi_exchange_groups();

    momenta* momtable = new momenta(cnfg, mpi);

    momtable->set();

    positions* postable = new positions(cnfg, mpi);

    postable->set();

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

    gfield<double,9> uf_global(Nx, Ny);

    //evolution
    lfield<double,9> xi_local_x(cnfg->Nxl, cnfg->Nyl);
    lfield<double,9> xi_local_y(cnfg->Nxl, cnfg->Nyl);

    gfield<double,9> xi_global_x(Nx, Ny);
    gfield<double,9> xi_global_y(Nx, Ny);
    gfield<double,9> xi_global_x_tmp(Nx, Ny);
    gfield<double,9> xi_global_y_tmp(Nx, Ny);

    gfield<double,9> uxiu_global_tmp(Nx,Ny);
    gfield<double,9> xi_global_tmp(Nx,Ny);

    //initiaization of kernel fields
    gfield<double,9> kernel_xbarx(Nx, Ny);
    gfield<double,9> kernel_xbary(Nx, Ny);

    lfield<double,9> A_local(cnfg->Nxl, cnfg->Nyl);
    lfield<double,9> B_local(cnfg->Nxl, cnfg->Nyl);

    lfield<double,9> uxiulocal_x(cnfg->Nxl, cnfg->Nyl);
    lfield<double,9> uxiulocal_y(cnfg->Nxl, cnfg->Nyl);

    gfield<double,9>* uf_global_hermitian;


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

		fourier2->execute2D(&f,1);

		f.solvePoisson(0.0001 * pow(MVmodel->g_parameter,2.0) * MVmodel->mu_parameter, MVmodel->g_parameter, momtable);

		fourier2->execute2D(&f,0);

		f.exponentiate();

		uf *= f;
    	}

        double step = 0.0001;

        //evolution
        for(int langevin = 0; langevin < 400; langevin++){

		const clock_t begin_time = std::clock();

                printf("Performing evolution step no. %i\n", langevin);

		xi_local_x.setToZero();
		xi_local_y.setToZero();

                xi_local_x.setGaussian(random_generator,1);
                xi_local_y.setGaussian(random_generator,2);

                printf("gathering local xi to global\n");
                xi_global_x.allgather(&xi_local_x, mpi);
                xi_global_y.allgather(&xi_local_y, mpi);

                printf("gathering local uf to global\n");
                uf_global.allgather(&uf, mpi);

		A_local.setToZero();
		B_local.setToZero();

                uf_global_hermitian = uf_global.hermitian();

                printf("starting iteration over global lattice\n");
                //for(int i = 0; i < cnfg->Nxl*cnfg->Nyl; i++){
                for(int x = 0; x < cnfg->Nxl; x++){
	                for(int y = 0; y < cnfg->Nyl; y++){

        	                int x_global = x + mpi->getPosX()*cnfg->Nxl;
                                int y_global = y + mpi->getPosY()*cnfg->Nyl;

				kernel_xbarx.setToZero();
				kernel_xbary.setToZero();

                                kernel_xbarx.setKernelXbarX(x_global, y_global, postable);
                                kernel_xbary.setKernelXbarY(x_global, y_global, postable);

                                xi_global_x_tmp = kernel_xbarx * xi_global_x;
                                xi_global_y_tmp = kernel_xbary * xi_global_y;


                                xi_global_tmp = xi_global_x_tmp + xi_global_y_tmp;


                                A_local.reduceAndSet(x, y, &xi_global_tmp);


                                uxiu_global_tmp = uf_global * xi_global_tmp * (*uf_global_hermitian);

                                B_local.reduceAndSet(x, y, &uxiu_global_tmp);
                        }
                }

                delete uf_global_hermitian;

		A_local.exponentiate(sqrt(step));

        	B_local.exponentiate(-sqrt(step));

        	uf = B_local * uf * A_local;
		
		std::cout << float( std::clock () - begin_time ) /  CLOCKS_PER_SEC << std::endl;
	}

    	//-------------------------------------------------------
	//------CORRELATION FUNCTION-----------------------------
	//-------------------------------------------------------

	//compute correlation function
	//should be X2K
//   	fourier->execute1D(&uf, 0);
	fourier2->execute2D(&uf,1);
    
	uf.trace(corr);

    	corr_global->allgather(corr, mpi);	

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

    sum.print(momtable, 1.0/3.0/accumulator.size(), mpi);


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
 
