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

    cnfg->stat = 1;

    mpi_class* mpi = new mpi_class(argc, argv);

    mpi->mpi_init(cnfg);

    mpi->mpi_exchange_grid();

    momenta* momtable = new momenta(cnfg, mpi);

    momtable->set();

    positions postable(cnfg, mpi);

    postable.set();

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

    lfield<double,9> uftmp(cnfg->Nxl, cnfg->Nyl);


    lfield<double,1> zero(cnfg->Nxl, cnfg->Nyl);
//    zero.setToZero();

    int langevin_steps = 50;
    int measurements = 1;

    std::vector<lfield<double,1>> sum(measurements, zero);
    std::vector<lfield<double,1>> err(measurements, zero);

//-------------------------------------------------------
//-------------------------------------------------------

for(int stat = 0; stat < cnfg->stat; stat++){

	//const clock_t begin_time_stat = std::clock();
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
	

		f.setMVModel(MVmodel);

		fourier2->execute2D(&f,1);

		f.solvePoisson(0.0001 * pow(MVmodel->g_parameter,2.0) * MVmodel->mu_parameter, MVmodel->g_parameter, momtable);

		fourier2->execute2D(&f,0);

		uf *= f;
    	}

        clock_gettime(CLOCK_MONOTONIC, &finishi);

        elapsedi = (finishi.tv_sec - starti.tv_sec);
        elapsedi += (finishi.tv_nsec - starti.tv_nsec) / 1000000000.0;

        std::cout<<"Initial condition time: " << elapsedi << std::endl;

        double step = 0.0008;


//loop over possible distances squared
for(int ix = 0; ix < Nx; ix++){
//for(int iy = ix-2; iy < ix+3; iy++){
for(int iy = ix-1; iy < ix+2; iy++){
if(iy >= 0 && iy < Ny){


	printf("COMPUTATIONS FOR ix = %i and iy = %i\n", ix, iy);

        double dix = ix;
        if( dix >= Nx/2 )
	        dix = dix - Nx;
        if( dix < -Nx/2 )
                dix = dix + Nx;

        double diy = iy;
        if( diy >= Ny/2 )
                diy = diy - Ny;
        if( diy < -Ny/2 )
                diy = diy + Ny;

	int rr = dix*dix + diy*diy;

	uftmp = uf;


        //evolution
        for(int langevin = 0; langevin < langevin_steps; langevin++){

                struct timespec starte, finishe;
                double elapsede;

                clock_gettime(CLOCK_MONOTONIC, &starte);

                printf("Performing evolution step no. %i\n", langevin);

		generate_gaussian(&xi_local_x, &xi_local_y, mpi, cnfg);

                printf("gathering local xi to global\n");
                xi_global_x.allgather(&xi_local_x, mpi);
                xi_global_y.allgather(&xi_local_y, mpi);

                printf("gathering local uf to global\n");
                uf_global.allgather(&uftmp, mpi);

		A_local.setToZero();
		B_local.setToZero();

                printf("starting iteration over global lattice\n");
                for(int x = 0; x < cnfg->Nxl; x++){
	                for(int y = 0; y < cnfg->Nyl; y++){

        	                int x_global = x + mpi->getPosX()*cnfg->Nxl;
                                int y_global = y + mpi->getPosY()*cnfg->Nyl;

				//rr should be the distance squared
				prepare_A_and_B_local(x, y, x_global, y_global, &xi_global_x, &xi_global_y, &A_local, &B_local, &uf_global, &postable, rr, HATTA_COUPLING_CONSTANT, LINEAR_KERNEL);

                        }
                }

		update_uf(&uftmp, &B_local, &A_local, step);
		
	        clock_gettime(CLOCK_MONOTONIC, &finishe);

        	elapsede = (finishe.tv_sec - starte.tv_sec);
	        elapsede += (finishe.tv_nsec - starte.tv_nsec) / 1000000000.0;

        	std::cout<<"Evolution time: " << elapsede << std::endl;

	    	//-------------------------------------------------------
		//------CORRELATION FUNCTION-----------------------------
		//-------------------------------------------------------

		if( langevin+1 == langevin_steps ){ //% (int)(langevin_steps / measurements) == 0 ){

			int time = 0; //(int)(langevin * measurements / langevin_steps);

			lfield<double,9> uf_copy(uftmp);

			fourier2->execute2D(&uf_copy,1);
    
			uf_copy.trace(corr);

		    	corr_global->allgather(corr, mpi);	

   			corr_global->average_and_symmetrize();

			corr_global->reduce_hatta(&sum[time], &err[time], mpi, ix, iy);
			//corr_global->reduce(&sum[time], &err[time], mpi);

		}


	}
}
}
}
        clock_gettime(CLOCK_MONOTONIC, &finish);

       	elapsed = (finish.tv_sec - start.tv_sec);
        elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;

       	std::cout<<"Statistics time: " << elapsed << std::endl;

    }

    //for(int i = measurements-1; i < measurements; i++){
    //        print(&sum[i], &err[i], momtable, 1.0/3.0/cnfg->stat, mpi);
    //}
    //print(&sum[0], &err[0], momtable, 1.0/3.0/cnfg->stat, mpi);

    const std::string file_name = "position_evolution_coupling_hatta";

    //for(int i = measurements-1; i < measurements; i++){
            print(0, &sum[0], &err[0], momtable, 1.0/3.0/cnfg->stat, mpi, file_name);
    //}


//-------------------------------------------------------
//------DEALLOCATE AND CLEAN UP--------------------------
//-------------------------------------------------------

    delete cnfg;

    delete momtable;

    delete MVmodel;

    delete fourier2;

    delete mpi;

    delete corr;

    delete corr_global;

    MPI_Finalize();


return 1;
}
 
