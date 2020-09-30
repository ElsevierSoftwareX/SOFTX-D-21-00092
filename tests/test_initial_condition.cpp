#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include <time.h>

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

    cnfg->stat = 100;

    mpi_class* mpi = new mpi_class(argc, argv);

    mpi->mpi_init(cnfg);

    mpi->mpi_exchange_grid();

    momenta* momtable = new momenta(cnfg, mpi);

    momtable->set();

    rand_class* random_generator = new rand_class(mpi,cnfg);

    MV_class* MVmodel = new MV_class(1.0, 0.03, 50);

//    fftw1D* fourier = new fftw1D(cnfg);

    fftw2D* fourier2 = new fftw2D(cnfg);

//    fourier->init1D(mpi->getRowComm(), mpi->getColComm());    

    fourier2->init2D();    

    printf("ALLOCATION\n");

//allocation
//-------------------------------------------------------
    //construct initial state
    lfield<double,9> f(cnfg->Nxl, cnfg->Nyl);
    lfield<double,9> uf(cnfg->Nxl, cnfg->Nyl);

//-------------------------------------------------------

    //correlation function
    lfield<double,1>* corr = new lfield<double,1>(cnfg->Nxl, cnfg->Nyl);
    gfield<double,1>* corr_global = new gfield<double,1>(Nx, Ny);

//-------------------------------------------------------
//------ACCUMULATE STATISTICS----------------------------
//-------------------------------------------------------

//    std::vector<lfield<double,1>*> accumulator;
    lfield<double,1> sum(cnfg->Nxl, cnfg->Nyl);

    sum.setToZero();

//-------------------------------------------------------
//-------------------------------------------------------

for(int stat = 0; stat < cnfg->stat; stat++){

//      const clock_t begin_time_stat = std::clock();
        struct timespec start, finish;
        double elapsed;

        clock_gettime(CLOCK_MONOTONIC, &start);


	printf("Gatherting stat sample no. %i\n", stat);

	//-------------------------------------------------------
	//------INITIAL STATE------------------------------------
	//-------------------------------------------------------

        uf.setToUnit();

        for(int i = 0; i < MVmodel->NyGet(); i++){

                //f.setToZero();

                f.setMVModel(MVmodel);

                fourier2->execute2D(&f,1);

                f.solvePoisson(0.0001 * pow(MVmodel->gGet(),2.0) * MVmodel->muGet(), MVmodel->gGet(), momtable);

                fourier2->execute2D(&f,0);

                //f.exponentiate();

                uf *= f;
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

        sum += *corr_ptr;

        delete corr_ptr;

        clock_gettime(CLOCK_MONOTONIC, &finish);

        elapsed = (finish.tv_sec - start.tv_sec);
        elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;

        std::cout<<"Statistics time: " << elapsed << std::endl;

    }

    sum.print(momtable, 1.0/3.0/cnfg->stat, mpi);

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
 
