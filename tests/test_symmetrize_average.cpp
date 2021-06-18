#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

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

    cnfg->stat = 160;

    mpi_class* mpi = new mpi_class(argc, argv);

    mpi->mpi_init(cnfg);

    mpi->mpi_exchange_grid();


    momenta* momtable = new momenta(cnfg, mpi);

    momtable->set();

    rand_class* random_generator = new rand_class(mpi,cnfg);

    MV_class* MVmodel = new MV_class(1.0, 0.48, 4);

    fftw2D* fourier2 = new fftw2D(cnfg);

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

    lfield<double,1> zero(cnfg->Nxl, cnfg->Nyl);

    std::vector<lfield<double,1>> sum(1, zero);
    std::vector<lfield<double,1>> err(1, zero);


//-------------------------------------------------------
//-------------------------------------------------------

for(int stat = 0; stat < cnfg->stat; stat++){


	printf("Gatherting stat sample no. %i\n", stat);

	//-------------------------------------------------------
	//------INITIAL STATE------------------------------------
	//-------------------------------------------------------

	printf("Constructing initial state from the MV model\n");

	uf.setToOne();

    	for(int i = 0; i < MVmodel->NyGet(); i++){
	
		printf("Iteration %i\n", i);

		f.setToOne();

		uf = uf * f;
    	}

    	//-------------------------------------------------------
	//------CORRELATION FUNCTION-----------------------------
	//-------------------------------------------------------

	uf.trace(corr);

    	corr_global->allgather(corr, mpi);	

	corr_global->average_and_symmetrize();

        corr_global->reduce(&sum[0], &err[0], mpi);

    }

    for(int i = 0; i < 1; i++){
            printf("iterator = %i\n", i);
            print_position(i, &sum[i], &err[i], momtable, cnfg->stat, mpi, "test_average_symmetrize_output");
    }

    printf("Expected result: should be %e on each site\n", (pow(3.0,4.0)*pow(3.0,4.0)*3.0)*3.0);

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
 
