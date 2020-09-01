#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <ctime>

#include <complex.h>

#include "../su3_matrix.h"
#include "../matrix.h"

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


int main(int argc, char *argv[]) {

    printf("INITIALIZATION\n");

    config* cnfg = new config;

    mpi_class* mpi = new mpi_class(argc, argv);

    mpi->mpi_init(cnfg);

    mpi->mpi_exchange_grid();

    momenta* momtable = new momenta(cnfg, mpi);

    momtable->set();

    positions postable(cnfg, mpi);

    postable.set();

    rand_class* random_generator = new rand_class(mpi,cnfg);

    MV_class* MVmodel = new MV_class(1.0, cnfg->mu/Nx, cnfg->elementaryWilsonLines);

    fftw2D* fourier2 = new fftw2D(cnfg);

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

    lfield<double,9> uftmp(cnfg->Nxl, cnfg->Nyl);

//-------------------------------------------------------

    lfield<double,9> uxiulocal_x(cnfg->Nxl, cnfg->Nyl);
    lfield<double,9> uxiulocal_y(cnfg->Nxl, cnfg->Nyl);

//-------------------------------------------------------
//------ACCUMULATE STATISTICS----------------------------
//-------------------------------------------------------

    lfield<double,1> zero(cnfg->Nxl, cnfg->Nyl);

    lfield<double,9> uf_copy(cnfg->Nxl, cnfg->Nyl);

    //correlation function
    lfield<double,1>* corr = new lfield<double,1>(cnfg->Nxl, cnfg->Nyl);
    gfield<double,1>* corr_global = new gfield<double,1>(Nx, Ny);

    std::vector<lfield<double,1>> sum(cnfg->langevin_steps, zero);
    std::vector<lfield<double,1>> err(cnfg->langevin_steps, zero);

//-------------------------------------------------------
//-------------------------------------------------------

//create sigma correlation matrix for the noise vectors in position space
//perform cholesky decomposition to get the square root of the correlation matrix

gmatrix<double>* cholesky;

if(cnfg->position_evolution == 1 && cnfg->noise_coupling_constant == 1){

        corr->setCorrelationsForCouplingConstant(momtable);
	
        fourier2->execute2D(corr, 0);

        corr_global->allgather(corr, mpi);

        cholesky = new gmatrix<double>(Nx*Ny,Nx*Ny);

        cholesky->decompose(corr_global);
        printf("cholesky decomposition finished\n");
}

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
	
		f.setMVModel(MVmodel, random_generator);

		fourier2->execute2D(&f,1);

		f.solvePoisson(cnfg->mass * pow(MVmodel->g_parameter,2.0) * MVmodel->mu_parameter, MVmodel->g_parameter, momtable);

		fourier2->execute2D(&f,0);

		uf *= f;
    	}

        clock_gettime(CLOCK_MONOTONIC, &finishi);

        elapsedi = (finishi.tv_sec - starti.tv_sec);
        elapsedi += (finishi.tv_nsec - starti.tv_nsec) / 1000000000.0;

        std::cout<<"Initial condition time: " << elapsedi << std::endl;

int upper_bound_x = 1;
if( cnfg->hatta_coupling_constant == 1 ){
	upper_bound_x = Nx;
}
//hatta iteration over positions
//loop over possible distances squared
for(int ix = 0; ix < upper_bound_x; ix++){

int upper_bound_y = 1;
int lower_bound_y = 0;
if( cnfg->hatta_coupling_constant == 1 ){ 
	lower_bound_y = ix-1;
	upper_bound_y = ix+2;
}
for(int iy = lower_bound_y; iy < upper_bound_y; iy++){
if(iy >= 0 && iy < Ny){

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

        int rr_hatta = dix*dix + diy*diy;

        uftmp = uf;

        //evolution
        for(int langevin = 0; langevin < cnfg->langevin_steps; langevin++){

                struct timespec starte, finishe;
                double elapsede;

                clock_gettime(CLOCK_MONOTONIC, &starte);

                printf("Performing evolution step no. %i\n", langevin);

		if( cnfg->momentum_evolution == 1 && cnfg->noise_coupling_constant == 1 ){
			generate_gaussian_with_noise_coupling_constant(&xi_local_x, &xi_local_y, momtable, mpi, cnfg);
		}else{
			generate_gaussian(&xi_local_x, &xi_local_y, mpi, cnfg);
		}

		if( cnfg->momentum_evolution == 1){

	                fourier2->execute2D(&xi_local_x, 1);
        	        fourier2->execute2D(&xi_local_y, 1);

			if( cnfg->sqrt_coupling_constant == 1 ){
				//set coupling on
				prepare_A_local(&A_local, &xi_local_x, &xi_local_y, momtable, mpi, SQRT_COUPLING_CONSTANT, cnfg->KernelChoice);
			}
			if( cnfg->noise_coupling_constant == 1 ){
				//set coupling off
				prepare_A_local(&A_local, &xi_local_x, &xi_local_y, momtable, mpi, NOISE_COUPLING_CONSTANT, cnfg->KernelChoice);
			}

                	fourier2->execute2D(&A_local, 0);
	                fourier2->execute2D(&xi_local_x, 0);
        	        fourier2->execute2D(&xi_local_y, 0);

			uxiulocal(&uxiulocal_x, &uxiulocal_y, &uftmp, &xi_local_x, &xi_local_y);

	                fourier2->execute2D(&uxiulocal_x, 1);
        	        fourier2->execute2D(&uxiulocal_y, 1);

			if( cnfg->sqrt_coupling_constant == 1 ){
				//set coupling on
	       			prepare_A_local(&B_local, &uxiulocal_x, &uxiulocal_y, momtable, mpi, SQRT_COUPLING_CONSTANT, cnfg->KernelChoice);
			}
			if( cnfg->noise_coupling_constant == 1 ){
				//set coupling off
	       			prepare_A_local(&B_local, &uxiulocal_x, &uxiulocal_y, momtable, mpi, NOISE_COUPLING_CONSTANT, cnfg->KernelChoice);
			}

                	fourier2->execute2D(&B_local, 0);
	        
		}	
		
		if( cnfg->position_evolution == 1){

                	printf("gathering local xi to global\n");
	                xi_global_x.allgather(&xi_local_x, mpi);
        	        xi_global_y.allgather(&xi_local_y, mpi);

			if( cnfg->noise_coupling_constant == 1){

		                xi_global_x.multiplyByCholesky(cholesky);
                		xi_global_y.multiplyByCholesky(cholesky);

			}

	                printf("gathering local uf to global\n");
        	        uf_global.allgather(&uftmp, mpi);

			A_local.setToZero();
			B_local.setToZero();

        	        printf("starting iteration over global lattice\n");
	                for(int x = 0; x < cnfg->Nxl; x++){
	        	        for(int y = 0; y < cnfg->Nyl; y++){

        		                int x_global = x + mpi->getPosX()*cnfg->Nxl;
                	                int y_global = y + mpi->getPosY()*cnfg->Nyl;

					if( cnfg->noise_coupling_constant == 1 ){
						//coupling has to be turned off
						prepare_A_and_B_local(x, y, x_global, y_global, &xi_global_x, &xi_global_y, &A_local, &B_local, &uf_global, &postable, 0, NOISE_COUPLING_CONSTANT, cnfg->KernelChoice);
					}
					if( cnfg->sqrt_coupling_constant == 1 ){
						//coupling has to be turned on
						prepare_A_and_B_local(x, y, x_global, y_global, &xi_global_x, &xi_global_y, &A_local, &B_local, &uf_global, &postable, 0, SQRT_COUPLING_CONSTANT, cnfg->KernelChoice);
					}
					if( cnfg->hatta_coupling_constant == 1 ){
						//coupling has to be turned on
						prepare_A_and_B_local(x, y, x_global, y_global, &xi_global_x, &xi_global_y, &A_local, &B_local, &uf_global, &postable, rr_hatta, HATTA_COUPLING_CONSTANT, cnfg->KernelChoice);
					}


                        	}
                	}

		}

		update_uf(&uftmp, &B_local, &A_local, cnfg->step);
		
	        clock_gettime(CLOCK_MONOTONIC, &finishe);

        	elapsede = (finishe.tv_sec - starte.tv_sec);
	        elapsede += (finishe.tv_nsec - starte.tv_nsec) / 1000000000.0;

        	std::cout<<"Evolution time: " << elapsede << std::endl;

	    	//-------------------------------------------------------
		//------CORRELATION FUNCTION-----------------------------
		//-------------------------------------------------------

		if( langevin % (int)(cnfg->langevin_steps / cnfg->measurements) == 0 ){

			int time = (int)(langevin * cnfg->measurements / cnfg->langevin_steps);

			uf_copy = uftmp;

			fourier2->execute2D(&uf_copy,1);
    
			uf_copy.trace(corr);

		    	corr_global->allgather(corr, mpi);	

   			corr_global->average_and_symmetrize();

			if( cnfg->hatta_coupling_constant == 1 ){
				corr_global->reduce_hatta(&sum[time], &err[time], mpi, ix, iy);
			}else{
				corr_global->reduce(&sum[time], &err[time], mpi);
			}
		}
	}

//end of hatta iteration over positions
}}}

        clock_gettime(CLOCK_MONOTONIC, &finish);

       	elapsed = (finish.tv_sec - start.tv_sec);
        elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;

       	std::cout<<"Statistics time: " << elapsed << std::endl;

    }

    const std::string file_name = "output_optimized";

    for(int i = cnfg->measurements-1; i < cnfg->measurements; i++){
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


