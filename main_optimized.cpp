/* 
 * This file is part of the JIMWLK numerical solution package (https://github.com/piotrkorcyl/jimwlk).
 * Copyright (c) 2020 P. Korcyl
 * 
 * This program is free software: you can redistribute it and/or modify  
 * it under the terms of the GNU General Public License as published by  
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful, but 
 * WITHOUT ANY WARRANTY; without even the implied warranty of 
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License 
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 * 
 * File: main_optimized.cpp
 * Authors: P. Korcyl
 * Contact: piotr.korcyl@uj.edu.pl
 * 
 * Version: 1.0
 * 
 * Description:
 * Main function, contains all functionality, optimized with respect to threading
 * 
 */


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

#include "kinematical_constraints.h"

int main(int argc, char *argv[]) {

    printf("INITIALIZATION\n");

    config* cnfg = new config;

    printf("SETUP: reading setup from %s configuration file\n", argv[2]);

    if(argc == 3){
	cnfg->read_config_from_file(argv[2]);
    }else{
	printf("Usage: mpi_optimized MPI_processes configuration_file\n");
	exit(0);
    }

    mpi_class* mpi = new mpi_class(argc, argv);

    mpi->mpi_init(cnfg);

    mpi->mpi_exchange_grid();

    momenta* momtable = new momenta(cnfg, mpi);

    momtable->set();

    positions postable(cnfg, mpi);

    postable.set();

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
    gfield<double,9> uf_copy_global(Nx, Ny);

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
    gfield<double,9> uf_global_zero(Nx, Ny);

    lfield<double,9> uf_copy(cnfg->Nxl, cnfg->Nyl);

    //correlation function
    lfield<double,1>* corr = new lfield<double,1>(cnfg->Nxl, cnfg->Nyl);
    gfield<double,1>* corr_global = new gfield<double,1>(Nx, Ny);

    std::vector<lfield<double,1>> sum(cnfg->measurements, zero);
    std::vector<lfield<double,1>> err(cnfg->measurements, zero);

//-------------------------------------------------------
//-------KEEPING U HISTORY-------------------------------
//-------------------------------------------------------

//first we need to decide how many scale r we need and how many history steps for each of them we may need
//
//

for(int ix = 0; ix < 1; ix++){
for(int iy = ix+1; iy < 2; iy++){
//    int ix = 2;
//    int iy = 2;

    int initial_scale = ix*ix+iy*iy;

    int rr[1000];
    int rr_rap[1000];

    rr_rap[0] = 0;
//
    int rapidities = kinematical_constraints(initial_scale, cnfg->langevin_steps, cnfg->step, rr, rr_rap);

//
    printf("THIS SETUP HAS INITIAL SCALE SQUARED %i \n", initial_scale);
    printf("THIS SETUP REQUIRES %i SEPARATE RAPIDITIES TO BE SIMULATED!!!\n", rapidities);
    printf("THIS SETUP INVOLVES ADDITIONAL SCALES:\n");
    for(int ii = 0; ii < rapidities; ii++){
	printf("%i\n", rr[ii]);
    }
    printf("THIS SETUP INVOLVES ADDITIONAL SCALES STARTING AT RAPIDITIES:\n");
    for(int ii = 0; ii < rapidities; ii++){
	printf("%i\n", rr_rap[ii]);
    }

    printf("STARTING SIMULATION\n");

//

//    rr = (int*)malloc(rapidities*sizeof(int));

    std::vector<gfield<double,9>> evolution(rapidities, uf_global_zero);


//-------------------------------------------------------
//------MAIN STAT LOOP-----------------------------------
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

	for(int rap = 0; rap < rapidities; rap++){

		uf.setToUnit();

    		for(int i = 0; i < MVmodel->NyGet(); i++){
	
			f.setMVModel(MVmodel);

			fourier2->execute2D(&f,1);

			f.solvePoisson(cnfg->mass * pow(MVmodel->gGet(),2.0) * MVmodel->muGet(), MVmodel->gGet(), momtable);

			fourier2->execute2D(&f,0);

			uf *= f;
    		}

		uf_global.allgather(&uf, mpi);

		evolution[rap] = uf_global;
	}

        clock_gettime(CLOCK_MONOTONIC, &finishi);

        elapsedi = (finishi.tv_sec - starti.tv_sec);
        elapsedi += (finishi.tv_nsec - starti.tv_nsec) / 1000000000.0;

        std::cout<<"Initial condition time: " << elapsedi << std::endl;

		//----------------------------------------------------------------
		//------------MAIN EVOLUTION LOOP---------------------------------
	        //----------------------------------------------------------------
	        for(int langevin = 0; langevin < cnfg->langevin_steps; langevin++){

			printf("langevin step %i\n", langevin);

	                struct timespec starte, finishe;
	                double elapsede;

        	        clock_gettime(CLOCK_MONOTONIC, &starte);

			for(int rap = 0; rap < rapidities; rap++){

				//if the evolution of that scale started, we do it
				if(langevin >= rr_rap[rap]){

					printf("running evolution step for scale %i\n", rr[rap]);

					uf_global = evolution[rap];

				 	uf_global.reduce_position(&uftmp, mpi);

					generate_gaussian(&xi_local_x, &xi_local_y, mpi, cnfg);
						
					//----------POSITION SPACE EVOLUTION-------------
				
					xi_global_x.allgather(&xi_local_x, mpi);
					xi_global_y.allgather(&xi_local_y, mpi);

		        	        //uf_global.allgather(&uftmp, mpi);

					A_local.setToZero();
					B_local.setToZero();

			                for(int x = 0; x < cnfg->Nxl; x++){
       						for(int y = 0; y < cnfg->Nyl; y++){

	       				                int x_global = x + mpi->getPosX()*cnfg->Nxl;
      				        		int y_global = y + mpi->getPosY()*cnfg->Nyl;

							prepare_A_and_B_local_with_history(x, y, x_global, y_global, &xi_global_x, &xi_global_y, &A_local, 
								&B_local, &uf_global, &postable, rr, rap, rapidities, cnfg->CouplingChoice, cnfg->KernelChoice, evolution, langevin, cnfg->step);

       						}
               				}

					MPI_Barrier(MPI_COMM_WORLD);
	
					//-----------UPDATE WILSON LINES-----------------

					update_uf(&uftmp, &B_local, &A_local, cnfg->step);
		
				        clock_gettime(CLOCK_MONOTONIC, &finishe);

				       	elapsede = (finishe.tv_sec - starte.tv_sec);
				        elapsede += (finishe.tv_nsec - starte.tv_nsec) / 1000000000.0;

				       	std::cout<<"Evolution time: " << elapsede << std::endl;

				        uf_global.allgather(&uftmp, mpi);

					evolution[rap] = uf_global;
				
				}//if evolve that scale

			    	//-------------------------------------------------------
				//------CORRELATION FUNCTION-----------------------------
				//-------------------------------------------------------

				if( langevin % (int)(cnfg->langevin_steps / cnfg->measurements) == 0 ){

					int time = (int)(langevin * cnfg->measurements / cnfg->langevin_steps);

					printf("time = %i\n", time);

					evolution[0].average_reduce_hatta(&sum[time], &err[time], mpi, ix, iy);				
				}
	
			}//end for loop over the scales

		}//end evolution loop

	//if no evolution - compute correlation function directly from initial condition
	if( cnfg->EvolutionChoice == NO_EVOLUTION ){

		int time = 0;

		uf_copy = uf;

		fourier2->execute2D(&uf_copy,1);
    
		uf_copy.trace(corr);

	    	corr_global->allgather(corr, mpi);	

		corr_global->reduce(&sum[time], &err[time], mpi);
	}


        clock_gettime(CLOCK_MONOTONIC, &finish);

       	elapsed = (finish.tv_sec - start.tv_sec);
        elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;

       	std::cout<<"Statistics time: " << elapsed << std::endl;

}//end of stat loop

}
}

//------------------------------------------------------
//----------WRITE DOWN CORRELATION FNUCTION TO FILE-----
//------------------------------------------------------

//        std::string file_name = "output_optimized_";
	std::string file_name = cnfg->file_name;

                if(cnfg->EvolutionChoice == MOMENTUM_EVOLUTION)
                        file_name += "_MOMENTUM_EVOLUTION_";
                
                if(cnfg->EvolutionChoice == POSITION_EVOLUTION)
                        file_name += "_POSITION_EVOLUTION_";
  
		if(cnfg->EvolutionChoice == NO_EVOLUTION)
                        file_name += "_NO_EVOLUTION_";
             

                if(cnfg->CouplingChoice == SQRT_COUPLING_CONSTANT)
                        file_name += "SQRT_COUPLING_CONSTANT_";
                
                if(cnfg->CouplingChoice == NOISE_COUPLING_CONSTANT)
                        file_name += "NOISE_COUPLING_CONSTANT_";
                
                if(cnfg->CouplingChoice == HATTA_COUPLING_CONSTANT)
                        file_name += "HATTA_COUPLING_CONSTANT_";
  
		if(cnfg->CouplingChoice == NO_COUPLING_CONSTANT)
                        file_name += "NO_COUPLING_CONSTANT_";
             

                if(cnfg->KernelChoice == LINEAR_KERNEL)
                        file_name += "LINEAR_KERNEL";
                
                if(cnfg->KernelChoice == SIN_KERNEL)
                        file_name += "SIN_KERNEL";
               
	std::cout << "pelna nazwa pliku: " << file_name << "\n";


    for(int i = 0; i < cnfg->measurements; i++){
	    printf("iterator = %i\n", i);
            print(i, &sum[i], &err[i], momtable, cnfg->stat, mpi, file_name);
    }

//-------------------------------------------------------
//------CLEAN FINISH: DEALLOCATE AND CLEAN UP------------
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


