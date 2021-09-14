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
//    gfield<double,9> fglobal(Nx, Ny);
//    gfield<double,9> fglobal_average(Nx, Ny);
    lfield<double,9> uf(cnfg->Nxl, cnfg->Nyl);

//    gfield<double,9> uf_global(Nx, Ny);
//    gfield<double,9> uf_copy_global(Nx, Ny);

    //evolution
//    lfield<double,9> xi_local_x(cnfg->Nxl, cnfg->Nyl);
//    lfield<double,9> xi_local_y(cnfg->Nxl, cnfg->Nyl);

//    gfield<double,9> xi_global_x(Nx, Ny);
//    gfield<double,9> xi_global_y(Nx, Ny);

//    lfield<double,9> A_local(cnfg->Nxl, cnfg->Nyl);
//    lfield<double,9> B_local(cnfg->Nxl, cnfg->Nyl);

//    lfield<double,9> uftmp(cnfg->Nxl, cnfg->Nyl);

//-------------------------------------------------------

//    lfield<double,9> uxiulocal_x(cnfg->Nxl, cnfg->Nyl);
//    lfield<double,9> uxiulocal_y(cnfg->Nxl, cnfg->Nyl);

//-------------------------------------------------------
//------ACCUMULATE STATISTICS----------------------------
//-------------------------------------------------------

    lfield<double,1> zero(cnfg->Nxl, cnfg->Nyl);

    lfield<double,9> uf_copy(cnfg->Nxl, cnfg->Nyl);

    //correlation function
    lfield<double,1>* corr = new lfield<double,1>(cnfg->Nxl, cnfg->Nyl);
    gfield<double,1>* corr_global = new gfield<double,1>(Nx, Ny);

    lfield<double,1>* initial_corr = new lfield<double,1>(cnfg->Nxl, cnfg->Nyl);


    std::vector<lfield<double,1>> sum(cnfg->measurements, zero);
    std::vector<lfield<double,1>> err(cnfg->measurements, zero);

//-------------------------------------------------------
//-------------------------------------------------------

//create sigma correlation matrix for the noise vectors in position space
//perform cholesky decomposition to get the square root of the correlation matrix

gmatrix<double>* cholesky;
/*
if(cnfg->EvolutionChoice == POSITION_EVOLUTION && cnfg->CouplingChoice == NOISE_COUPLING_CONSTANT){

        corr->setCorrelationsForCouplingConstant(momtable);
	
        fourier2->execute2D(corr, 0);

        corr_global->allgather(corr, mpi);

        cholesky = new gmatrix<double>(Nx*Ny,Nx*Ny);

        cholesky->decompose(corr_global);
        printf("cholesky decomposition finished\n");
}
*/

initial_corr->setCorrelationsGaussian(momtable, 32.0, mpi);

fourier2->execute2D(initial_corr,1);


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

	uf.setToUnit();

//	int* source_positions = (int*)malloc(2*Nx*Ny*sizeof(int));
//	double* source_values = (double*)malloc(8*Nx*Ny*sizeof(double));

    	for(int i = 0; i < MVmodel->NyGet(); i++){
	
		//f.setMVModel(MVmodel);
		//f.setGaussianModel(momtable, 1.0);
		f.setGaussianModel(initial_corr, MVmodel);

/*
		const double disp = pow(MVmodel->gGet(),2.0) * MVmodel->muGet() / sqrt(MVmodel->NyGet());
     
		if( mpi->getRank() == 0 ){



		        #pragma omp parallel for simd default(shared)
			for(int is = 0; is < Nx*Ny; is++){

		                static __thread std::ranlux24* generator = nullptr;
                		if (!generator){
		                         std::hash<std::thread::id> hasher;
                		         generator = new std::ranlux24(clock() + hasher(std::this_thread::get_id()));
		                }
                		std::uniform_int_distribution<> uniform_distribution_x(0, Nx-1);
		                std::uniform_int_distribution<> uniform_distribution_y(0, Ny-1);
				std::normal_distribution<double> distribution{  0.0, disp };

	                	int negative_x = uniform_distribution_x(*generator);
	        	        int negative_y = uniform_distribution_y(*generator);
        	        	int positive_x = uniform_distribution_x(*generator);
	        	        int positive_y = uniform_distribution_y(*generator);
				int negative = negative_x*Ny+negative_y;
	                	int positive = positive_x*Ny+positive_y;

				source_positions[2*is+0] = positive;
				source_positions[2*is+1] = negative;

                		for(int k = 0; k < 8; k++){
		                        source_values[is*8+k] = distribution(*generator);
                		}

			}
		}	

		MPI_Bcast(source_positions, 2*Nx*Ny, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(source_values, 8*Nx*Ny, MPI_DOUBLE, 0, MPI_COMM_WORLD);

		f.setMVModel(MVmodel, source_positions, source_values, mpi);

		//fglobal_average.allreduce(&fglobal, mpi);

                //fglobal_average.reduce_position(&f, mpi);
*/
//		fourier2->execute2D(&f,1);

//		f.solvePoisson(cnfg->mass * pow(MVmodel->gGet(),2.0) * MVmodel->muGet(), MVmodel->gGet(), momtable);
//		f.solvePoisson(cnfg->mass, MVmodel->gGet(), momtable);

		fourier2->execute2D(&f,0);

		uf *= f;
    	}

//	free(source_positions);
//	free(source_values);

        clock_gettime(CLOCK_MONOTONIC, &finishi);

        elapsedi = (finishi.tv_sec - starti.tv_sec);
        elapsedi += (finishi.tv_nsec - starti.tv_nsec) / 1000000000.0;

        std::cout<<"Initial condition time: " << elapsedi << std::endl;

/*
	//-------------------------------------------------------
	//---------------IF EVOLUTION----------------------------
	//------------------------------------------------------- 
	
	if( cnfg->EvolutionChoice != NO_EVOLUTION ){

		//-------SETUP FOR HATTA COUPLING CONSTANT-----------

		int upper_bound_x = 1;
		if( cnfg->CouplingChoice == HATTA_COUPLING_CONSTANT ){
			upper_bound_x = 48; //Nx;
		}
		//hatta iteration over positions
		//loop over possible distances squared
		for(int ix = 0; ix < upper_bound_x; ix++){

			int upper_bound_y = 1;
			int lower_bound_y = 0;
			if( cnfg->CouplingChoice == HATTA_COUPLING_CONSTANT ){ 
				lower_bound_y = ix; //-1;
				upper_bound_y = ix+1; //+2;
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

					//----------------------------------------------------------------------------
					//---------KEEP ORIGINAL INITIAL CONDITION AND ITERATE OVER DISTANCES rr_hatta
					//----------------------------------------------------------------------------

				        uftmp = uf;
		
					//----------------------------------------------------------------
					//------------MAIN EVOLUTION LOOP---------------------------------
				        //----------------------------------------------------------------
				        for(int langevin = 0; langevin < cnfg->langevin_steps; langevin++){

				                struct timespec starte, finishe;
				                double elapsede;

				                clock_gettime(CLOCK_MONOTONIC, &starte);

				                printf("Performing evolution step no. %i for position %i %i out of %i %i\n", langevin, ix, iy, upper_bound_x, upper_bound_y);

						if( cnfg->EvolutionChoice == MOMENTUM_EVOLUTION && cnfg->CouplingChoice == NOISE_COUPLING_CONSTANT ){
							generate_gaussian_with_noise_coupling_constant(&xi_local_x, &xi_local_y, momtable, mpi, cnfg);
						}else{
							generate_gaussian(&xi_local_x, &xi_local_y, mpi, cnfg);
						}

						//----------MOMENTUM SPACE EVOLUTION-------------

						if( cnfg->EvolutionChoice == MOMENTUM_EVOLUTION ){

							if( cnfg->CouplingChoice == SQRT_COUPLING_CONSTANT || cnfg->CouplingChoice == NO_COUPLING_CONSTANT){
						                fourier2->execute2D(&xi_local_x, 1);
					        	        fourier2->execute2D(&xi_local_y, 1);
							}

							prepare_A_local(&A_local, &xi_local_x, &xi_local_y, momtable, mpi, cnfg->CouplingChoice, cnfg->KernelChoice);

				                	fourier2->execute2D(&A_local, 0);
					                fourier2->execute2D(&xi_local_x, 0);
				        	        fourier2->execute2D(&xi_local_y, 0);

							uxiulocal(&uxiulocal_x, &uxiulocal_y, &uftmp, &xi_local_x, &xi_local_y);

					                fourier2->execute2D(&uxiulocal_x, 1);
				        	        fourier2->execute2D(&uxiulocal_y, 1);

				       			prepare_A_local(&B_local, &uxiulocal_x, &uxiulocal_y, momtable, mpi, cnfg->CouplingChoice, cnfg->KernelChoice);

				                	fourier2->execute2D(&B_local, 0);

							MPI_Barrier(MPI_COMM_WORLD);
	        
						}//end if momentum evolution	
		
						//----------POSITION SPACE EVOLUTION-------------
				
						if( cnfg->EvolutionChoice == POSITION_EVOLUTION ){

				                	printf("gathering local xi to global\n");
					                xi_global_x.allgather(&xi_local_x, mpi);
				        	        xi_global_y.allgather(&xi_local_y, mpi);

							if( cnfg->CouplingChoice == NOISE_COUPLING_CONSTANT ){

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

									prepare_A_and_B_local(x, y, x_global, y_global, &xi_global_x, &xi_global_y, &A_local, 
											&B_local, &uf_global, &postable, rr_hatta, cnfg->CouplingChoice, cnfg->KernelChoice);

                        					}
                					}

							MPI_Barrier(MPI_COMM_WORLD);

						}//end if position space evolution

						//-----------UPDATE WILSON LINES-----------------

						update_uf(&uftmp, &B_local, &A_local, cnfg->step);
		
					        clock_gettime(CLOCK_MONOTONIC, &finishe);

				        	elapsede = (finishe.tv_sec - starte.tv_sec);
					        elapsede += (finishe.tv_nsec - starte.tv_nsec) / 1000000000.0;

				        	std::cout<<"Evolution time: " << elapsede << std::endl;


						if( (langevin == 100) || (langevin == 200) || (langevin == 500) ){

							printf("RESOLUTION REDUCTION!!!!\n");
							printf("fine-graining at langevin step = %i\n", langevin);

						        uf_global.allgather(&uftmp, mpi);
	
						        uf_global.fine_grain(uf_copy_global);

						        uf_copy_global.reduce_position(&uftmp, mpi);

						}

					    	//-------------------------------------------------------
						//------CORRELATION FUNCTION-----------------------------
						//-------------------------------------------------------

						if( langevin % (int)(cnfg->langevin_steps / cnfg->measurements) == 0 ){

							int time = (int)(langevin * cnfg->measurements / cnfg->langevin_steps);

							uf_copy = uftmp;

							if( cnfg->CouplingChoice != HATTA_COUPLING_CONSTANT ){

								fourier2->execute2D(&uf_copy,1);
    	
								uf_copy.trace(corr);

							    	corr_global->allgather(corr, mpi);	

					   			corr_global->average_and_symmetrize();

								corr_global->reduce_position(corr, mpi);

								fourier2->execute2D(corr,0);

							    	corr_global->allgather(corr, mpi);	

								corr_global->reduce(&sum[time], &err[time], mpi);
		
							}else{

					        	        uf_copy_global.allgather(&uf_copy, mpi);

								uf_copy_global.average_reduce_hatta(&sum[time], &err[time], mpi, ix, iy);
							}
							
						}

					}//end evolution loop

		//end of hatta iteration over positions
		}}}

    	}//if evolution
*/

	//if no evolution - compute correlation function directly from initial condition
	if( cnfg->EvolutionChoice == NO_EVOLUTION ){

		int time = 0;

		uf_copy = uf;

                fourier2->execute2D(&uf_copy,1);
        
                uf_copy.trace(corr);

                corr_global->allgather(corr, mpi);      

                corr_global->average_and_symmetrize();

                corr_global->reduce_position(corr, mpi);

                fourier2->execute2D(corr,0);

                corr_global->allgather(corr, mpi);      

                corr_global->reduce(&sum[time], &err[time], mpi);
	}


        clock_gettime(CLOCK_MONOTONIC, &finish);

       	elapsed = (finish.tv_sec - start.tv_sec);
        elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;

       	std::cout<<"Statistics time: " << elapsed << std::endl;

}//end of stat loop

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
            print_position(i, &sum[i], &err[i], momtable, cnfg->stat, mpi, file_name);
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


