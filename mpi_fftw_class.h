#ifndef H_MPI_FFTW
#define H_MPI_FFTW

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include <complex.h>

#include "su3_complex.h"
#include "su3_matrix.h"

#include "config.h"

#include "mpi_class.h"

#include "mpi_pos.h"
#include "mpi_split.h"
#include "mpi_gather.h"
#include "utils.h"

#include <fftw3.h>
#include <fftw3-mpi.h>

//#include "zheevh3.h"

#include <omp.h>

#include <mpi.h>

#include <math.h>

class fftw {

    private:

    const ptrdiff_t N0 = Nx, N1 = Ny;
    fftw_plan planX;
    fftw_complex *data_localX;
    fftw_plan planY;
    fftw_complex *data_localY;

    ptrdiff_t alloc_localX, local_n0X, local_n0_startX;
    ptrdiff_t alloc_localY, local_n0Y, local_n0_startY;


    fftw_plan plan;
    fftw_complex *data;

    ptrdiff_t alloc_local, local_n0, local_n0_start;

    public:

    fftw(void){


        fftw_init_threads();
        fftw_mpi_init();
        fftw_plan_with_nthreads(omp_get_max_threads());

    }

    int init1D(MPI_Comm col_comm, MPI_Comm row_comm){

     	// get local data size and allocate 
    	alloc_localY = fftw_mpi_local_size_1d(N0, col_comm, FFTW_FORWARD, FFTW_ESTIMATE,
                                         &local_n0Y, &local_n0_startY, &local_n0Y, &local_n0_startY);

   	data_localY = fftw_alloc_complex(alloc_localY);

    	// create plan for in-place forward DFT 
    	planY = fftw_mpi_plan_dft_1d(N0, data_localY, data_localY, col_comm,
                                FFTW_FORWARD, FFTW_ESTIMATE);    


     	// get local data size and allocate 
    	alloc_localX = fftw_mpi_local_size_1d(N1, row_comm, FFTW_FORWARD, FFTW_ESTIMATE,
                                         &local_n0X, &local_n0_startX, &local_n0X, &local_n0_startX);

   	data_localX = fftw_alloc_complex(alloc_localX);

    	// create plan for in-place forward DFT 
    	planX = fftw_mpi_plan_dft_1d(N1, data_localX, data_localX, row_comm,
                                FFTW_FORWARD, FFTW_ESTIMATE);    
    }

    int init2D(void){


        /* get local data size and allocate */
        alloc_local = fftw_mpi_local_size_2d(N0, N1, MPI_COMM_WORLD,
                                         &local_n0, &local_n0_start);

        data = fftw_alloc_complex(alloc_local);

        /* create plan for in-place forward DFT */
        plan = fftw_mpi_plan_dft_2d(N0, N1, data, data, MPI_COMM_WORLD,
                                FFTW_FORWARD, FFTW_ESTIMATE);
    }


    ~fftw(void){

	fftw_free(data_localX);
 	fftw_destroy_plan(planX);

	fftw_free(data_localY);
 	fftw_destroy_plan(planY);

        fftw_cleanup_threads();
    }

    int execute1d(void){

    int size;

    int rank;

    //proc_grid = y + procy * x
    int pos_x = rank/(procy);
    int pos_y = rank - pos_x * (procy);


    printf("STARTING TRUE FFTW 1D\n");

    int i,j;

    fftw_complex data_global[N0*N1];
    double data_global_re[N0*N1];
    double data_global_im[N0*N1];


    fftw_complex data_global_tmp[N0*N1];

    fftw_complex* data_localX;
    fftw_complex* data_localY;

    double data_local_re[Nxl*Nyl];
    double data_local_im[Nxl*Nyl];


    for(j = pos_y*Nyl; j < (pos_y+1)*Nyl; j++){
 
   	// initialize data to some function my_function(x,y) 
    	for (i = 0; i < Nxl; ++i){
         		data_localY[i][0] = data_global[(i+pos_x*Nxl)*N1+j][0];
          		data_localY[i][1] = data_global[(i+pos_x*Nxl)*N1+j][1];
	}

    	// compute transforms, in-place, as many times as desired 
	fftw_mpi_execute_dft(planY, data_localY, data_localY);

    	for (i = 0; i < Nxl; ++i) {
	       		data_global_tmp[(i+pos_x*Nxl)*N1+j][0] = data_localY[i][0];
	       		data_global_tmp[(i+pos_x*Nxl)*N1+j][1] = data_localY[i][1];
    	}

    }

    for(i = pos_x*Nxl; i < (pos_x+1)*Nxl; i++){

   	// initialize data to some function my_function(x,y) 
    	for (j = 0; j < Nyl; ++j){
         		data_localX[j][0] = data_global_tmp[i*Ny+j+pos_y*Nyl][0];
          		data_localX[j][1] = data_global_tmp[i*Ny+j+pos_y*Nyl][1];
	}

    	// compute transforms, in-place, as many times as desired 
    	fftw_mpi_execute_dft(planX, data_localX, data_localX);

    	for (j = 0; j < Nyl; ++j) {
	       		data_global[i*Ny+j+pos_y*Nyl][0] = data_localX[j][0];
	       		data_global[i*Ny+j+pos_y*Nyl][1] = data_localX[j][1];
    	}

    	for (j = 0; j < Nyl; ++j) {
	       		data_local_re[(i-pos_x*Nxl)*Nyl+j] = data_localX[j][0];
	       		data_local_im[(i-pos_x*Nxl)*Nyl+j] = data_localX[j][1];
   	}


   }

}

   int execute2d(void){

       /* compute transforms, in-place, as many times as desired */
       fftw_execute(plan);

    }

};

#endif
