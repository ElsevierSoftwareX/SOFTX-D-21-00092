#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include <complex.h>

#include "su3_complex.h"
#include "su3_matrix.h"

#include "config.h"
#include "mpi_pos.h"
#include "mpi_split.h"
#include "mpi_gather.h"
#include "utils.h"

#include <fftw3.h>
#include <fftw3-mpi.h>

#include "field.h"

#include "mpi_fftw_class.h"

//#include "zheevh3.h"

#include <omp.h>

#include <math.h>

#include "mpi_class.h"

int main(int argc, char *argv[]) {


    printf("STARTING CLASS PROGRAM\n");

    mpi_class mpi(argc, argv);

    mpi.mpi_init();

    mpi.mpi_exchange_grid();

    mpi.mpi_exchange_groups();

    printf("TESTING PROGRAM\n");

    double XX[Nx*Ny];
    double YY[Nx*Ny];

    int ii;
    for(ii = 0; ii < Nx*Ny; ii++){
	XX[ii] = 0;
    }
    for(ii = 0; ii < Nx*Ny; ii++){
	YY[ii] = 0;
    }

    int xx, yy;

    for(xx = 0; xx < Nxl; xx++){
    	for(yy = 0; yy < Nyl; yy++){

		XX[buf_pos(xx, yy)] = mpi.getRank()*100+loc_pos(xx, yy) + 1;
	}
    }

    for(xx = 0; xx < Nx; xx++){
    	for(yy = 0; yy < Ny; yy++){

		XX[yy + Ny*xx] = yy + Ny*xx + 1;
	}
    }


//    mpi_split(XX, p);

//    mpi_gather(YY, p);

    printf("FIELD ALLOCATION AND BOUNDARY EXCHANGE\n");

    field<double> f(32,32);

    f.mpi_exchange_boundaries();

    printf("REMAINING JUNK\n");


    fftw_init_threads();
    fftw_mpi_init();

    fftw_plan_with_nthreads(omp_get_max_threads());

    const ptrdiff_t N0 = Nx, N1 = Ny;
    fftw_plan plan;
    fftw_complex *data;
    ptrdiff_t alloc_local, local_n0, local_n0_start, i, j;


    	/* get local data size and allocate */
    	alloc_local = fftw_mpi_local_size_2d(N0, N1, MPI_COMM_WORLD,
                                         &local_n0, &local_n0_start);

	printf("rank %i: N0 = %i, N1 = %i, local: Nxl = %i, Nyl = %i\n", mpi.getRank(), N0, N1, Nxl, Nyl);
	printf("rank %i: local_n0 = %i\n", mpi.getRank(), local_n0);
	printf("rank %i: local_n0_start = %i\n", mpi.getRank(), local_n0_start);

    	data = fftw_alloc_complex(alloc_local);

    	/* create plan for in-place forward DFT */
	plan = fftw_mpi_plan_dft_2d(N0, N1, data, data, MPI_COMM_WORLD,
                                FFTW_FORWARD, FFTW_ESTIMATE);


//int iter;
//for(iter = 0; iter < 200; iter++){


    	/* initialize data to some function my_function(x,y) */
    	for (i = 0; i < local_n0; ++i) for (j = 0; j < N1; ++j){
       		data[i*N1 + j][0] = 1.0*sin((i+local_n0_start)*3.1415/N0)*cos(j*3.1415/N1);
       		data[i*N1 + j][1] = 0.0;

	//	printf("input data (%i, %i) = %e + i%e\n", i+local_n0_start, j, data[i*N1+j][0], data[i*N1+j][1]);
    	}


    	/* compute transforms, in-place, as many times as desired */
    	fftw_execute(plan);

//}

//  	for (i = 0; i < local_n0; ++i) for (j = 0; j < N1; ++j){
//		printf("output data (%i, %i) = %e + i%e\n", i+local_n0_start, j, data[i*N1+j][0], data[i*N1+j][1]);
//  	}



    	fftw_destroy_plan(plan);
//------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------

    printf("STARTING TRUE FFTW 1D\n");


    fftw_complex data_global[N0*N1];
    double data_global_re[N0*N1];
    double data_global_im[N0*N1];

    //proc_grid = y + procy * x
    int pos_x = mpi.getRank()/(procy);
    int pos_y = mpi.getRank() - pos_x * (procy);


    	for (i = 0; i < N0; ++i) for (j = 0; j < N1; ++j){
       		data_global[i*N1 + j][0] = 1.0*sin(i*3.1415/N0)*cos(j*3.1415/N1);
       		data_global[i*N1 + j][1] = 0.0;

	//	printf("input data (%i, %i) = %e + i%e\n", i, j, data_global[i*N1+j][0], data_global[i*N1+j][1]);
    	}


    fftw_complex data_global_tmp[N0*N1];

    fftw_complex* data_localX;
    fftw_complex* data_localY;

    double data_local_re[Nxl*Nyl];
    double data_local_im[Nxl*Nyl];

     	// get local data size and allocate 
    	alloc_local = fftw_mpi_local_size_1d(N0, mpi.getColComm(), FFTW_FORWARD, FFTW_ESTIMATE,
                                         &local_n0, &local_n0_start, &local_n0, &local_n0_start);

   	data_localY = fftw_alloc_complex(alloc_local);

    	// create plan for in-place forward DFT 
    	plan = fftw_mpi_plan_dft_1d(N0, data_localY, data_localY, mpi.getColComm(),
                                FFTW_FORWARD, FFTW_ESTIMATE);    



//    #pragma omp parallel for simd shared(data_global, data_global_tmp, plan) private(data_localY)
    for(j = pos_y*Nyl; j < (pos_y+1)*Nyl; j++){


 
   	// initialize data to some function my_function(x,y) 
    	for (i = 0; i < Nxl; ++i){
         		data_localY[i][0] = data_global[(i+pos_x*Nxl)*N1+j][0];
          		data_localY[i][1] = data_global[(i+pos_x*Nxl)*N1+j][1];
	}

    	// compute transforms, in-place, as many times as desired 
//    	fftw_execute(plan);
	fftw_mpi_execute_dft(plan, data_localY, data_localY);

    	for (i = 0; i < Nxl; ++i) {
	       		data_global_tmp[(i+pos_x*Nxl)*N1+j][0] = data_localY[i][0];
	       		data_global_tmp[(i+pos_x*Nxl)*N1+j][1] = data_localY[i][1];
    	}

   }

	fftw_free(data_localY);
 	fftw_destroy_plan(plan);

//        fftw_plan plan;
//        fftw_complex *data;
//        ptrdiff_t alloc_local, local_n0, local_n0_start;


     	// get local data size and allocate 
    	alloc_local = fftw_mpi_local_size_1d(N1, mpi.getRowComm(), FFTW_FORWARD, FFTW_ESTIMATE,
                                         &local_n0, &local_n0_start, &local_n0, &local_n0_start);

   	data_localX = fftw_alloc_complex(alloc_local);

    	// create plan for in-place forward DFT 
    	plan = fftw_mpi_plan_dft_1d(N1, data_localX, data_localX, mpi.getRowComm(),
                                FFTW_FORWARD, FFTW_ESTIMATE);    



//    #pragma omp parallel for simd shared(data_global_tmp, data_global, data_local_re, data_local_im, plan) private(data_localX)
    for(i = pos_x*Nxl; i < (pos_x+1)*Nxl; i++){


   	// initialize data to some function my_function(x,y) 
    	for (j = 0; j < Nyl; ++j){
         		data_localX[j][0] = data_global_tmp[i*Ny+j+pos_y*Nyl][0];
          		data_localX[j][1] = data_global_tmp[i*Ny+j+pos_y*Nyl][1];
	}

    	// compute transforms, in-place, as many times as desired 
//	fftw_execute(plan);
    	fftw_mpi_execute_dft(plan, data_localX, data_localX);


    	for (j = 0; j < Nyl; ++j) {
	       		data_global[i*Ny+j+pos_y*Nyl][0] = data_localX[j][0];
	       		data_global[i*Ny+j+pos_y*Nyl][1] = data_localX[j][1];
    	}

    	for (j = 0; j < Nyl; ++j) {
	       		data_local_re[(i-pos_x*Nxl)*Nyl+j] = data_localX[j][0];
	       		data_local_im[(i-pos_x*Nxl)*Nyl+j] = data_localX[j][1];
   	}


   }

	fftw_free(data_localX);
 	fftw_destroy_plan(plan);

//	for (i = 0; i < Nxl; ++i) for (j = 0; j < Nyl; ++j){
//		printf("rank %i: output data (%i, %i) = %e + i%e\n", rank, i+pos_x*Nxl, j+pos_y*Nyl, data_global[(i+pos_x*Nxl)*Ny+j+pos_y*Nyl][0], data_global[(i+pos_x*Nxl)*Ny+j+pos_y*Nyl][1]);
//    	}

    fftw_cleanup_threads();


/*

su3_matrix<double> AA;

double w[3];


AA.m[0] = 2.0;
AA.m[4] = 3.0;
AA.m[8] = 4.0;

AA.zheevh3(&w[0]);

printf("eigenvalues: %e %e %e\n", w[0], w[1], w[2]);

*/

    MPI_Finalize();

return 1;
}

