#ifndef H_MPI_FFTW
#define H_MPI_FFTW

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include <complex>

#include "su3_complex.h"
#include "su3_matrix.h"

#include "config.h"

#include "mpi_class.h"

#include <fftw3.h>
#include <fftw3-mpi.h>

//#include "zheevh3.h"

#include <omp.h>

#include <mpi.h>

#include <math.h>

#include "field.h"

class fftw {

    protected:

    const ptrdiff_t N0 = Nx, N1 = Ny;

    int Nxl, Nyl;

    fftw_plan planX2K;
    fftw_plan planK2X;
    fftw_complex *data_local;
    fftw_complex *data_localb;

    ptrdiff_t alloc_local, local_n0, local_n0_start;

    public:

    fftw(config *cnfg){


	int nthreads = 6;

        fftw_init_threads();
        fftw_mpi_init();

	printf("FFTW can be run with %i threads; running with %i threads instead\n", omp_get_max_threads(), nthreads);

        fftw_plan_with_nthreads(nthreads);

	printf("FFTW plan setup with %i threads created\n", nthreads);

        Nxl = cnfg->Nxl;
        Nyl = cnfg->Nyl;

    }

    ~fftw(void){

	fftw_free(data_local);
 	fftw_destroy_plan(planK2X);
 	fftw_destroy_plan(planX2K);

        fftw_cleanup_threads();
    }


};

class fftw1D : public fftw { 

    protected:

//    fftw_plan planX;
//    fftw_complex *data_localX;
    fftw_plan planYK2X;
    fftw_plan planYX2K;

    fftw_complex *data_localY;

//    ptrdiff_t alloc_localX, local_n0X, local_n0_startX;
    ptrdiff_t alloc_localY, local_n0Y, local_n0_startY;

    public:

    fftw1D(config *cnfg) : fftw{ cnfg} {};

    ~fftw1D(void){

	fftw_free(data_localY);

 	fftw_destroy_plan(planYX2K);
 	fftw_destroy_plan(planYK2X);

    }


    int init1D(MPI_Comm col_comm, MPI_Comm row_comm){

     	// get local data size and allocate 
    	alloc_localY = fftw_mpi_local_size_1d(N0, col_comm, FFTW_FORWARD, FFTW_ESTIMATE,
                                         &local_n0Y, &local_n0_startY, &local_n0Y, &local_n0_startY);

   	data_localY = fftw_alloc_complex(alloc_localY);

    	// create plan for in-place forward DFT 
    	planYX2K = fftw_mpi_plan_dft_1d(N0, data_localY, data_localY, col_comm,
                                FFTW_FORWARD, FFTW_ESTIMATE);    


    	planYK2X = fftw_mpi_plan_dft_1d(N0, data_localY, data_localY, col_comm,
                                FFTW_BACKWARD, FFTW_ESTIMATE);    


     	// get local data size and allocate 
    	alloc_local = fftw_mpi_local_size_1d(N1, row_comm, FFTW_FORWARD, FFTW_ESTIMATE,
                                         &local_n0, &local_n0_start, &local_n0, &local_n0_start);

   	data_local = fftw_alloc_complex(alloc_local);

    	// create plan for in-place forward DFT 
    	planX2K = fftw_mpi_plan_dft_1d(N1, data_local, data_local, row_comm,
                                FFTW_FORWARD, FFTW_ESTIMATE);    

    	planK2X = fftw_mpi_plan_dft_1d(N1, data_local, data_local, row_comm,
                                FFTW_BACKWARD, FFTW_ESTIMATE);    

	return 1;
    }

    template<class T, int t> int execute1D(lfield<T,t>* f, int dir){

    int i,j,k;

    int pos_x = 0;
    int pos_y = 0;

double scale_after_fft_X;
double scale_after_fft_Y;

//        if(fft_dir == X2K) then
//            fbwd = FFTW_FORWARD
//            scale_after_fft = real(1.0,kind=REALKND) / &
//             & real(volume_global(),kind=REALKND)
//            plan = plan_fw
//        else if(fft_dir == K2X) then
//            fbwd = FFTW_BACKWARD
//            scale_after_fft = real(1.0,kind=REALKND)
//            plan = plan_bw

if( dir ){
	scale_after_fft_X = 1.0/(1.0*N0);
	scale_after_fft_Y = 1.0/(1.0*N1);
} else {
	scale_after_fft_X = 1.0;
	scale_after_fft_Y = 1.0;
}



//    printf("in execute1D: t = %i\n", t);

for(k = 0; k < t; k++){

//	 printf("in execute1D: component: k = %i\n",k);

//    for(j = pos_y*Nyl; j < (pos_y+1)*Nyl; j++){
     for(j = 0; j < Nyl; j++){

//	 printf("in execute1D: iterating over Y: j=%i\n",j);

   	// initialize data to some function my_function(x,y) 
    	for (i = 0; i < Nxl; ++i){
         		data_localY[i][0] = f->u[k][i*Nyl+j].real(); //data_global[(i+pos_x*Nxl)*N1+j][0];
          		data_localY[i][1] = f->u[k][i*Nyl+j].imag(); //data_global[(i+pos_x*Nxl)*N1+j][1];
	}

    	// compute transforms, in-place, as many times as desired 
	if( dir ){
		fftw_mpi_execute_dft(planYX2K, data_localY, data_localY);
	}else{
		fftw_mpi_execute_dft(planYK2X, data_localY, data_localY);
	}

    	for (i = 0; i < Nxl; ++i) {
			f->u[k][i*Nyl+j] = (data_localY[i][0] + I*data_localY[i][1])*scale_after_fft_Y;

//	       		data_global_tmp[(i+pos_x*Nxl)*N1+j][0] = data_localY[i][0];
//	       		data_global_tmp[(i+pos_x*Nxl)*N1+j][1] = data_localY[i][1];
    	}

    }

//    for(i = pos_x*Nxl; i < (pos_x+1)*Nxl; i++){
    for(i = 0; i < Nxl; i++){

//	 printf("in execute1D: iterating over X: i=%i\n",i);

   	// initialize data to some function my_function(x,y) 
    	for (j = 0; j < Nyl; ++j){
         		data_localY[j][0] = f->u[k][i*Nyl+j].real(); //data_global_tmp[i*Ny+j+pos_y*Nyl][0];
          		data_localY[j][1] = f->u[k][i*Nyl+j].imag(); //data_global_tmp[i*Ny+j+pos_y*Nyl][1];
	}

    	// compute transforms, in-place, as many times as desired 
	if( dir ){
	    	fftw_mpi_execute_dft(planX2K, data_localY, data_localY);
	}else{
	    	fftw_mpi_execute_dft(planK2X, data_localY, data_localY);
	}

    	for (j = 0; j < Nyl; ++j) {

	       		f->u[k][i*Nyl+j] = (data_localY[j][0] + I*data_localY[j][1])*scale_after_fft_X;

//	       		data_global[i*Ny+j+pos_y*Nyl][0] = data_localX[j][0];
//	       		data_global[i*Ny+j+pos_y*Nyl][1] = data_localX[j][1];

    	}
   }
}

return 1;
}


};

class fftw2D : public fftw {

    public:

    fftw2D(config *cnfg) : fftw{ cnfg} {};

    int init2D(void){

	printf("FFTW2D: alloc_local\n");

        /* get local data size and allocate */
        alloc_local = fftw_mpi_local_size_2d(N0, N1, MPI_COMM_WORLD,
                                         &local_n0, &local_n0_start);

	printf("FFTW2D: data_local\n");

        data_local = fftw_alloc_complex(alloc_local);
        data_localb = fftw_alloc_complex(alloc_local);

	printf("FFTW2D: planX2K\n");

        /* create plan for in-place forward DFT */
        planX2K = fftw_mpi_plan_dft_2d(N0, N1, data_local, data_local, MPI_COMM_WORLD,
                                FFTW_FORWARD, FFTW_ESTIMATE);

	printf("FFTW2D: planK2X\n");

        planK2X = fftw_mpi_plan_dft_2d(N0, N1, data_localb, data_localb, MPI_COMM_WORLD,
                                FFTW_BACKWARD, FFTW_ESTIMATE);

	
	return 1;
    }

template<class T, int t> int execute2D(lfield<T,t>* f, int dir){

       /* compute transforms, in-place, as many times as desired */
//       fftw_execute(planX2K);

int i,j,k;
double scale_after_fft;

//        if(fft_dir == X2K) then
//            fbwd = FFTW_FORWARD
//            scale_after_fft = real(1.0,kind=REALKND) / &
//             & real(volume_global(),kind=REALKND)
//            plan = plan_fw
//        else if(fft_dir == K2X) then
//            fbwd = FFTW_BACKWARD
//            scale_after_fft = real(1.0,kind=REALKND)
//            plan = plan_bw

if( dir ){
	scale_after_fft = 1.0/(1.0*N0*N1);
} else {
	scale_after_fft = 1.0;
}



for(k = 0; k < t; k++){

	#pragma omp parallel for simd collapse(2) default(shared)
	for (i = 0; i < Nxl; ++i){
		for(j = 0; j < Nyl; j++){
                        data_local[i*Nyl+j][0] = f->u[k][i*Nyl+j].real(); //data_global[(i+pos_x*Nxl)*N1+j][0];
                        data_local[i*Nyl+j][1] = f->u[k][i*Nyl+j].imag(); //data_global[(i+pos_x*Nxl)*N1+j][1];
		}
        }


        if( dir ){
                fftw_mpi_execute_dft(planX2K, data_local, data_local);
        }else{
                fftw_mpi_execute_dft(planK2X, data_local, data_local);
        }

	#pragma omp parallel for simd collapse(2) default(shared)
	for (i = 0; i < Nxl; ++i){
		for(j = 0; j < Nyl; j++){
                        f->u[k][i*Nyl+j] = std::complex<double>(data_local[i*Nyl+j][0]*scale_after_fft, data_local[i*Nyl+j][1]*scale_after_fft);
		}
        }

}

return 1;
}

};
  
#endif
