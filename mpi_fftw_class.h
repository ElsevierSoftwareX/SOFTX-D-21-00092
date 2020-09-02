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
 * File: mpi_fftw_class.h
 * Authors: P. Korcyl
 * Contact: piotr.korcyl@uj.edu.pl
 * 
 * Version: 1.0
 * 
 * Description:
 * Class containing linkage to the fftw3 functions
 * 
 */

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
    const ptrdiff_t volume[2] = {Nx,Ny};

    int Nxl, Nyl;

    fftw_plan planX2K;
    fftw_plan planK2X;
    fftw_complex *data_local;
//    fftw_complex *data_localb;

    ptrdiff_t alloc_local, local_n0, local_n0_start;

    fftw_plan planX2K_single;
    fftw_plan planK2X_single;
    fftw_complex *data_local_single;
//    fftw_complex *data_localb_single;

    ptrdiff_t alloc_local_single, local_n0_single, local_n0_start_single;


    public:

    fftw(config *cnfg){


	int nthreads = 12;

        fftw_init_threads();
        fftw_mpi_init();

	printf("FFTW can be run with %i threads; running with %i threads instead\n", omp_get_max_threads(), nthreads);

        fftw_plan_with_nthreads(nthreads);

	printf("FFTW plan setup with %i threads created\n", nthreads);

        Nxl = cnfg->Nxl;
        Nyl = cnfg->Nyl;

    }

    virtual ~fftw(void){};

//	fftw_free(data_local);
// 	fftw_destroy_plan(planK2X);
// 	fftw_destroy_plan(planX2K);
//
//        fftw_cleanup_threads();
//    }


};

class fftw2D : public fftw {

    public:

    fftw2D(config *cnfg) : fftw{ cnfg} {};

    ~fftw2D(void){

 	fftw_free(data_local);
// 	fftw_free(data_localb);
 	fftw_destroy_plan(planK2X);
 	fftw_destroy_plan(planX2K);

	fftw_free(data_local_single);
//	fftw_free(data_localb_single);
 	fftw_destroy_plan(planK2X_single);
 	fftw_destroy_plan(planX2K_single);

        fftw_cleanup_threads();
    }

    int init2D(void){

	printf("FFTW2D: alloc_local\n");

        /* get local data size and allocate */
        alloc_local_single = fftw_mpi_local_size_2d(N0, N1, MPI_COMM_WORLD,
                                         &local_n0_single, &local_n0_start_single);

//ptrdiff_t fftw_mpi_local_size_many(int rnk, const ptrdiff_t *n,
//                                   ptrdiff_t howmany,
//                                   ptrdiff_t block0,
//                                   MPI_Comm comm,
//                                   ptrdiff_t *local_n0,
//                                   ptrdiff_t *local_0_start);

	alloc_local = fftw_mpi_local_size_many(2, volume,
                                   9,
                                   FFTW_MPI_DEFAULT_BLOCK,
                                   MPI_COMM_WORLD,
                                   &local_n0, &local_n0_start);

	printf("FFTW2D: data_local\n");

        data_local_single = fftw_alloc_complex(alloc_local_single);
//        data_localb = fftw_alloc_complex(alloc_local);

        data_local = fftw_alloc_complex(alloc_local);

	printf("FFTW2D: planX2K\n");

        /* create plan for in-place forward DFT */
        planX2K_single = fftw_mpi_plan_dft_2d(N0, N1, data_local_single, data_local_single, MPI_COMM_WORLD,
                                FFTW_FORWARD, FFTW_ESTIMATE);

//	fftw_plan fftw_mpi_plan_many_dft(int rnk, const ptrdiff_t *n,
//                                 ptrdiff_t howmany, ptrdiff_t block, ptrdiff_t tblock,
//                                 fftw_complex *in, fftw_complex *out,
//                                 MPI_Comm comm, int sign, unsigned flags);

	planX2K = fftw_mpi_plan_many_dft(2, volume,
                                 9, FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK,
                                 data_local, data_local,
                                 MPI_COMM_WORLD, FFTW_FORWARD, FFTW_ESTIMATE);

	printf("FFTW2D: planK2X\n");

        planK2X_single = fftw_mpi_plan_dft_2d(N0, N1, data_local_single, data_local_single, MPI_COMM_WORLD,
                                FFTW_BACKWARD, FFTW_ESTIMATE);

	planK2X = fftw_mpi_plan_many_dft(2, volume,
                                 9, FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK,
                                 data_local, data_local,
                                 MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_ESTIMATE);

	return 1;
    }

template<class T, int t> int execute2D(lfield<T,t>* f, int dir){

       /* compute transforms, in-place, as many times as desired */
//       fftw_execute(planX2K);

int i,j,k;
double scale_after_fft = 1.0/sqrt(N0*N1);

//        if(fft_dir == X2K) then
//            fbwd = FFTW_FORWARD
//            scale_after_fft = real(1.0,kind=REALKND) / &
//             & real(volume_global(),kind=REALKND)
//            plan = plan_fw
//        else if(fft_dir == K2X) then
//            fbwd = FFTW_BACKWARD
//            scale_after_fft = real(1.0,kind=REALKND)
//            plan = plan_bw

//if( dir ){
////	scale_after_fft = 1.0/(1.0*N0*N1);
//	scale_after_fft = 1.0;
//} else {
//	scale_after_fft = 1.0;
//}

int COPY = 1;

if(t == 1 ){

if( COPY ){

	#pragma omp parallel for simd collapse(2) default(shared)
	for (i = 0; i < Nxl; ++i){
		for(j = 0; j < Nyl; j++){
                        data_local_single[i*Nyl+j][0] = f->u[(i*Nyl+j)].real(); //data_global[(i+pos_x*Nxl)*N1+j][0];
                        data_local_single[i*Nyl+j][1] = f->u[(i*Nyl+j)].imag(); //data_global[(i+pos_x*Nxl)*N1+j][1];
		}
        }

        if( dir ){
                fftw_mpi_execute_dft(planX2K_single, data_local_single, data_local_single);
        }else{
                fftw_mpi_execute_dft(planK2X_single, data_local_single, data_local_single);
        }

}

/*
http://www.fftw.org/fftw3_doc/Complex-numbers.html

C++ has its own complex<T> template class, defined in the standard <complex> header file. Reportedly, the C++ standards committee has recently agreed to mandate that the storage format used for this type be binary-compatible with the C99 type, i.e. an array T[2] with consecutive real [0] and imaginary [1] parts. (See report http://www.open-std.org/jtc1/sc22/WG21/docs/papers/2002/n1388.pdf WG21/N1388.) Although not part of the official standard as of this writing, the proposal stated that: “This solution has been tested with all current major implementations of the standard library and shown to be working.” To the extent that this is true, if you have a variable complex<double> *x, you can pass it directly to FFTW via reinterpret_cast<fftw_complex*>(x). 
*/

if( !COPY ){

	std::complex<double>* ptr = f->u;

        if( dir ){
                fftw_mpi_execute_dft(planX2K_single, reinterpret_cast<fftw_complex*>(ptr), reinterpret_cast<fftw_complex*>(ptr));
        }else{
                fftw_mpi_execute_dft(planK2X_single, reinterpret_cast<fftw_complex*>(ptr), reinterpret_cast<fftw_complex*>(ptr));
        }
}

if( COPY ){

	#pragma omp parallel for simd collapse(2) default(shared)
	for (i = 0; i < Nxl; ++i){
		for(j = 0; j < Nyl; j++){
                        f->u[(i*Nyl+j)] = std::complex<double>(data_local_single[i*Nyl+j][0]*scale_after_fft, data_local_single[i*Nyl+j][1]*scale_after_fft);
		}
        }

}

}else if(t == 9){

if( COPY ){

	#pragma omp parallel for simd collapse(2) default(shared)
	for (i = 0; i < Nxl; ++i){
		for(j = 0; j < Nyl; j++){
			for(k = 0; k < 9; k++){
	                        data_local[(i*Nyl+j)*t+k][0] = f->u[(i*Nyl+j)*t+k].real(); //data_global[(i+pos_x*Nxl)*N1+j][0];
        	                data_local[(i*Nyl+j)*t+k][1] = f->u[(i*Nyl+j)*t+k].imag(); //data_global[(i+pos_x*Nxl)*N1+j][1];
			}
		}
        }


        if( dir ){
                fftw_mpi_execute_dft(planX2K, data_local, data_local);
        }else{
                fftw_mpi_execute_dft(planK2X, data_local, data_local);
        }
}

if( !COPY ){

	std::complex<double>* ptr = f->u;

        if( dir ){
                fftw_mpi_execute_dft(planX2K, reinterpret_cast<fftw_complex*>(ptr), reinterpret_cast<fftw_complex*>(ptr));
        }else{
                fftw_mpi_execute_dft(planK2X, reinterpret_cast<fftw_complex*>(ptr), reinterpret_cast<fftw_complex*>(ptr));
        }
}

if( COPY ){

	#pragma omp parallel for simd collapse(2) default(shared)
	for (i = 0; i < Nxl; ++i){
		for(j = 0; j < Nyl; j++){
			for(k = 0; k < 9; k++){
	                        f->u[(i*Nyl+j)*t+k] = std::complex<double>(data_local[(i*Nyl+j)*t+k][0]*scale_after_fft, data_local[(i*Nyl+j)*t+k][1]*scale_after_fft);
			}
		}
        }
}

}


return 1;
}

int execute2D(std::complex<double>* f, int dir){

int i,j,k;
const double scale_after_fft = 1.0/sqrt(N0*N1);

//if( dir ){
//	scale_after_fft = 1.0/(1.0*N0*N1);
//} else {
//	scale_after_fft = 1.0;
//}

	#pragma omp parallel for simd collapse(2) default(shared)
	for (i = 0; i < Nxl; ++i){
		for(j = 0; j < Nyl; j++){
                        data_local_single[i*Nyl+j][0] = f[i*Nyl+j].real(); //data_global[(i+pos_x*Nxl)*N1+j][0];
                        data_local_single[i*Nyl+j][1] = f[i*Nyl+j].imag(); //data_global[(i+pos_x*Nxl)*N1+j][1];
		}
        }

        if( dir ){
                fftw_mpi_execute_dft(planX2K_single, data_local_single, data_local_single);
        }else{
                fftw_mpi_execute_dft(planK2X_single, data_local_single, data_local_single);
        }

	#pragma omp parallel for simd collapse(2) default(shared)
	for (i = 0; i < Nxl; ++i){
		for(j = 0; j < Nyl; j++){
                        f[i*Nyl+j] = std::complex<double>(data_local_single[i*Nyl+j][0]*scale_after_fft, data_local_single[i*Nyl+j][1]*scale_after_fft);
		}
        }

return 1;
}

};
 
#endif
