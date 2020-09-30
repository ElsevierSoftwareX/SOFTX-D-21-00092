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
 * File: field.h
 * Authors: P. Korcyl
 * Contact: piotr.korcyl@uj.edu.pl
 * 
 * Version: 1.0
 * 
 * Description:
 * Classes representing lfield and gfield objects, basic ingredients of the implementation
 * 
 */

#ifndef H_FIELD
#define H_FIELD

#include <iostream>
#include <stdlib.h>
#include <complex>
#include <ccomplex>

#include "config.h"

#include <omp.h>

#include <mpi.h>
#include <math.h>

#include "mpi_class.h"

#include "MV_class.h"
#include "rand_class.h"

#include "momenta.h"
#include "positions.h"

#include <random>
#include <time.h>
#include <thread>


/********************************************//**
 * Definition of the main class of type field. Contains pointer to the data array and Nx and Ny sizes.
 ***********************************************/
template<class T, int t> class field {

	private:

		int Nxl, Nyl;

	public:

		std::complex<T>* u;		

		field(int NNx, int NNy);
		field(const field<T,t> &in);
		virtual ~field(void){};
};

/********************************************//**
 * Constructor. Allocates memory and sets it to zero. 
 ***********************************************/
template<class T, int t> field<T,t>::field(int NNx, int NNy) {

	u = NULL;

	u = (std::complex<T>*)malloc(t*NNx*NNy*sizeof(std::complex<T>));

	if(u == NULL){
		printf("field contructor: malloc unsuccessful. Aborting.\n");
		exit(0);
	}

	for(int j = 0; j < t*NNx*NNy; j++)
			u[j] = 0.0;

	Nxl = NNx;
	Nyl = NNy;
}

/********************************************//**
 * Copy constructor.
************************************************/
template<class T, int t> field<T,t>::field(const field<T,t> &in) {

	//std::cout<<"Executing base class copy constructor"<<std::endl;

	this->u = NULL;

	this->u = (std::complex<T>*)malloc(t*in.Nxl*in.Nyl*sizeof(std::complex<T>));

	if(u == NULL){
		printf("field copy contructor: malloc unsuccessful. Aborting.\n");
		exit(0);
	}

	for(int j = 0; j < t*in.Nxl*in.Nyl; j++)
			this->u[j] = in.u[j];

	this->Nxl = in.Nxl;
	this->Nyl = in.Nyl;

}


template<class T, int t> class lfield;

template<class T> class gmatrix;

/********************************************//**
 * Derived class gfield. Contains the entire object on each node. Nxg and Nyg refer to the total size of the lattice.
 ***********************************************/
template<class T, int t> class gfield: public field<T,t> {

	/// @brief Specialization of the field class to a GLOBAL object.
  	/// @param sizes of the global lattice
  	/// @throws exception in case of errors.

	private:

		int Nxg, Nyg;

	public:

		int allgather(lfield<T,t>* ulocal, mpi_class* mpi);

		gfield(int NNx, int NNy) : field<T,t>{NNx, NNy} { Nxg = NNx; Nyg = NNy;};

		~gfield();

		friend class fftw2D;

		inline int getNxg(void) const {
			return Nxg;
		}

		inline int getNyg(void) const {
			return Nyg;
		}

		int average_and_symmetrize();

		gfield<T,t>& operator= ( const gfield<T,t>& f );

		gfield<T,t>* hermitian();

		int setKernelXbarX(int x, int y, positions* postable, Kernel KernelChoice);
		int setKernelXbarY(int x, int y, positions* postable, Kernel KernelChoice);
		int setKernelXbarXWithCouplingConstant(int x, int y, positions* postable, Kernel KernelChoice);
		int setKernelXbarYWithCouplingConstant(int x, int y, positions* postable, Kernel KernelChoice);

		int setCorrelationsForCouplingConstant();

		int multiplyByCholesky(gmatrix<T>* mm);

		lfield<T,t>* reduce(int NNx, int NNy, mpi_class* mpi);
		int reduce(lfield<T,t>* sum, lfield<T,t>* err, mpi_class* mpi);
		int reduce_hatta(lfield<T,t>* sum, lfield<T,t>* err, mpi_class* mpi, int xr, int yr);

		int setToZero(void){
			for(int i = 0; i < t*Nxg*Nyg; i ++){
				this->u[i] = 0.0;
			}
		return 1;
		}

		int printDebug(void);

};

/********************************************//**
 * Derived class destructor.
 ***********************************************/
template<class T, int t> gfield<T,t>::~gfield() {

	free(this->u);

}

template<class T, int t> class lfield: public field<T,t> {

	private:

		int Nxl, Nyl;
		int Nxl_buf, Nyl_buf;

	public:

		lfield(int NNx, int NNy) : field<T,t>{NNx, NNy} { Nxl = NNx; Nyl = NNy; };
		lfield(const lfield<T,t> &in);

		~lfield();

		friend class fftw2D;

		int mpi_exchange_boundaries(mpi_class* mpi);

		friend class gfield<T,t>;

		inline int getNxl(void) const {
			return Nxl;
		}

		inline int getNyl(void) const {
			return Nyl;
		}

		int loc_pos(int x, int y){

			return y + Nyl*x;
		}

		int buf_pos(int x, int y){

			return ((y+(Nyl_buf-Nyl)/2) + Nyl_buf*(x+(Nxl_buf-Nxl)/2));
		}

		int buf_pos_ex(int x, int y){

			return (y + Nyl_buf*x);
		}


		int setToZero(void){

			#pragma omp parallel for simd default(shared)
			for(int i = 0; i < t*Nxl*Nyl; i ++){
				this->u[i] = 0.0;
			}
		return 1;
		}


		int setToUnit(void){
			#pragma omp parallel for simd default(shared)
			for(int i = 0; i < t*Nxl*Nyl; i+=t){
				this->u[i+0] = 1.0;
				this->u[i+4] = 1.0;
				this->u[i+8] = 1.0;

				this->u[i+1] = 0.0;
				this->u[i+2] = 0.0;
				this->u[i+3] = 0.0;
				this->u[i+5] = 0.0;
				this->u[i+6] = 0.0;
				this->u[i+7] = 0.0;
			}
		return 1;
		}
		int setToOne(void){
			#pragma omp parallel for simd default(shared)
			for(int i = 0; i < t*Nxl*Nyl; i ++){
				this->u[i] = 1.0;
			}
		return 1;
		}

		int setToFixed(void){
			for(int ix = 0; ix < Nxl; ix ++){
				for(int iy = 0; iy < Nyl; iy ++){

					for(int k = 0; k < t; k++){
						if( ix < Nxl/2 && iy < Nyl/2 ){
							this->u[(ix*Nyl + iy)*t+k] = ((ix)%Nxl)*((ix)%Nxl) + ((iy)%Nyl)*((iy)%Nyl);
						}
						if( ix > Nxl/2 && iy < Nyl/2 ){
							this->u[(ix*Nyl + iy)*t+k] = ((ix-Nxl)%Nxl)*((ix-Nxl)%Nxl) + ((iy)%Nyl)*((iy)%Nyl);
						}
						if( ix < Nxl/2 && iy > Nyl/2 ){
							this->u[(ix*Nyl + iy)*t+k] = ((ix)%Nxl)*((ix)%Nxl) + ((iy-Nyl)%Nyl)*((iy-Nyl)%Nyl);
						}
						if( ix > Nxl/2 && iy > Nyl/2 ){
							this->u[(ix*Nyl + iy)*t+k] = ((ix-Nxl)%Nxl)*((ix-Nxl)%Nxl) + ((iy-Nyl)%Nyl)*((iy-Nyl)%Nyl);
						}
					}
				}
			}
		return 1;
		}


		int setMVModel(MV_class* MVconfig);
		int setUnitModel(rand_class* rr);

		int setGaussian(mpi_class* rr, config* cnfg);

		int solvePoisson(double mass, double g, momenta* momtable);

		int exponentiate();
		int exponentiate(double s);

		int setKernelPbarX(momenta* mom, mpi_class* mpi, Kernel KernelChoice);
		int setKernelPbarY(momenta* mom, mpi_class* mpi, Kernel KernelChoice);

		int setKernelPbarXWithCouplingConstant(momenta* mom, mpi_class* mpi, Kernel KernelChoice);
		int setKernelPbarYWithCouplingConstant(momenta* mom, mpi_class* mpi, Kernel KernelChoice);

		int setCorrelationsForCouplingConstant(momenta* mom);

		lfield<T,t>* hermitian();

		lfield<T,t>& operator= ( const lfield<T,t>& f );
		lfield<T,t>& operator*= ( const lfield<T,t>& f );
		lfield<T,t>& operator+= ( const lfield<T,t>& f );

		int trace(lfield<double,1>* cc);
		int average(lfield<double,1>* cc);

		int reduceAndSet(int x, int y, gfield<T,t>* field);

		int print(momenta* mom);
		int print(momenta* mom, double x, mpi_class* mpi);


		int printDebug();
		int printDebug(double x);
		int printDebug(int i);
		int printDebug(double x, mpi_class* mpi);
		int printDebugRadial(double x);


};

template<class T, int t> lfield<T,t>::lfield(const lfield<T,t> &in) : field<T,t>(in) {

	//std::cout<<"Executing derived class copy constructor"<<std::endl;

	this->Nxl = in.getNxl();
	this->Nyl = in.getNyl();

}


template<class T, int t> lfield<T,t>::~lfield() {

	free(this->u);

}

template<class T, int t> lfield<T,t>& lfield<T,t>::operator= ( const lfield<T,t>& f ){

			if( this != &f ){

			#pragma omp parallel for simd default(shared)
			for(int i = 0; i < f.getNxl()*f.getNyl(); i ++){
		
				for(int k = 0; k < t; k++){

					this->u[i*t+k] = f.u[i*t+k];
				}
			}


			}

		return *this;
		}

template<class T, int t> gfield<T,t>& gfield<T,t>::operator= ( const gfield<T,t>& f ){

			if( this != &f ){

			#pragma omp parallel for simd default(shared)
			for(int i = 0; i < f.getNxg()*f.getNyg(); i ++){
				for(int k = 0; k < t; k++){

					this->u[i*t+k] = f.u[i*t+k];
				}
			}

			}

		return *this;
		}

template<class T, int t> lfield<T,t>& lfield<T,t>::operator*= ( const lfield<T,t>& f ){

			if( this != &f ){

			#pragma omp parallel for simd default(shared)
			for(int i = 0; i < f.getNxl()*f.getNyl(); i++){
			
				su3_matrix<double> A,B,C;

				for(int k = 0; k < t; k++){

					A.m[k] = this->u[i*t+k];
					B.m[k] = f.u[i*t+k];
				}

		                B.exponentiate(1.0);
		
				C = A*B;

				for(int k = 0; k < t; k++){

					this->u[i*t+k] = C.m[k];
				}
			}

			}

		return *this;
		}

template<class T, int t> lfield<T,t>& lfield<T,t>::operator+= ( const lfield<T,t>& f ){

			if( this != &f ){

			#pragma omp parallel for simd default(shared)
			for(int i = 0; i < f.getNxl()*f.getNyl(); i++){
				for(int k = 0; k < t; k++){

					this->u[i*t+k] += f.u[i*t+k];
				}
			}

			}

		return *this;
		}

template<class T, int t> lfield<T,t> operator * ( const lfield<T,t> &f , const lfield<T,t> &g){

			lfield<T,t> result(f.getNxl(), f.getNyl());

			result.setToZero();

			if( f.getNxl() == g.getNxl() && f.getNyl() == g.getNyl() ){

			#pragma omp parallel for simd default(shared)
			for(int i = 0; i < f.getNxl()*f.getNyl(); i++){

				su3_matrix<double> A,B,C;
		
				for(int k = 0; k < t; k++){

					A.m[k] = f.u[i*t+k]; 
					B.m[k] = g.u[i*t+k];
				}
		
				C = A*B;

				for(int k = 0; k < t; k++){

					result.u[i*t+k] = C.m[k];
				}
			}

			}else{

				printf("Size of input objects in * operator is different!\n");

			}


		return result;
		}

template<class T, int t> gfield<T,t> operator * ( const gfield<T,t> &f , const gfield<T,t> &g){

			gfield<T,t> result(f.getNxg(), f.getNyg());

			result.setToZero();

			if( f.getNxg() == g.getNxg() && f.getNyg() == g.getNyg() ){

			#pragma omp parallel for simd default(shared)
			for(int i = 0; i < f.getNxg()*f.getNyg(); i++){
			
				su3_matrix<double> A,B,C;

				for(int k = 0; k < t; k++){

					A.m[k] = f.u[i*t+k];
					B.m[k] = g.u[i*t+k];
				}
		
				C = A*B;

				for(int k = 0; k < t; k++){

					result.u[i*t+k] = C.m[k];
				}
			}

			}else{

				printf("Size of input objects in * operator is different!\n");

			}


		return result;
		}



template<class T, int t> lfield<T,t> operator + ( const lfield<T,t> &f, const lfield<T,t>& g ){

			lfield<T,t> result(f.getNxl(), f.getNyl());

			result.setToZero();

			if( f.getNxl() == g.getNxl() && f.getNyl() == g.getNyl() ){

			#pragma omp parallel for simd default(shared)
			for(int i = 0; i < f.getNxl()*f.getNyl(); i ++){

				su3_matrix<double> A,B,C;
		
				for(int k = 0; k < t; k++){

					A.m[k] = f.u[i*t+k];
					B.m[k] = g.u[i*t+k];
				}
		
				C = A+B;

				for(int k = 0; k < t; k++){

					result.u[i*t+k] = C.m[k];
				}
			}

			}else{

				printf("Size of input objects in + operator is different!\n");

			}

		return result;
		}


template<class T, int t> gfield<T,t> operator + ( const gfield<T,t> &f, const gfield<T,t>& g ){

			gfield<T,t> result(f.getNxg(), f.getNyg());

			result.setToZero();

			if( f.getNxg() == g.getNxg() && f.getNyg() == g.getNyg() ){

			#pragma omp parallel for simd default(shared)
			for(int i = 0; i < f.getNxg()*f.getNyg(); i ++){
			
				su3_matrix<double> A,B,C;

				for(int k = 0; k < t; k++){

					A.m[k] = f.u[i*t+k];
					B.m[k] = g.u[i*t+k];
				}
		
				C = A+B;

				for(int k = 0; k < t; k++){

					result.u[i*t+k] = C.m[k];
				}
			}

			}else{

				printf("Size of input objects in + operator is different!\n");

			}

		return result;
		}



template<class T, int t> int gfield<T,t>::allgather(lfield<T,t>* ulocal, mpi_class* mpi){


	T* data_local_re = (T*)malloc(ulocal->getNxl()*ulocal->getNyl()*sizeof(T));
	T* data_local_im = (T*)malloc(ulocal->getNxl()*ulocal->getNyl()*sizeof(T));

	if(data_local_re == NULL || data_local_im == NULL){
		printf("allgather: malloc unsuccessful. Aborting.\n");
		exit(0);
	}

	T* data_global_re = (T*)malloc(Nxg*Nyg*sizeof(T));
	T* data_global_im = (T*)malloc(Nxg*Nyg*sizeof(T));

	if(data_global_re == NULL || data_global_im == NULL){
		printf("allgather: malloc unsuccessful. Aborting.\n");
		exit(0);
	}

	int i,k;

	for(k = 0; k < t; k++){

		#pragma omp parallel for simd default(shared)
		for(i = 0; i < ulocal->getNxl()*ulocal->getNyl(); i++){

			data_local_re[i] = ulocal->u[i*t+k].real();
			data_local_im[i] = ulocal->u[i*t+k].imag();

		}

   		MPI_Allgather(data_local_re, ulocal->getNxl()*ulocal->getNyl(), MPI_DOUBLE, data_global_re, ulocal->getNxl()*ulocal->getNyl(), MPI_DOUBLE, MPI_COMM_WORLD); 
	   	MPI_Allgather(data_local_im, ulocal->getNxl()*ulocal->getNyl(), MPI_DOUBLE, data_global_im, ulocal->getNxl()*ulocal->getNyl(), MPI_DOUBLE, MPI_COMM_WORLD); 

		int size = mpi->getSize();

		for(int kk = 0; kk < size; kk++){
		
			int local_volume = ulocal->getNxl() * ulocal->getNyl();

			#pragma omp parallel for simd collapse(2) default(shared)
			for(int xx = 0; xx < ulocal->getNxl(); xx++){
				for(int yy = 0; yy < ulocal->getNyl(); yy++){

					int i = xx*ulocal->getNyl() + yy;

					//will only work for parallelization in y direction
					//int ii = xx*Nyg + (yy + kk*ulocal->Nyl);

					//will only work for parallelization in x direction
					int ii = (xx + kk*ulocal->Nxl)*Nyg + yy;

					this->u[ii*t+k] = std::complex<double>(data_global_re[i+kk*local_volume], data_global_im[i+kk*local_volume]);

				}
			}
		}
	}

	free(data_local_re);
	free(data_local_im);

	free(data_global_re);
	free(data_global_im);

	return 1;
}

template<class T,int t> int lfield<T,t>::mpi_exchange_boundaries(mpi_class* mpi){

    double *bufor_send_n;
    double *bufor_receive_n;

    double *bufor_send_p;
    double *bufor_receive_p;


    if( mpi->getExchangeX() == 1 ){

	    int yy; 

	    bufor_send_n = (double*) malloc(Nyl_buf*sizeof(double));
	    bufor_receive_n = (double*) malloc(Nyl_buf*sizeof(double));

  	    if(bufor_receive_n == NULL || bufor_send_n == NULL){
		printf("mpi_exchange_boundaries: malloc unsuccessful. Aborting.\n");
		exit(0);
	    }

  	    for(yy = 0; yy < Nyl; yy++){
		bufor_send_n[yy] = this->u[buf_pos(Nxl-1,yy)].real();
	    }

	    printf("X data exchange: rank %i sending to %i\n", mpi->getRank(), mpi->getXNeighbourNext());
	    printf("X data exchange: rank %i receiving from %i\n", mpi->getRank(), mpi->getXNeighbourNext());

	    MPI_Send(bufor_send_n, Nyl_buf, MPI_DOUBLE, mpi->getXNeighbourNext(), 11, MPI_COMM_WORLD);
    	    MPI_Recv(bufor_receive_n, Nyl_buf, MPI_DOUBLE, mpi->getXNeighbourPrevious(), 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  	    for(yy = 0; yy < Nyl; yy++){
		this->u[buf_pos_ex(0,yy)] = bufor_receive_n[yy];
	    }

   	    bufor_send_p = (double*) malloc(Nyl_buf*sizeof(double));
	    bufor_receive_p = (double*) malloc(Nyl_buf*sizeof(double));

  	    if(bufor_receive_p == NULL || bufor_send_p == NULL){
		printf("mpi_exchange_boundaries: malloc unsuccessful. Aborting.\n");
		exit(0);
	    }

	    for(yy = 0; yy < Nyl; yy++){
		bufor_send_p[yy] = this->u[buf_pos(0,yy)].real();
	    }
	
 	    printf("X data exchange: rank %i sending to %i\n", mpi->getRank(), mpi->getXNeighbourPrevious());
	    printf("X data exchange: rank %i receiving to %i\n", mpi->getRank(), mpi->getXNeighbourPrevious());

	    MPI_Send(bufor_send_p, Nyl_buf, MPI_DOUBLE, mpi->getXNeighbourPrevious(), 12, MPI_COMM_WORLD);
    	    MPI_Recv(bufor_receive_p, Nyl_buf, MPI_DOUBLE, mpi->getXNeighbourNext(), 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  	    for(yy = 0; yy < Nyl; yy++){
		this->u[buf_pos_ex(Nxl+1,yy)] = bufor_receive_p[yy];
	    }
    }

    if( mpi->getExchangeY() == 1 ){

	    int xx; 

	    bufor_send_n = (double*) malloc(Nxl_buf*sizeof(double));
	    bufor_receive_n = (double*) malloc(Nxl_buf*sizeof(double));

  	    if(bufor_receive_n == NULL || bufor_send_n == NULL){
		printf("mpi_exchange_boundaries: malloc unsuccessful. Aborting.\n");
		exit(0);
	    }

  	    for(xx = 0; xx < Nxl; xx++){
		bufor_send_n[xx] = this->u[buf_pos(xx,Nyl-1)].real();
	    }

	    printf("Y data exchange: rank %i sending to %i\n", mpi->getRank(), mpi->getYNeighbourNext());
	    printf("Y data exchange: rank %i receiving from %i\n", mpi->getRank(), mpi->getYNeighbourNext());

	    MPI_Send(bufor_send_n, Nxl_buf, MPI_DOUBLE, mpi->getYNeighbourNext(), 13, MPI_COMM_WORLD);
    	    MPI_Recv(bufor_receive_n, Nxl_buf, MPI_DOUBLE, mpi->getYNeighbourPrevious(), 13, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for(xx = 0; xx < Nxl; xx++){
		this->u[buf_pos_ex(xx,0)] = bufor_receive_n[xx];
	    }

	    bufor_send_p = (double*) malloc(Nxl_buf*sizeof(double));
	    bufor_receive_p = (double*) malloc(Nxl_buf*sizeof(double));

  	    if(bufor_receive_p == NULL || bufor_send_p == NULL){
		printf("mpi_exchange_boundaries: malloc unsuccessful. Aborting.\n");
		exit(0);
	    }

	    for(xx = 0; xx < Nxl; xx++){
		bufor_send_p[xx] = this->u[buf_pos(xx,0)].real();
	    }
	
 	    printf("Y data exchange: rank %i sending to %i\n", mpi->getRank(), mpi->getYNeighbourPrevious());
	    printf("Y data exchange: rank %i receiving to %i\n", mpi->getRank(), mpi->getYNeighbourPrevious());

	    MPI_Send(bufor_send_p, Nxl_buf, MPI_DOUBLE, mpi->getYNeighbourPrevious(), 14, MPI_COMM_WORLD);
    	    MPI_Recv(bufor_receive_p, Nxl_buf, MPI_DOUBLE, mpi->getYNeighbourNext(), 14, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  	    for(xx = 0; xx < Nxl; xx++){
		this->u[buf_pos_ex(xx,Nyl+1)] = bufor_receive_p[xx];
	    }
    }

return 1;
}

template<class T, int t> int lfield<T,t>::setMVModel(MV_class* MVconfig){

	if(t == 9){

	const double EPS = 10e-12;

	#pragma omp parallel for simd default(shared)
	for(int i = 0; i < Nxl*Nyl; i++){

                static __thread std::ranlux24* generator = nullptr;
                if (!generator){
                         std::hash<std::thread::id> hasher;
                         generator = new std::ranlux24(clock() + hasher(std::this_thread::get_id()));
                }
                std::normal_distribution<double> distribution{0.0, MVconfig->gGet() * MVconfig->muGet() / sqrt(MVconfig->NyGet()) };


	    //set to zero
            for(int j = 0; j < t; j++)
                this->u[i*t+j] = 0.0;

	        double n[8];

		for(int k = 0; k < 8; k++){
	         	n[k] = distribution(*generator); 
		}

	//these are the LAMBDAs and not the generators t^a = lambda/2.

	// 0 1 2
	// 3 4 5
	// 6 7 8

		this->u[i*t+1] += std::complex<double>(n[0],0.0);
		this->u[i*t+3] += std::complex<double>(n[0],0.0);


       		this->u[i*t+1] += std::complex<double>(0.0,n[1]);
		this->u[i*t+3] -= std::complex<double>(0.0,n[1]);


       		this->u[i*t+0] += std::complex<double>(n[2],0.0);
		this->u[i*t+4] -= std::complex<double>(n[2],0.0);


       		this->u[i*t+2] += std::complex<double>(n[3],0.0);
		this->u[i*t+6] += std::complex<double>(n[3],0.0);


       		this->u[i*t+2] += std::complex<double>(0.0,n[4]);
		this->u[i*t+6] -= std::complex<double>(0.0,n[4]);


       		this->u[i*t+5] += std::complex<double>(n[5],0.0);
		this->u[i*t+7] += std::complex<double>(n[5],0.0);


       		this->u[i*t+5] += std::complex<double>(0.0,n[6]);
		this->u[i*t+7] -= std::complex<double>(0.0,n[6]);


      		this->u[i*t+0] += std::complex<double>(n[7]/sqrt(3.0),0.0);
		this->u[i*t+4] += std::complex<double>(n[7]/sqrt(3.0),0.0);
		this->u[i*t+8] += std::complex<double>(-2.0*n[7]/sqrt(3.0),0.0);
	}

	}else{

		printf("Invalid lfield classes for setMVModel function\n");

	}


return 1;
}

template<class T, int t> int lfield<T,t>::setUnitModel(rand_class* rr){

	if(t == 9){

	const double EPS = 10e-12;

	// 0 1 2
	// 3 4 5
	// 6 7 8

//	#pragma omp parallel for simd default(shared)
	for(int i = 0; i < Nxl*Nyl; i++){


		double n[8];

		for(int k = 0; k < 8; k++)
                	this->u[i*t+k] = sqrt( -2.0 * log( EPS + rr->get() ) ) * cos( rr->get() * 2.0 * M_PI);
		
	}

	}else{

		printf("Invalid lfield classes for setMVModel function\n");

	}


return 1;
}


template<class T, int t> int lfield<T,t>::setGaussian(mpi_class* mpi, config* cnfg){

	if(t == 9){

	const double EPS = 10e-12;

	// 0 1 2
	// 3 4 5
	// 6 7 8

        //rand_class* rr = new rand_class(mpi,cnfg);

	#pragma omp parallel for simd default(shared)
	for(int i = 0; i < Nxl*Nyl; i++){

		static __thread std::ranlux24* generator = nullptr;
	        if (!generator){
		  	 std::hash<std::thread::id> hasher;
			 generator = new std::ranlux24(clock() + hasher(std::this_thread::get_id()));
		}
    		std::normal_distribution<double> distribution{0.0,1.0};

	    	//set to zero
	    	for(int j = 0; j < t; j++)
			this->u[i*t+j] = 0.0;


	    	double n[8];

		for(int k = 0; k < 8; k++)
                	n[k] = distribution(*generator); 

		this->u[i*t+1] += std::complex<double>(n[0],0.0);
		this->u[i*t+3] += std::complex<double>(n[0],0.0);


		this->u[i*t+1] += std::complex<double>(0.0,n[1]);
		this->u[i*t+3] -= std::complex<double>(0.0,n[1]);


		this->u[i*t+0] += std::complex<double>(n[2],0.0);
		this->u[i*t+4] -= std::complex<double>(n[2],0.0);


       		this->u[i*t+2] += std::complex<double>(n[3],0.0);
		this->u[i*t+6] += std::complex<double>(n[3],0.0);


       		this->u[i*t+2] += std::complex<double>(0.0,n[4]);
		this->u[i*t+6] -= std::complex<double>(0.0,n[4]);


       		this->u[i*t+5] += std::complex<double>(n[5],0.0);
		this->u[i*t+7] += std::complex<double>(n[5],0.0);


       		this->u[i*t+5] += std::complex<double>(0.0,n[6]);
		this->u[i*t+7] -= std::complex<double>(0.0,n[6]);


       		this->u[i*t+0] += std::complex<double>(n[7]/sqrt(3.0),0.0);
		this->u[i*t+4] += std::complex<double>(n[7]/sqrt(3.0),0.0);
		this->u[i*t+8] += std::complex<double>(-2.0*n[7]/sqrt(3.0),0.0);
	}

	}else{

		printf("Invalid lfield classes for setGaussian function\n");

	}


return 1;
}


template<class T, int t> int lfield<T, t>::solvePoisson(double mass, double g, momenta* mom){

	#pragma omp parallel for simd default(shared)
	for(int i = 0; i < Nxl*Nyl; i++){
		for(int k = 0; k < t; k++){
			this->u[i*t+k] *= std::complex<double>(-1.0*g/(-mom->phat2(i) + mass*mass), 0.0);
		}
	}

return 1;
}

template<class T, int t > int lfield<T, t>::exponentiate(){

	#pragma omp parallel for simd default(shared)
	for(int i = 0; i < Nxl*Nyl; i++){
	
		su3_matrix<double> A;

		for(int k = 0; k < t; k++){

			A.m[k] = this->u[i*t+k];
		}
		
		A.exponentiate(1.0);

		for(int k = 0; k < t; k++){

			this->u[i*t+k] = A.m[k];
		}

	}

return 1;
}

template<class T, int t> int lfield<T, t>::exponentiate(double s){

	#pragma omp parallel for simd default(shared)
	for(int i = 0; i < Nxl*Nyl; i++){

		su3_matrix<double> A;

		for(int k = 0; k < t; k++){

			A.m[k] = s*(this->u[i*t+k]);
		}
		
		A.exponentiate(-1.0);

		for(int k = 0; k < t; k++){

			this->u[i*t+k] = A.m[k];
		}

	}

return 1;
}


template<class T, int t> int lfield<T,t>::setKernelPbarX(momenta* mom, mpi_class* mpi, Kernel KernelChoice){

	if(t == 9){

	#pragma omp parallel for simd collapse(2) default(shared)
	for(int ix = 0; ix < Nxl; ix++){
	for(int iy = 0; iy < Nyl; iy++){
	//for(int i = 0; i < Nxl*Nyl; i++){

		int i = ix*Nyl + iy;

                int xx = ix + mpi->getPosX()*(Nxl);
                int yy = iy + mpi->getPosY()*(Nyl);

                if( xx >= Nx/2 )
                        xx = xx - Nx;

                if( yy >= Ny/2 )
                        yy = yy - Ny;

                double px = 2.0 * M_PI * xx / (1.0 * Nx);
                double py = 2.0 * M_PI * yy / (1.0 * Ny);

                if( fabs(px*px+py*py) > 10e-9 ){

      	        	if( KernelChoice == LINEAR_KERNEL ){

				this->u[i*t+0] = std::complex<double>(0.0, -2.0*M_PI*px/(px*px+py*py));
				this->u[i*t+4] = this->u[i*t+0];
				this->u[i*t+8] = this->u[i*t+0];

			}
                        if( KernelChoice == SIN_KERNEL ){

				this->u[i*t+0] = std::complex<double>(0.0, -2.0*M_PI*mom->pbarX(i)/mom->phat2(i));
				this->u[i*t+4] = this->u[i*t+0];
				this->u[i*t+8] = this->u[i*t+0];
                        }
		}
	}}

	}else{

		printf("Invalid lfield classes for setKernelPbarX function\n");

	}



return 1;
}

template<class T, int t> int lfield<T,t>::setKernelPbarY(momenta* mom, mpi_class* mpi, Kernel KernelChoice){


	if(t == 9){

	#pragma omp parallel for simd collapse(2) default(shared)
        for(int ix = 0; ix < Nxl; ix++){
        for(int iy = 0; iy < Nyl; iy++){
	//for(int i = 0; i < Nxl*Nyl; i++){

                int i = ix*Nyl + iy;

                int xx = ix + mpi->getPosX()*(Nxl);
                int yy = iy + mpi->getPosY()*(Nyl);

                if( xx >= Nx/2 )
                        xx = xx - Nx;

                if( yy >= Ny/2 )
                        yy = yy - Ny;

                double px = 2.0 * M_PI * xx / (1.0 * Nx);
                double py = 2.0 * M_PI * yy / (1.0 * Ny);

                if( fabs(px*px+py*py) > 10e-9 ){

                        if( KernelChoice == LINEAR_KERNEL ){

                                this->u[i*t+0] = std::complex<double>(0.0, -2.0*M_PI*py/(px*px+py*py));
                                this->u[i*t+4] = this->u[i*t+0];
                                this->u[i*t+8] = this->u[i*t+0];

                        }
                        if( KernelChoice == SIN_KERNEL ){

                                this->u[i*t+0] = std::complex<double>(0.0, -2.0*M_PI*mom->pbarY(i)/mom->phat2(i));
                                this->u[i*t+4] = this->u[i*t+0];
                                this->u[i*t+8] = this->u[i*t+0];
                        }
		}
        }}


	}else{

		printf("Invalid lfield classes for setKernelPbarY function\n");

	}



return 1;
}


template<class T, int t> int lfield<T,t>::setKernelPbarXWithCouplingConstant(momenta* mom, mpi_class* mpi, Kernel KernelChoice){

	if(t == 9){

	#pragma omp parallel for simd collapse(2) default(shared)
        for(int ix = 0; ix < Nxl; ix++){
        for(int iy = 0; iy < Nyl; iy++){
	//for(int i = 0; i < Nxl*Nyl; i++){

                int i = ix*Nyl + iy;

                int xx = ix + mpi->getPosX()*(Nxl);
                int yy = iy + mpi->getPosY()*(Nyl);

                if( xx >= Nx/2 )
                        xx = xx - Nx;

                if( yy >= Ny/2 )
                        yy = yy - Ny;

                double px = 2.0 * M_PI * xx / (1.0 * Nx);
                double py = 2.0 * M_PI * yy / (1.0 * Ny);

		double coupling_constant = 4.0*M_PI/( (11.0-2.0*3.0/3.0)*log( pow( pow(15.0*15.0/6.0/6.0,1.0/0.2) + pow((mom->phat2(i)*Nx*Ny)/6.0/6.0,1.0/0.2) , 0.2) ) );

		if( fabs(px*px+py*py) > 10e-9 ){

                        if( KernelChoice == LINEAR_KERNEL ){

                                this->u[i*t+0] = std::complex<double>(0.0, -2.0*M_PI*sqrt(coupling_constant)*px/(px*px+py*py));
                                this->u[i*t+4] = this->u[i*t+0];
                                this->u[i*t+8] = this->u[i*t+0];

                        }
                        if( KernelChoice == SIN_KERNEL ){

                                this->u[i*t+0] = std::complex<double>(0.0, -2.0*M_PI*sqrt(coupling_constant)*mom->pbarX(i)/mom->phat2(i));
                                this->u[i*t+4] = this->u[i*t+0];
                                this->u[i*t+8] = this->u[i*t+0];
                        }
		}
        }}

	}else{

		printf("Invalid lfield classes for setKernelPbarX function\n");

	}



return 1;
}

template<class T, int t> int lfield<T,t>::setKernelPbarYWithCouplingConstant(momenta* mom, mpi_class* mpi, Kernel KernelChoice){


	if(t == 9){

	#pragma omp parallel for simd default(shared)
        for(int ix = 0; ix < Nxl; ix++){
        for(int iy = 0; iy < Nyl; iy++){
	//for(int i = 0; i < Nxl*Nyl; i++){

                int i = ix*Nyl + iy;

                int xx = ix + mpi->getPosX()*(Nxl);
                int yy = iy + mpi->getPosY()*(Nyl);

                if( xx >= Nx/2 )
                        xx = xx - Nx;

                if( yy >= Ny/2 )
                        yy = yy - Ny;

                double px = 2.0 * M_PI * xx / (1.0 * Nx);
                double py = 2.0 * M_PI * yy / (1.0 * Ny);

		double coupling_constant = 4.0*M_PI/( (11.0-2.0*3.0/3.0)*log( pow( pow(15.0*15.0/6.0/6.0,1.0/0.2) + pow((mom->phat2(i)*Nx*Ny)/6.0/6.0,1.0/0.2) , 0.2) ) );

		if( fabs(px*px+py*py) > 10e-9 ){

                        if( KernelChoice == LINEAR_KERNEL ){

                                this->u[i*t+0] = std::complex<double>(0.0, -2.0*M_PI*sqrt(coupling_constant)*py/(px*px+py*py));
                                this->u[i*t+4] = this->u[i*t+0];
                                this->u[i*t+8] = this->u[i*t+0];

                        }
                        if( KernelChoice == SIN_KERNEL ){

                                this->u[i*t+0] = std::complex<double>(0.0, -2.0*M_PI*sqrt(coupling_constant)*mom->pbarY(i)/mom->phat2(i));
                                this->u[i*t+4] = this->u[i*t+0];
                                this->u[i*t+8] = this->u[i*t+0];
                        }
		}
        }}

	}else{

		printf("Invalid lfield classes for setKernelPbarY function\n");

	}



return 1;
}

template<class T, int t> int gfield<T,t>::setKernelXbarX(int x_global, int y_global, positions* pos, Kernel KernelChoice){

	if(t == 9){

	#pragma omp parallel for simd collapse(2) default(shared)
	for(int xx = 0; xx < Nxg; xx++){
		for(int yy = 0; yy < Nyg; yy++){

			int i = xx*Nyg+yy;
		
			double dx, dy, rrr;

			if( KernelChoice == LINEAR_KERNEL ){

				dx = x_global - xx;
                                if( dx >= Nxg/2 )
                                	dx = dx - Nxg;
                                if( dx < -Nxg/2 )
                                	dx = dx + Nxg;
                                         
                                dy = y_global - yy;
                                if( dy >= Nyg/2 )
                                       	dy = dy - Nyg;
                                if( dy < -Nyg/2 )
                                        dy = dy + Nyg;

                                rrr = 1.0*(dx*dx+dy*dy);
			}
			if( KernelChoice == SIN_KERNEL ){

                                int ii = 0;
                                if( x_global >= xx)
                                        ii += (x_global - xx)*Ny;
                                else
                                        ii += (x_global - xx + Nx)*Ny;

                                if( y_global >= yy)
                                        ii += (y_global - yy);
                                else
                                        ii += (y_global - yy + Ny);

                                dx = pos->xhatX(ii);
                                dy = pos->xhatY(ii);

                                rrr = pos->xbar2(ii);

			}

			if( rrr > 10e-9 ){

				this->u[i*t+0] = std::complex<double>(dx/rrr, 0.0);
				this->u[i*t+4] = this->u[i*t+0];
				this->u[i*t+8] = this->u[i*t+0];

			}
		}
	}

	}else{

		printf("Invalid lfield classes for setKernelPbarX function\n");

	}

return 1;
}

template<class T, int t> int gfield<T,t>::setKernelXbarY(int x_global, int y_global, positions* pos, Kernel KernelChoice){

	if(t == 9){

	#pragma omp parallel for simd collapse(2) default(shared)
	for(int xx = 0; xx < Nxg; xx++){
		for(int yy = 0; yy < Nxg; yy++){

			int i = xx*Nyg+yy;

                        double dx, dy, rrr;

                        if( KernelChoice == LINEAR_KERNEL ){

                                dx = x_global - xx;
                                if( dx >= Nxg/2 )
                                        dx = dx - Nxg;
                                if( dx < -Nxg/2 )
                                        dx = dx + Nxg;

                                dy = y_global - yy;
                                if( dy >= Nyg/2 ) 
                                        dy = dy - Nyg;
                                if( dy < -Nyg/2 )
                                        dy = dy + Nyg;

                                rrr = 1.0*(dx*dx+dy*dy);
                        }
                        if( KernelChoice == SIN_KERNEL ){

                                int ii = 0;
                                if( x_global >= xx)
                                        ii += (x_global - xx)*Ny;
                                else
                                        ii += (x_global - xx + Nx)*Ny;

                                if( y_global >= yy)
                                        ii += (y_global - yy);
                                else
                                        ii += (y_global - yy + Ny);

                                dx = pos->xhatX(ii);
                                dy = pos->xhatY(ii);

                                rrr = pos->xbar2(ii);
                        }

			if( rrr > 10e-9 ){

				this->u[i*t+0] = std::complex<double>(dy/rrr, 0.0);
				this->u[i*t+4] = this->u[i*t+0];
				this->u[i*t+8] = this->u[i*t+0];

			}
		}
	}

	}else{

		printf("Invalid lfield classes for setKernelPbarY function\n");

	}



return 1;
}


template<class T, int t> int gfield<T,t>::setKernelXbarXWithCouplingConstant(int x_global, int y_global, positions* pos, Kernel KernelChoice){

	if(t == 9){

	//for(int i = 0; i < Nxl*Nyl; i++){
	#pragma omp parallel for simd collapse(2) default(shared)
	for(int xx = 0; xx < Nxg; xx++){
		for(int yy = 0; yy < Nyg; yy++){

			int i = xx*Nyg+yy;

                        double dx, dy, rrr;

                        if( KernelChoice == LINEAR_KERNEL ){

                                dx = x_global - xx;
                                if( dx >= Nxg/2 )
                                        dx = dx - Nxg;
                                if( dx < -Nxg/2 )
                                        dx = dx + Nxg;

                                dy = y_global - yy;
                                if( dy >= Nyg/2 ) 
                                        dy = dy - Nyg;
                                if( dy < -Nyg/2 )
                                        dy = dy + Nyg;

                                rrr = 1.0*(dx*dx+dy*dy);
                        }
                        if( KernelChoice == SIN_KERNEL ){

                                int ii = 0;
                                if( x_global >= xx)
                                        ii += (x_global - xx)*Ny;
                                else
                                        ii += (x_global - xx + Nx)*Ny;

                                if( y_global >= yy)
                                        ii += (y_global - yy);
                                else
                                        ii += (y_global - yy + Ny);

                                dx = pos->xhatX(ii);
                                dy = pos->xhatY(ii);

                                rrr = pos->xbar2(ii);

                        }

			double coupling_constant = 4.0*M_PI/(  (11.0-2.0*3.0/3.0) * log( pow( pow(15.0*15.0/6.0/6.0,1.0/0.2) + 1.26/pow(6.0*6.0*(dx*dx+dy*dy)/Nxg/Nyg,1.0/0.2) , 0.2 ) ));

			if( rrr > 10e-9 ){

				this->u[i*t+0] = std::complex<double>(sqrt(coupling_constant)*dx/rrr, 0.0);
				this->u[i*t+4] = this->u[i*t+0];
				this->u[i*t+8] = this->u[i*t+0];

			}else{
	
				this->u[i*t+0] = std::complex<double>(0.0, 0.0);
				this->u[i*t+4] = this->u[i*t+0];
				this->u[i*t+8] = this->u[i*t+0];
	
			}
		}
	}

	}else{

		printf("Invalid lfield classes for setKernelPbarX function\n");

	}

return 1;
}

template<class T, int t> int gfield<T,t>::setKernelXbarYWithCouplingConstant(int x_global, int y_global, positions* pos, Kernel KernelChoice){

	if(t == 9){

	#pragma omp parallel for simd collapse(2) default(shared)
	for(int xx = 0; xx < Nxg; xx++){
		for(int yy = 0; yy < Nxg; yy++){

			int i = xx*Nyg+yy;

                        double dx, dy, rrr;

                        if( KernelChoice == LINEAR_KERNEL ){

                                dx = x_global - xx;
                                if( dx >= Nxg/2 )
                                        dx = dx - Nxg;
                                if( dx < -Nxg/2 )
                                        dx = dx + Nxg;

                                dy = y_global - yy;
                                if( dy >= Nyg/2 ) 
                                        dy = dy - Nyg;
                                if( dy < -Nyg/2 )
                                        dy = dy + Nyg;

                                rrr = 1.0*(dx*dx+dy*dy);
                        }
                        if( KernelChoice == SIN_KERNEL ){

                                int ii = 0;
                                if( x_global >= xx)
                                        ii += (x_global - xx)*Ny;
                                else
                                        ii += (x_global - xx + Nx)*Ny;

                                if( y_global >= yy)
                                        ii += (y_global - yy);
                                else
                                        ii += (y_global - yy + Ny);

                                dx = pos->xhatX(ii);
                                dy = pos->xhatY(ii);

                                rrr = pos->xbar2(ii);

                        }

			double coupling_constant = 4.0*M_PI/(  (11.0-2.0*3.0/3.0) * log( pow( pow(15.0*15.0/6.0/6.0,1.0/0.2) + 1.26/pow(6.0*6.0*(dx*dx+dy*dy)/Nxg/Nyg,1.0/0.2) , 0.2 ) ));

			if( rrr > 10e-9 ){

				this->u[i*t+0] = std::complex<double>(sqrt(coupling_constant)*dy/rrr, 0.0);
				this->u[i*t+4] = this->u[i*t+0];
				this->u[i*t+8] = this->u[i*t+0];

			}else{

				this->u[i*t+0] = std::complex<double>(0.0, 0.0);
				this->u[i*t+4] = this->u[i*t+0];
				this->u[i*t+8] = this->u[i*t+0];

			}
		}
	}

	}else{

		printf("Invalid lfield classes for setKernelPbarY function\n");

	}



return 1;
}


template<class T, int t> lfield<T,t>* lfield<T,t>::hermitian(void){

	lfield<T,t>* result = new lfield<T,t>(Nxl, Nyl);

	if(t == 9){

	// 0 1 2
	// 3 4 5
	// 6 7 8

	#pragma omp parallel for simd default(shared)
	for(int i = 0; i < Nxl*Nyl; i++){

		result->u[i*t+0] = std::conj(this->u[i*t+0]);
		result->u[i*t+1] = std::conj(this->u[i*t+3]);
		result->u[i*t+2] = std::conj(this->u[i*t+6]);
		result->u[i*t+3] = std::conj(this->u[i*t+1]);
		result->u[i*t+4] = std::conj(this->u[i*t+4]);
		result->u[i*t+5] = std::conj(this->u[i*t+7]);
		result->u[i*t+6] = std::conj(this->u[i*t+2]);
		result->u[i*t+7] = std::conj(this->u[i*t+5]);
		result->u[i*t+8] = std::conj(this->u[i*t+8]);

	}

	}else{

		printf("Invalid lfield classes for hermitian function\n");

	}

return result;
}

template<class T, int t> gfield<T,t>* gfield<T,t>::hermitian(void){

	gfield<T,t>* result = new gfield<T,t>(Nx, Ny);

	if(t == 9){

	// 0 1 2
	// 3 4 5
	// 6 7 8

	#pragma omp parallel for simd default(shared)
	for(int i = 0; i < Nx*Ny; i++){

		result->u[i*t+0] = std::conj(this->u[i*t+0]);
		result->u[i*t+1] = std::conj(this->u[i*t+3]);
		result->u[i*t+2] = std::conj(this->u[i*t+6]);
		result->u[i*t+3] = std::conj(this->u[i*t+1]);
		result->u[i*t+4] = std::conj(this->u[i*t+4]);
		result->u[i*t+5] = std::conj(this->u[i*t+7]);
		result->u[i*t+6] = std::conj(this->u[i*t+2]);
		result->u[i*t+7] = std::conj(this->u[i*t+5]);
		result->u[i*t+8] = std::conj(this->u[i*t+8]);

	}

	}else{

		printf("Invalid lfield classes for hermitian function\n");

	}

return result;
}

//template<typename T>

template<class T, int t> int lfield<T,t>::trace(lfield<double,1>* cc){

//template<class T, int t> int lfield<T,t>::trace(template<class TT, int tt> lfield<TT,tt>* cc){

	// 0 1 2
	// 3 4 5
	// 6 7 8

	if(t == 9 ){

	#pragma omp parallel for simd default(shared)
	for(int i = 0; i < Nxl*Nyl; i++){

                su3_matrix<double> A;
                su3_matrix<double> B;
                su3_matrix<double> C;

                for(int k = 0; k < t; k++){
                        A.m[k] = this->u[i*t+k]/(1.0*Nx*Ny);
                }

		B.m[0] = std::conj(this->u[i*t+0]/(1.0*Nx*Ny));
		B.m[1] = std::conj(this->u[i*t+3]/(1.0*Nx*Ny));
		B.m[2] = std::conj(this->u[i*t+6]/(1.0*Nx*Ny));
		B.m[3] = std::conj(this->u[i*t+1]/(1.0*Nx*Ny));
		B.m[4] = std::conj(this->u[i*t+4]/(1.0*Nx*Ny));
		B.m[5] = std::conj(this->u[i*t+7]/(1.0*Nx*Ny));
		B.m[6] = std::conj(this->u[i*t+2]/(1.0*Nx*Ny));
		B.m[7] = std::conj(this->u[i*t+5]/(1.0*Nx*Ny));
		B.m[8] = std::conj(this->u[i*t+8]/(1.0*Nx*Ny));

                C = A*B;

                cc->u[i*1+0] = C.m[0] + C.m[4] + C.m[8];
	}

	}else{

		printf("Invalid lfield classes for trace function\n");

	}


return 1;
}

template<class T, int t> int lfield<T,t>::average(lfield<double,1>* cc){

	if(t == 9 ){

	#pragma omp parallel for simd default(shared)
	for(int i = 0; i < Nxl*Nyl; i++){

//                for(int k = 0; k < t; k++){
//                         cc->u[0][i] += this->u[k][i];
//                }
                         cc->u[i*1+0] = this->u[i*t+0];

	}

	}else{

		printf("Invalid lfield classes for trace function\n");

	}


return 1;
}



template<class T, int t> int gfield<T,t>::average_and_symmetrize(void){

	printf("avegare_and_symmetrize: t = %i\n", t);

	gfield<T,t>* corr_tmp = new gfield(Nx,Ny);

	//corr_tmp->setToZero();
	
	#pragma omp parallel for simd collapse(2) default(shared)
	for(int i = 0; i < Nx; i++){
		for(int j = 0; j < Ny; j++){
			corr_tmp->u[(i*Ny+j)*t+0] = 0.5*(this->u[(i*Ny+j)*t+0] + this->u[(j*Ny+i)*t+0]);
		}
	}

	#pragma omp parallel for simd collapse(2) default(shared)
	for(int i = 0; i < Nx; i++){
		for(int j = 0; j < Ny; j++){
			
			int a = abs(i-Nx)%Nx;
			int b = abs(j-Ny)%Ny;

			this->u[(i*Ny+j)*t+0] = 0.25*(	corr_tmp->u[(i*Ny+j)*t+0] +
							corr_tmp->u[(a*Ny+j)*t+0] + 
							corr_tmp->u[(i*Ny+b)*t+0] + 
							corr_tmp->u[(a*Ny+b)*t+0]);
		}
	}

	delete corr_tmp;

return 1;
}


template<class T, int t> lfield<T,t>* gfield<T,t>::reduce(int NNx, int NNy, mpi_class* mpi){

	lfield<T,t>* corr_tmp = new lfield<T,t>(NNx,NNy);

	corr_tmp->setToZero();	

	#pragma omp parallel for simd collapse(2) default(shared)
	for(int i = 0; i < NNx; i++){
		for(int j = 0; j < NNy; j++){
			//x += this->u[(i+mpi->getPosX()*NNx)*Nyg+j+mpi->getPosY()*NNy][0];
			corr_tmp->u[(i*NNy+j)*t+0] = this->u[((i+mpi->getPosX()*NNx)*Ny+j+mpi->getPosY()*NNy)*t+0];
		}
	}

return corr_tmp;
}


template<class T, int t> int gfield<T,t>::reduce(lfield<T,t>* sum, lfield<T,t>* err, mpi_class* mpi){

	int NNx = sum->getNxl();
	int NNy = sum->getNyl();

	#pragma omp parallel for simd collapse(2) default(shared)
	for(int i = 0; i < NNx; i++){
		for(int j = 0; j < NNy; j++){
			//x += this->u[(i+mpi->getPosX()*NNx)*Nyg+j+mpi->getPosY()*NNy][0];
			sum->u[(i*NNy+j)*t+0] += this->u[((i+mpi->getPosX()*NNx)*Ny+j+mpi->getPosY()*NNy)*t+0];
			err->u[(i*NNy+j)*t+0] += pow(this->u[((i+mpi->getPosX()*NNx)*Ny+j+mpi->getPosY()*NNy)*t+0],2.0);
		}
	}

return 1;
}

template<class T, int t> int gfield<T,t>::reduce_hatta(lfield<T,t>* sum, lfield<T,t>* err, mpi_class* mpi, int xr, int yr ){

	int NNx = sum->getNxl();
	int NNy = sum->getNyl();

	int xr_local = xr%(sum->getNxl());
	int yr_local = yr%(sum->getNyl());

//check MPI parallelization!!!

//	#pragma omp parallel for simd collapse(2) default(shared)
//	for(int i = 0; i < NNx; i++){
//		for(int j = 0; j < NNy; j++){

	//!!! we should only select the momenta which correspond to the distance rr !!!

//we only set it on the rank which contains this part of the global lattice
if( mpi->getPosX() == xr/(sum->getNxl()) && mpi->getPosY() == yr/(sum->getNyl()) ){

	sum->u[(xr_local*NNy+yr_local)*t+0] += this->u[(xr*Ny+yr)*t+0];
	err->u[(xr_local*NNy+yr_local)*t+0] += pow(this->u[(xr*Ny+yr)*t+0],2.0);

}
	

//		}
//	}

return 1;
}





template<class T, int t> int lfield<T,t>::reduceAndSet(int x_local, int y_local, gfield<T,t>* f){

		for(int k = 0; k < t; k++){

		double sum_re = 0;
		double sum_im = 0;

		#pragma omp parallel for collapse(2) default(shared) reduction(+:sum_re) reduction(+:sum_im)
		for(int xx = 0; xx < f->getNxg(); xx++){
			for(int yy = 0; yy < f->getNyg(); yy++){

				sum_re += f->u[(xx*f->getNyg()+yy)*t+k].real();		
				sum_im += f->u[(xx*f->getNyg()+yy)*t+k].imag();		

			}
		}

		this->u[(x_local*Nyl+y_local)*t+k] = std::complex<double>(sum_re, sum_im);

		}


return 1;
}

template<class T, int t> int gfield<T,t>::setCorrelationsForCouplingConstant(){

	const double w = pow(15.0*15.0/6.0/6.0,1.0/0.2);
	const double f = 4.0*M_PI/ (11.0-2.0*3.0/3.0);

	#pragma omp parallel for simd collapse(2) default(shared)
	for(int xx = 0; xx < Nxg; xx++){
		for(int yy = 0; yy < Nxg; yy++){

			int i = xx*Nyg+yy;

			double sqrt_coupling_constant = f / log( pow( w + 1.26/(1.0e-12+pow(6.0*6.0*(xx*xx+yy*yy)/Nxg/Nyg,1.0/0.2)) , 0.2 ) );
        
			this->u[i*t+0] = sqrt_coupling_constant;
		}
	}

return 1;
}

template<class T, int t> int lfield<T,t>::setCorrelationsForCouplingConstant(momenta* mom){

	const double w = pow(15.0*15.0/6.0/6.0,1.0/0.2);
	const double f = 4.0*M_PI/ (11.0-2.0*3.0/3.0);

	//#pragma omp parallel for simd collapse(2) default(shared)
	for(int xx = 0; xx < Nxl; xx++){
		for(int yy = 0; yy < Nyl; yy++){

			int i = xx*Nyl+yy;

			double sqrt_coupling_constant = f / log( pow(w + pow((mom->phat2(i)*Nx*Ny)/6.0/6.0,1.0/0.2) , 0.2) );
       
			this->u[i*t+0] = sqrt_coupling_constant/sqrt(Nx*Ny);
		}
	}

return 1;
}

template<class T, int t> int gfield<T,t>::multiplyByCholesky(gmatrix<T>* mm){

	gfield<T,t>* tmp = new gfield<T,t>(Nxg, Nyg); 

	for(int tt = 0; tt < t; tt++){
	
		for(int xx = 0; xx < Nxg; xx++){
			for(int yy = 0; yy < Nyg; yy++){
	
				tmp->u[(xx*Nyg+yy)*t+tt] = 0;

				for(int xxi = 0; xxi < Nxg; xxi++){
					for(int yyi = 0; yyi < Nyg; yyi++){
	
						tmp->u[(xx*Nyg+yy)*t+tt] += mm->u[(xx*Nyg+yy)*Nxg*Nyg + xxi*Nyg+yyi] * this->u[(xxi*Nyg+yyi)*t+tt];

					}
				}
			}
		}

		for(int xx = 0; xx < Nxg; xx++){
			for(int yy = 0; yy < Nyg; yy++){
	
				this->u[(xx*Nyg+yy)*t+tt] = tmp->u[(xx*Nyg+yy)*t+tt];

			}
		}	
	}

	delete tmp;

return 1;
} 

template<class T, int t> int lfield<T,t>::print(momenta* mom){

	for(int k = 0; k < t; k++){

	for(int xx = 0; xx < Nxl; xx++){
		for(int yy = 0; yy < Nxl; yy++){

			int i = xx*Nyl+yy;
			printf("%i %i %f %f\n", xx+1, yy+1, this->u[i*t+k].real(), this->u[i*t+k].imag());

		}
	}

	}
	
return 1;
}

template<class T, int t> int lfield<T,t>::print(momenta* mom, double x, mpi_class* mpi){


	for(int k = 0; k < mpi->getSize(); k++){

		if( mpi->getRank() == k ){

			for(int xx = 0; xx < Nxl; xx++){
				for(int yy = 0; yy < Nyl; yy++){

					int i = xx*Nyl+yy;
      
					if( fabs(xx + mpi->getPosX()*Nxl - yy - mpi->getPosY()*Nyl) <= 4 ){
 
						printf("%i %i %i %i %f %e %e\n", xx, mpi->getPosX(), yy, mpi->getPosY(), sqrt(mom->phat2(i)), x*(mom->phat2(i))*(this->u[i*t+0].real()), x*(mom->phat2(i))*(this->u[i*t+0].imag()));

					}
				}
			}

		}else{

			printf("###\n");
		}

		MPI_Barrier(MPI_COMM_WORLD);

	}

return 1;
}

template<class T, int t> int print(lfield<T,t>* sum, lfield<T,t>* err, momenta* mom, double x, mpi_class* mpi){


        printf("### object size = %i, %i\n", sum->getNxl(), sum->getNyl());

        for(int k = 0; k < mpi->getSize(); k++){

                if( mpi->getRank() == k ){

                        for(int xx = 0; xx < sum->getNxl(); xx++){
                                for(int yy = 0; yy < sum->getNyl(); yy++){

                                        int i = xx*(sum->getNyl())+yy;

                                        if( fabs(xx + mpi->getPosX()*(sum->getNxl()) - yy - mpi->getPosY()*(sum->getNyl())) <= 4 ){

                                                printf("%i %i %i %i %f %e %e\n", xx, mpi->getPosX(), yy, mpi->getPosY(), sqrt(mom->phat2(i)), x*(mom->phat2(i))*(sum->u[i*t+0].real()), x*(mom->phat2(i))*(err->u[i*t+0].real()));

                                        }
                                }
                        }

                }else{

                        printf("###\n");
                }

                MPI_Barrier(MPI_COMM_WORLD);

        }

return 1;
}

template<class T, int t> int lfield<T,t>::printDebug(){

	for(int xx = 0; xx < Nxl; xx++){
		for(int yy = 0; yy < Nxl; yy++){

			int i = xx*Nyl+yy;

			for(int k = 0; k < t; k++){
       
				printf("%i %i %f %f\n",xx, yy, this->u[i*t+k].real(), this->u[i*t+k].imag());
			}
		}
	}

return 1;
}

template<class T, int t> int gfield<T,t>::printDebug(){

	for(int xx = 0; xx < Nxg; xx++){
		for(int yy = 0; yy < Nxg; yy++){

			int i = xx*Nyg+yy;

			for(int k = 0; k < t; k++){
       
				printf("%i %i %f %f\n",xx, yy, this->u[i*t+k].real(), this->u[i*t+k].imag());
			}
		}
	}

return 1;
}

template<class T, int t> int lfield<T,t>::printDebug(double x){


	for(int xx = 0; xx < Nxl; xx++){
		for(int yy = 0; yy < Nxl; yy++){

			int i = xx*Nyl+yy;

			for(int k = 0; k < t; k++){

				double a = this->u[i*t+k].real();
				double b = this->u[i*t+k].imag();

//				if( a*a + b*b > 0.01 )      
				printf("%i %i %f %f\n",xx, yy, x*a, x*b);
			}
		}
	}

return 1;
}


template<class T, int t> int lfield<T,t>::printDebug(int ii){

	int xxx = ii/Nyl;
	int yyy = ii-xxx*Nyl;

	for(int xx = 0; xx < Nxl; xx++){
		for(int yy = 0; yy < Nxl; yy++){

			int i = xx*Nyl+yy;

			for(int k = 0; k < t; k++){

				int s1 = k/9;
				int s2 = k-s1*9;

				double a = this->u[i*t+k].real();
				double b = this->u[i*t+k].imag();

				if( a*a + b*b > 0.01 )      
					printf("%i %i %i %i %i %i %f %f\n",xxx, yyy, xx, yy, s1, s2, a, b);
			}
		}
	}

return 1;
}


template<class T, int t> int lfield<T,t>::printDebug(double x, mpi_class* mpi){

	for(int xx = 0; xx < Nxl; xx++){
		for(int yy = 0; yy < Nyl; yy++){

			int i = xx*Nyl+yy;
       
			printf("%i %i %f %f\n",xx+mpi->getPosX()*Nxl, yy+mpi->getPosY()*Nyl, x*(this->u[i*t+0].real()), x*(this->u[i*t+0].imag()));
		}
	}

return 1;
}

template<class T, int t> int lfield<T,t>::printDebugRadial(double x){

	for(int xx = 0; xx < Nxl; xx++){
		for(int yy = 0; yy < Nxl; yy++){

			int i = xx*Nyl+yy;

 			double z; 
			if( xx < Nxl/2 && yy < Nyl/2 ){
				z = x/pow(((xx)%Nxl)*((xx)%Nxl) + ((yy)%Nyl)*((yy)%Nyl),2.0)/3.0;
			}
			if( xx > Nxl/2 && yy < Nyl/2 ){
				z = x/pow(((xx-Nxl)%Nxl)*((xx-Nxl)%Nxl) + ((yy)%Nyl)*((yy)%Nyl),2.0)/3.0;
			}
			if( xx < Nxl/2 && yy > Nyl/2 ){
				z = x/pow(((xx)%Nxl)*((xx)%Nxl) + ((yy-Nyl)%Nyl)*((yy-Nyl)%Nyl),2.0)/3.0;
			}
			if( xx > Nxl/2 && yy > Nyl/2 ){
				z = x/pow(((xx-Nxl)%Nxl)*((xx-Nxl)%Nxl) + ((yy-Nyl)%Nyl)*((yy-Nyl)%Nyl),2.0)/3.0;
			}

			printf("%i %i %f %f\n",xx, yy, z*(this->u[i*t+0].real()), z*(this->u[i*t+0].imag()));
		}
	}

return 1;
}


template<class T, int t> int uxiulocal(lfield<T,t>* uxiulocal_x, lfield<T,t>* uxiulocal_y, lfield<T,t>* uf, lfield<T,t>* xi_local_x, lfield<T,t>* xi_local_y){

///                uf_hermitian = uf.hermitian();
///
///                uxiulocal_x = uf * xi_local_x * (*uf_hermitian);
///                uxiulocal_y = uf * xi_local_y * (*uf_hermitian);
///
///                delete uf_hermitian;

        #pragma omp parallel for simd default(shared)
        for(int i = 0; i < uf->getNxl()*uf->getNyl(); i++){

	        su3_matrix<double> A,B,C,D,E,F;

                D.m[0] = std::conj(uf->u[i*t+0]);
                D.m[1] = std::conj(uf->u[i*t+3]);
                D.m[2] = std::conj(uf->u[i*t+6]);
                D.m[3] = std::conj(uf->u[i*t+1]);
                D.m[4] = std::conj(uf->u[i*t+4]);
                D.m[5] = std::conj(uf->u[i*t+7]);
                D.m[6] = std::conj(uf->u[i*t+2]);
                D.m[7] = std::conj(uf->u[i*t+5]);
                D.m[8] = std::conj(uf->u[i*t+8]);

               for(int k = 0; k < t; k++){
			A.m[k] = uf->u[i*t+k];
	                B.m[k] = xi_local_x->u[i*t+k]; ///(1.0*Nx*Ny);
        	        C.m[k] = xi_local_y->u[i*t+k]; ///(1.0*Nx*Ny);
		}

		E = A * B * D;
		F = A * C * D;

                for(int k = 0; k < t; k++){
 	              	uxiulocal_x->u[i*t+k] = E.m[k];
                	uxiulocal_y->u[i*t+k] = F.m[k];
                }
	}


return 1;
}


template<class T, int t> int update_uf(lfield<T,t>* uf, lfield<T,t>* B_local, lfield<T,t>* A_local, double step){

///              A_local.exponentiate(sqrt(step));
///
///              B_local.exponentiate(-sqrt(step));
///
///              uf = B_local * uf * A_local;

        #pragma omp parallel for simd default(shared) //private(A,B,C,D,E,F)
        for(int i = 0; i < B_local->getNxl()*B_local->getNyl(); i++){

	        su3_matrix<double> A,B,C,D,E,F;

                for(int k = 0; k < t; k++){
                        A.m[k] = uf->u[i*t+k];
                        B.m[k] = -sqrt(step)*B_local->u[i*t+k];
                        C.m[k] = sqrt(step)*A_local->u[i*t+k];
                }

                B.exponentiate(-1.0);
                C.exponentiate(-1.0);

                E = B * A * C;

                for(int k = 0; k < t; k++){
                        uf->u[i*t+k] = E.m[k];
                }
        }

return 1;
}

template<class T, int t> int prepare_A_local(lfield<T,t>* A_local, lfield<T,t>* xi_local_x, lfield<T,t>* xi_local_y, momenta* mom, mpi_class* mpi, Coupling p, Kernel kk){

        #pragma omp parallel for simd collapse(2) default(shared)
        for(int ix = 0; ix < A_local->getNxl(); ix++){
        for(int iy = 0; iy < A_local->getNyl(); iy++){

		int i = ix*A_local->getNyl() + iy;

	        su3_matrix<double> C,D,E;

		int xx = ix + mpi->getPosX()*(A_local->getNxl());
		int yy = iy + mpi->getPosY()*(A_local->getNyl());

                if( xx >= Nx/2 )
                	xx = xx - Nx;

                if( yy >= Ny/2 )
                        yy = yy - Ny;

		double px = 2.0 * M_PI * xx / (1.0 * Nx);
		double py = 2.0 * M_PI * yy / (1.0 * Ny);

		if( fabs(px*px+py*py) > 10e-9 ){
	
			double coupling_constant;


			std::complex<double> AA(0.0, 0.0);
                        std::complex<double> BB(0.0, 0.0);

			if( kk == LINEAR_KERNEL ){

				if( p == SQRT_COUPLING_CONSTANT ){
					coupling_constant = 4.0*M_PI/( (11.0-2.0*3.0/3.0)*log( pow( pow(15.0*15.0/6.0/6.0,1.0/0.2) + pow(((px*px+py*py)*Nx*Ny)/6.0/6.0,1.0/0.2) , 0.2) ) );
				}
				if( p == NOISE_COUPLING_CONSTANT ){
					coupling_constant = 1.0;
				}
				if( p == NO_COUPLING_CONSTANT ){
					coupling_constant = 1.0;
				}

	                        std::complex<double> A(0.0, -2.0*M_PI*sqrt(coupling_constant)*px/(px*px+py*py));
        	                std::complex<double> B(0.0, -2.0*M_PI*sqrt(coupling_constant)*py/(px*px+py*py));
		
				AA = A;
				BB = B;
			}
			if( kk == SIN_KERNEL ){

				if( p == SQRT_COUPLING_CONSTANT ){
					coupling_constant = 4.0*M_PI/( (11.0-2.0*3.0/3.0)*log( pow( pow(15.0*15.0/6.0/6.0,1.0/0.2) + pow((mom->phat2(i)*Nx*Ny)/6.0/6.0,1.0/0.2) , 0.2) ) );
				}
				if( p == NOISE_COUPLING_CONSTANT ){
					coupling_constant = 1.0;
				}
				if( p == NO_COUPLING_CONSTANT ){
					coupling_constant = 1.0;
				}

	                        std::complex<double> A(0.0, -2.0*M_PI*sqrt(coupling_constant)*mom->pbarX(i)/mom->phat2(i));
        	                std::complex<double> B(0.0, -2.0*M_PI*sqrt(coupling_constant)*mom->pbarY(i)/mom->phat2(i));

				AA = A;
				BB = B;
			}

     		        for(int k = 0; k < t; k++){

		                C.m[k] = AA*xi_local_x->u[i*t+k];
        		        D.m[k] = BB*xi_local_y->u[i*t+k];
			}

			E = C + D;
		}

                for(int k = 0; k < t; k++){
 	              	A_local->u[i*t+k] = E.m[k];
                }
	}
	}

return 1;
}

template<class T, int t> int prepare_A_local(lfield<T,t>* A_local, lfield<T,t>* xi_local_x, lfield<T,t>* xi_local_y, lfield<T,t>* kernel_pbarx, lfield<T,t>* kernel_pbary){

///              xi_local_x_tmp = kernel_pbarx * xi_local_x;
///              xi_local_y_tmp = kernel_pbary * xi_local_y;

///              A_local = xi_local_x_tmp + xi_local_y_tmp;


        #pragma omp parallel for simd default(shared) 
        for(int i = 0; i < A_local->getNxl()*A_local->getNyl(); i++){

	        su3_matrix<double> A,B,C,D,E,F;

                for(int k = 0; k < t; k++){
			A.m[k] = kernel_pbarx->u[i*t+k];
			B.m[k] = kernel_pbary->u[i*t+k];

	                C.m[k] = xi_local_x->u[i*t+k]/(1.0*Nx*Ny);
        	        D.m[k] = xi_local_y->u[i*t+k]/(1.0*Nx*Ny);
//	                C.m[k] = xi_local_x->u[i*t+k];
//        	        D.m[k] = xi_local_y->u[i*t+k];


		}

		E = A * C + B * D;

                for(int k = 0; k < t; k++){
 	              	A_local->u[i*t+k] = E.m[k];
                }
	}



return 1;
}

template<class T, int t> int generate_gaussian(lfield<T,t>* xi_local_x, lfield<T,t>* xi_local_y, mpi_class* mpi, config* cnfg){

	if(t == 9){

	const double EPS = 10e-12;

	// 0 1 2
	// 3 4 5
	// 6 7 8

	#pragma omp parallel for simd default(shared)
	for(int i = 0; i < xi_local_x->getNxl()*xi_local_x->getNyl(); i++){

		static __thread std::ranlux24* generator = nullptr;
	        if (!generator){
		  	 std::hash<std::thread::id> hasher;
			 generator = new std::ranlux24(clock() + hasher(std::this_thread::get_id()));
		}
		std::normal_distribution<double> distribution{0.0,1.0};	
  
    	    //set to zero
	    for(int j = 0; j < t; j++){
		xi_local_x->u[i*t+j] = 0.0;
		xi_local_y->u[i*t+j] = 0.0;
	    }

	    double n[8];
	    double m[8];

	    for(int k = 0; k < 8; k++){
                	n[k] = distribution(*generator);
                	m[k] = distribution(*generator);
	    }

	    std::complex<double> unit(0.0,1.0);
	    double sqrt_3 = sqrt(3.0);

		xi_local_x->u[i*t+1] += n[0];
		xi_local_x->u[i*t+3] += n[0];
		xi_local_y->u[i*t+1] += m[0];
		xi_local_y->u[i*t+3] += m[0];

       		xi_local_x->u[i*t+1] += unit*n[1];
		xi_local_x->u[i*t+3] -= unit*n[1];
		xi_local_y->u[i*t+1] += unit*m[1];
		xi_local_y->u[i*t+3] -= unit*m[1];

       		xi_local_x->u[i*t+0] += n[2];
		xi_local_x->u[i*t+4] -= n[2];
		xi_local_y->u[i*t+0] += m[2];
		xi_local_y->u[i*t+4] -= m[2];

       		xi_local_x->u[i*t+2] += n[3];
		xi_local_x->u[i*t+6] += n[3];
		xi_local_y->u[i*t+2] += m[3];
		xi_local_y->u[i*t+6] += m[3];

       		xi_local_x->u[i*t+2] += unit*n[4];
		xi_local_x->u[i*t+6] -= unit*n[4];
		xi_local_y->u[i*t+2] += unit*m[4];
		xi_local_y->u[i*t+6] -= unit*m[4];

       		xi_local_x->u[i*t+5] += n[5];
		xi_local_x->u[i*t+7] += n[5];
		xi_local_y->u[i*t+5] += m[5];
		xi_local_y->u[i*t+7] += m[5];

       		xi_local_x->u[i*t+5] += unit*n[6];
		xi_local_x->u[i*t+7] -= unit*n[6];
		xi_local_y->u[i*t+5] += unit*m[6];
		xi_local_y->u[i*t+7] -= unit*m[6];

       		xi_local_x->u[i*t+0] += n[7]/sqrt_3;
		xi_local_x->u[i*t+4] += n[7]/sqrt_3;
		xi_local_x->u[i*t+8] += -2.0*n[7]/sqrt_3;
		xi_local_y->u[i*t+0] += m[7]/sqrt_3;
		xi_local_y->u[i*t+4] += m[7]/sqrt_3;
		xi_local_y->u[i*t+8] += -2.0*m[7]/sqrt_3;

	}

	}else{

		printf("Invalid lfield classes for setGaussian function\n");

	}


return 1;
}

template<class T, int t> int generate_gaussian_with_noise_coupling_constant(lfield<T,t>* xi_local_x, lfield<T,t>* xi_local_y, momenta* mom, mpi_class* mpi, config* cnfg){

	if(t == 9){

	const double EPS = 10e-12;

	// 0 1 2
	// 3 4 5
	// 6 7 8

	const double tmp = pow(15.0*15.0/6.0/6.0,1.0/0.2);
	const double tmp2 = 4.0*M_PI/ (11.0-2.0*3.0/3.0);

	#pragma omp parallel for simd default(shared)
	for(int i = 0; i < xi_local_x->getNxl()*xi_local_x->getNyl(); i++){

	    static __thread std::ranlux24* generator = nullptr;
	    if (!generator){
		  	 std::hash<std::thread::id> hasher;
			 generator = new std::ranlux24(clock() + hasher(std::this_thread::get_id()));
	    }
    	    std::normal_distribution<double> distribution{0.0, 1.0};


	    double sqrt_coupling_constant = sqrt(tmp2 / log( pow( tmp + pow((mom->phat2(i)*Nx*Ny)/6.0/6.0,1.0/0.2) , 0.2) ));

	    //set to zero
	    for(int j = 0; j < t; j++){
		xi_local_x->u[i*t+j] = 0.0;
		xi_local_y->u[i*t+j] = 0.0;
	    }

	    double n[8];
	    double m[8];

	    for(int k = 0; k < 8; k++){
                	n[k] = distribution(*generator)*sqrt_coupling_constant;
                	m[k] = distribution(*generator)*sqrt_coupling_constant;
	    }

	    std::complex<double> unit(0.0,1.0);
	    double sqrt_3 = sqrt(3.0);

		xi_local_x->u[i*t+1] += n[0];
		xi_local_x->u[i*t+3] += n[0];
		xi_local_y->u[i*t+1] += m[0];
		xi_local_y->u[i*t+3] += m[0];

       		xi_local_x->u[i*t+1] += unit*n[1];
		xi_local_x->u[i*t+3] -= unit*n[1];
		xi_local_y->u[i*t+1] += unit*m[1];
		xi_local_y->u[i*t+3] -= unit*m[1];

       		xi_local_x->u[i*t+0] += n[2];
		xi_local_x->u[i*t+4] -= n[2];
		xi_local_y->u[i*t+0] += m[2];
		xi_local_y->u[i*t+4] -= m[2];

       		xi_local_x->u[i*t+2] += n[3];
		xi_local_x->u[i*t+6] += n[3];
		xi_local_y->u[i*t+2] += m[3];
		xi_local_y->u[i*t+6] += m[3];

       		xi_local_x->u[i*t+2] += unit*n[4];
		xi_local_x->u[i*t+6] -= unit*n[4];
		xi_local_y->u[i*t+2] += unit*m[4];
		xi_local_y->u[i*t+6] -= unit*m[4];

       		xi_local_x->u[i*t+5] += n[5];
		xi_local_x->u[i*t+7] += n[5];
		xi_local_y->u[i*t+5] += m[5];
		xi_local_y->u[i*t+7] += m[5];

       		xi_local_x->u[i*t+5] += unit*n[6];
		xi_local_x->u[i*t+7] -= unit*n[6];
		xi_local_y->u[i*t+5] += unit*m[6];
		xi_local_y->u[i*t+7] -= unit*m[6];

       		xi_local_x->u[i*t+0] += n[7]/sqrt_3;
		xi_local_x->u[i*t+4] += n[7]/sqrt_3;
		xi_local_x->u[i*t+8] += -2.0*n[7]/sqrt_3;
		xi_local_y->u[i*t+0] += m[7]/sqrt_3;
		xi_local_y->u[i*t+4] += m[7]/sqrt_3;
		xi_local_y->u[i*t+8] += -2.0*m[7]/sqrt_3;

	}

	}else{

		printf("Invalid lfield classes for setGaussian function\n");

	}


return 1;
}


template<class T, int t> int prepare_A_and_B_local(int x, int y, int x_global, int y_global, gfield<T,t>* xi_global_x, gfield<T,t>* xi_global_y, 
				lfield<T,t>* A_local, lfield<T,t>* B_local, gfield<T,t>* uf_global, positions* postable, int rr, Coupling p, Kernel kk){

	double sumAlocalRe[9];
	double sumAlocalIm[9];
	double sumBlocalRe[9];
	double sumBlocalIm[9];

        for(int k = 0; k < t; k++){

		sumAlocalRe[k] = 0.0;
		sumAlocalIm[k] = 0.0;
		sumBlocalRe[k] = 0.0;
		sumBlocalIm[k] = 0.0;

	}

        #pragma omp parallel for simd collapse(2) default(shared) reduction(+:sumAlocalRe[:9]), reduction(+:sumAlocalIm[:9]) reduction(+:sumBlocalRe[:9]), reduction(+:sumBlocalIm[:9]) 
        for(int xx = 0; xx < Nx; xx++){
                for(int yy = 0; yy < Ny; yy++){


			std::complex<double> A,B;
		        su3_matrix<double> C,D,E,F,G,H,K;

                        int i = xx*Ny+yy;

			double dx;
			double dy;
			double rrr;

			if( kk == SIN_KERNEL ){

				int ii = 0;
				if( x_global >= xx)
					ii += (x_global - xx)*Ny;
				else
					ii += (x_global - xx + Nx)*Ny;

				if( y_global >= yy)
					ii += (y_global - yy);
				else
					ii += (y_global - yy + Ny);

                        	dx = postable->xhatX(ii); 
                        	dy = postable->xhatY(ii); 
                        
                        	rrr = postable->xbar2(ii);
			}
	
			if( kk == LINEAR_KERNEL ){
			
                        	dx = x_global - xx;
	                        if( dx >= Nx/2 )
        	                      dx = dx - Nx;
                	        if( dx < -Nx/2 )
                  		      dx = dx + Nx;

                        	dy = y_global - yy;
	                        if( dy >= Ny/2 )
        	                        dy = dy - Ny;
                	        if( dy < -Ny/2 )
                        		dy = dy + Ny;
						
                        	rrr = 1.0*(dx*dx+dy*dy);
			}
				
                        double rrrmin = rrr;

			if( p == HATTA_COUPLING_CONSTANT ){
	                        //hatta condition!!!            
	                        if( rrr <= rr ){
	                                rrrmin = rrr;
	                        }else{
                                	rrrmin = rr;
                        	}
			}
	
			const double lambda = pow(15.0*15.0/6.0/6.0,1.0/0.2);

			double sqrt_coupling_constant;

			if( p == SQRT_COUPLING_CONSTANT ){
				sqrt_coupling_constant = sqrt(4.0*M_PI/(  (11.0-2.0*3.0/3.0) * log( pow( lambda + 1.26/pow(6.0*6.0*rrrmin/Nx/Ny,1.0/0.2) , 0.2 ) )) );
			}
			if( p == NOISE_COUPLING_CONSTANT ){
				sqrt_coupling_constant = 1.0;
			}
			if( p == NO_COUPLING_CONSTANT ){
				sqrt_coupling_constant = 1.0;
			}

                        if( rrr	 > 10e-9 ){

                                A.real(sqrt_coupling_constant*dx/rrr);
				A.imag(0.0);

                                B.real(sqrt_coupling_constant*dy/rrr);
				B.imag(0.0);
                        }

	                for(int k = 0; k < t; k++){

		                C.m[k] = A*xi_global_x->u[i*t+k];
        		        D.m[k] = B*xi_global_y->u[i*t+k];

				G.m[k] = uf_global->u[i*t+k];
			}

	                H.m[0] = std::conj(G.m[0]);
               		H.m[1] = std::conj(G.m[3]);
	                H.m[2] = std::conj(G.m[6]);
	                H.m[3] = std::conj(G.m[1]);
	                H.m[4] = std::conj(G.m[4]);
	                H.m[5] = std::conj(G.m[7]);
	                H.m[6] = std::conj(G.m[2]);
	                H.m[7] = std::conj(G.m[5]);
	                H.m[8] = std::conj(G.m[8]);


			E = C + D;

			K = G * E * H;

	                for(int k = 0; k < t; k++){

		                sumAlocalRe[k] += E.m[k].real();
        		        sumAlocalIm[k] += E.m[k].imag();

		                sumBlocalRe[k] += K.m[k].real();
        		        sumBlocalIm[k] += K.m[k].imag();
			}
		}
	}

        for(int k = 0; k < t; k++){
              	A_local->u[(x*A_local->getNyl()+y)*t+k].real(sumAlocalRe[k]);
              	B_local->u[(x*A_local->getNyl()+y)*t+k].real(sumBlocalRe[k]);
              	A_local->u[(x*A_local->getNyl()+y)*t+k].imag(sumAlocalIm[k]);
              	B_local->u[(x*A_local->getNyl()+y)*t+k].imag(sumBlocalIm[k]);
	}

return 1;
}

template<class T, int t> int print(int measurement, lfield<T,t>* sum, lfield<T,t>* err, momenta* mom, double x, mpi_class* mpi, std::string const &fileroot){


        FILE* f;
        char filename[100];

        sprintf(filename, "%s_%i_%i_mpi%i_r%i.dat", fileroot.c_str(), Nx, Ny, mpi->getSize(), mpi->getRank());

        f = fopen(filename, "a+");

        for(int xx = 0; xx < sum->getNxl(); xx++){
                for(int yy = 0; yy < sum->getNyl(); yy++){

                        int i = xx*(sum->getNyl())+yy;

                        if( fabs(xx + mpi->getPosX()*(sum->getNxl()) - yy - mpi->getPosY()*(sum->getNyl())) <= 4 ){

                                fprintf(f, "%i %i %i \t %f %e %e\n", measurement, xx+(mpi->getPosX()*(sum->getNxl())), yy+(mpi->getPosY()*(sum->getNyl())), sqrt(mom->phat2(i)), x*(mom->phat2(i))*(sum->u[i*t+0].real()), x*(mom->phat2(i))*(err->u[i*t+0].real()));

                        }
                }
        }

        fclose(f);

return 1;
}
#endif
