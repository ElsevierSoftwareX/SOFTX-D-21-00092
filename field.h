#ifndef H_FIELD
#define H_FIELD

#include <iostream>
#include <stdlib.h>
#include <complex>
#include <ccomplex>

//#include <complex.h>
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

template<class T, int t> class field {

	public:

		std::complex<T>** u;		

		int Nxl, Nyl;

	public:

		field(int NNx, int NNy);

		~field(void);
};

template<class T, int t> field<T,t>::field(int NNx, int NNy) {

	int i;

	u = (std::complex<T>**)malloc(t*sizeof(std::complex<T>*));

	for(i = 0; i < t; i++){

		u[i] = (std::complex<T>*)malloc(NNx*NNy*sizeof(std::complex<T>));

		for(int j = 0; j < NNx*NNy; j++)
			u[i][j] = 0.0;

	}

	Nxl = NNx;
	Nyl = NNy;
}

template<class T, int t> field<T,t>::~field() {

	int i;

	for(i = 0; i < t; i++){

		free(u[i]);

	}

}

template<class T, int t> class lfield;

template<class T> class gmatrix;

template<class T, int t> class gfield: public field<T,t> {


	public:

		int Nxg, Nyg;

		//T getZero(void){ return this.u[0][0]; }

		int allgather(lfield<T,t>* ulocal, mpi_class* mpi);

		gfield(int NNx, int NNy) : field<T,t>{NNx, NNy} { Nxg = NNx; Nyg = NNy;};

		friend class fftw2D;

		int average_and_symmetrize();

		gfield<T,t>& operator= ( const gfield<T,t>& f );

		gfield<T,t>* hermitian();

		int setKernelXbarX(int x, int y, positions* postable);
		int setKernelXbarY(int x, int y, positions* postable);
		int setKernelXbarXWithCouplingConstant(int x, int y, positions* postable);
		int setKernelXbarYWithCouplingConstant(int x, int y, positions* postable);

		int setCorrelationsForCouplingConstant();

		int multiplyByCholesky(gmatrix<T>* mm);

		lfield<T,t>* reduce(int NNx, int NNy, mpi_class* mpi);

		int setToZero(void){
			for(int i = 0; i < Nxg*Nyg; i ++){
				for(int k = 0; k < t; k++){
					this->u[k][i] = 0.0;
				}
			}
		return 1;
		}



};


template<class T, int t> class lfield: public field<T,t> {

	public:

		int Nxl, Nyl;
		int Nxl_buf, Nyl_buf;

		lfield(int NNx, int NNy) : field<T,t>{NNx, NNy} { Nxl = NNx; Nyl = NNy; };

		friend class fftw1D;

		int mpi_exchange_boundaries(mpi_class* mpi);

		friend class gfield<T,t>;

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
			for(int i = 0; i < Nxl*Nyl; i ++){
				for(int k = 0; k < t; k++){
					this->u[k][i] = 0.0;
				}
			}
		return 1;
		}


		int setToUnit(void){
			#pragma omp parallel for simd default(shared)
			for(int i = 0; i < Nxl*Nyl; i ++){
				this->u[0][i] = 1.0;
				this->u[4][i] = 1.0;
				this->u[8][i] = 1.0;

				this->u[1][i] = 0.0;
				this->u[2][i] = 0.0;
				this->u[3][i] = 0.0;
				this->u[5][i] = 0.0;
				this->u[6][i] = 0.0;
				this->u[7][i] = 0.0;
			}
		return 1;
		}
		int setToOne(void){
			#pragma omp parallel for simd default(shared)
			for(int i = 0; i < Nxl*Nyl; i ++){
				for(int k = 0; k < t; k++){
					this->u[k][i] = 1.0;
				}
			}
		return 1;
		}

		int setToFixed(void){
			for(int ix = 0; ix < Nxl; ix ++){
				for(int iy = 0; iy < Nyl; iy ++){

					for(int k = 0; k < t; k++){
						if( ix < Nxl/2 && iy < Nyl/2 ){
							this->u[k][ix*Nyl + iy] = ((ix)%Nxl)*((ix)%Nxl) + ((iy)%Nyl)*((iy)%Nyl);
						}
						if( ix > Nxl/2 && iy < Nyl/2 ){
							this->u[k][ix*Nyl + iy] = ((ix-Nxl)%Nxl)*((ix-Nxl)%Nxl) + ((iy)%Nyl)*((iy)%Nyl);
						}
						if( ix < Nxl/2 && iy > Nyl/2 ){
							this->u[k][ix*Nyl + iy] = ((ix)%Nxl)*((ix)%Nxl) + ((iy-Nyl)%Nyl)*((iy-Nyl)%Nyl);
						}
						if( ix > Nxl/2 && iy > Nyl/2 ){
							this->u[k][ix*Nyl + iy] = ((ix-Nxl)%Nxl)*((ix-Nxl)%Nxl) + ((iy-Nyl)%Nyl)*((iy-Nyl)%Nyl);
						}

						//this->u[k][ix*Nyl + iy] = (ix-Nxl)*(ix-Nxl) + (iy-Nyl)*(iy-Nyl) + I*0.0;
					}
				}
			}
		return 1;
		}


		int setMVModel(MV_class* MVconfig, rand_class* rr);
		int setUnitModel(rand_class* rr);

		int setGaussian(mpi_class* rr, config* cnfg);

		int solvePoisson(double mass, double g, momenta* momtable);

		int exponentiate();
		int exponentiate(double s);

		int setKernelPbarX(momenta* mom);
		int setKernelPbarY(momenta* mom);

		int setKernelPbarXWithCouplingConstant(momenta* mom);
		int setKernelPbarYWithCouplingConstant(momenta* mom);

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
		int printDebug(double x, mpi_class* mpi);
		int printDebugRadial(double x);


};

template<class T, int t> lfield<T,t>& lfield<T,t>::operator= ( const lfield<T,t>& f ){

			if( this != &f ){

			#pragma omp parallel for simd default(shared)
			for(int i = 0; i < f.Nxl*f.Nyl; i ++){
		
				for(int k = 0; k < t; k++){

					this->u[k][i] = f.u[k][i];
				}
			}


			}

		return *this;
		}

template<class T, int t> gfield<T,t>& gfield<T,t>::operator= ( const gfield<T,t>& f ){

			if( this != &f ){

			#pragma omp parallel for simd default(shared)
			for(int i = 0; i < f.Nxg*f.Nyg; i ++){
				for(int k = 0; k < t; k++){

					this->u[k][i] = f.u[k][i];
				}
			}

			}

		return *this;
		}

template<class T, int t> lfield<T,t>& lfield<T,t>::operator*= ( const lfield<T,t>& f ){

			if( this != &f ){

			#pragma omp parallel for simd default(shared)
			for(int i = 0; i < f.Nxl*f.Nyl; i++){
			
				su3_matrix<double> A,B,C;

				for(int k = 0; k < t; k++){

					A.m[k] = this->u[k][i];
					B.m[k] = f.u[k][i];
				}
		
				C = A*B;

				for(int k = 0; k < t; k++){

					this->u[k][i] = C.m[k];
				}
			}

			}

		return *this;
		}

template<class T, int t> lfield<T,t>& lfield<T,t>::operator+= ( const lfield<T,t>& f ){

			if( this != &f ){

			#pragma omp parallel for simd default(shared)
			for(int i = 0; i < f.Nxl*f.Nyl; i++){
				for(int k = 0; k < t; k++){

					this->u[k][i] += f.u[k][i];
				}
			}

			}

		return *this;
		}

template<class T, int t> lfield<T,t> operator * ( const lfield<T,t> &f , const lfield<T,t> &g){

			lfield<T,t> result(f.Nxl, f.Nyl);

			result.setToZero();

//			printf("checking actual size of result: %i, %i\n", result.Nxl, result.Nyl);
//			printf("checking actual size of input: %i, %i\n", f.Nxl, f.Nyl);
//			printf("checking actual size of input: %i, %i\n", g.Nxl, g.Nyl);

//			printf("starting multiplicatin in * operator, size of result t = %i\n", t);

			if( f.Nxl == g.Nxl && f.Nyl == g.Nyl ){

			#pragma omp parallel for simd default(shared)
			for(int i = 0; i < f.Nxl*f.Nyl; i++){

				su3_matrix<double> A,B,C;
		
//				printf("element %i\n", i);
	
				for(int k = 0; k < t; k++){

//					printf("direction %i\n", k);

					A.m[k] = f.u[k][i]; //this->u[k][i];
					B.m[k] = g.u[k][i];
				}
		
				C = A*B;

				for(int k = 0; k < t; k++){

					result.u[k][i] = C.m[k];
				}
			}

			}else{

				printf("Size of input objects in * operator is different!\n");

			}


		return result;
		}

template<class T, int t> gfield<T,t> operator * ( const gfield<T,t> &f , const gfield<T,t> &g){

			gfield<T,t> result(f.Nxg, f.Nyg);

			result.setToZero();

//			printf("checking actual size of result: %i, %i\n", result.Nxl, result.Nyl);
//			printf("checking actual size of input: %i, %i\n", f.Nxl, f.Nyl);
//			printf("checking actual size of input: %i, %i\n", g.Nxl, g.Nyl);

//			printf("starting multiplicatin in * operator, size of result t = %i\n", t);

			if( f.Nxg == g.Nxg && f.Nyg == g.Nyg ){

			#pragma omp parallel for simd default(shared)
			for(int i = 0; i < f.Nxg*f.Nyg; i++){
			
				su3_matrix<double> A,B,C;

//				printf("element %i\n", i);
	
				for(int k = 0; k < t; k++){

//					printf("direction %i\n", k);

					A.m[k] = f.u[k][i]; //this->u[k][i];
					B.m[k] = g.u[k][i];
				}
		
				C = A*B;

				for(int k = 0; k < t; k++){

					result.u[k][i] = C.m[k];
				}
			}

			}else{

				printf("Size of input objects in * operator is different!\n");

			}


		return result;
		}



template<class T, int t> lfield<T,t> operator + ( const lfield<T,t> &f, const lfield<T,t>& g ){

			lfield<T,t> result(f.Nxl, f.Nyl);

			result.setToZero();

//			printf("checking actual size: %i, %i\n", result.Nxl, result.Nyl);
//			printf("checking actual size of input: %i, %i\n", f.Nxl, f.Nyl);
//			printf("checking actual size of input: %i, %i\n", g.Nxl, g.Nyl);

//			printf("starting addition in + operator, size of result t = %i\n", t);

			if( f.Nxl == g.Nxl && f.Nyl == g.Nyl ){

			#pragma omp parallel for simd default(shared)
			for(int i = 0; i < f.Nxl*f.Nyl; i ++){

				su3_matrix<double> A,B,C;
		
				for(int k = 0; k < t; k++){

					A.m[k] = f.u[k][i];
					B.m[k] = g.u[k][i];
				}
		
				C = A+B;

				for(int k = 0; k < t; k++){

					result.u[k][i] = C.m[k];
				}
			}

			}else{

				printf("Size of input objects in + operator is different!\n");

			}


//			printf("Size of new object craeted in +operator: Nxl = %i, Nyl = %i\n", result.Nxl, result.Nyl);

		return result;
		}


template<class T, int t> gfield<T,t> operator + ( const gfield<T,t> &f, const gfield<T,t>& g ){

			gfield<T,t> result(f.Nxg, f.Nyg);

			result.setToZero();

//			printf("checking actual size: %i, %i\n", result.Nxl, result.Nyl);
//			printf("checking actual size of input: %i, %i\n", f.Nxl, f.Nyl);
//			printf("checking actual size of input: %i, %i\n", g.Nxl, g.Nyl);

//			printf("starting addition in + operator, size of result t = %i\n", t);

			if( f.Nxg == g.Nxg && f.Nyg == g.Nyg ){

			#pragma omp parallel for simd default(shared)
			for(int i = 0; i < f.Nxg*f.Nyg; i ++){
			
				su3_matrix<double> A,B,C;

				for(int k = 0; k < t; k++){

					A.m[k] = f.u[k][i];
					B.m[k] = g.u[k][i];
				}
		
				C = A+B;

				for(int k = 0; k < t; k++){

					result.u[k][i] = C.m[k];
				}
			}

			}else{

				printf("Size of input objects in + operator is different!\n");

			}


//			printf("Size of new object craeted in +operator: Nxl = %i, Nyl = %i\n", result.Nxl, result.Nyl);

		return result;
		}



template<class T, int t> int gfield<T,t>::allgather(lfield<T,t>* ulocal, mpi_class* mpi){


	static T* data_local_re = (T*)malloc(ulocal->Nxl*ulocal->Nyl*sizeof(T));
	static T* data_local_im = (T*)malloc(ulocal->Nxl*ulocal->Nyl*sizeof(T));

	static T* data_global_re = (T*)malloc(Nxg*Nyg*sizeof(T));
	static T* data_global_im = (T*)malloc(Nxg*Nyg*sizeof(T));

	int i,k;

	for(k = 0; k < t; k++){

		#pragma omp parallel for simd default(shared)
		for(i = 0; i < ulocal->Nxl*ulocal->Nyl; i++){

			data_local_re[i] = ulocal->u[k][i].real();
			data_local_im[i] = ulocal->u[k][i].imag();

		}

   		MPI_Allgather(data_local_re, ulocal->Nxl*ulocal->Nyl, MPI_DOUBLE, data_global_re, ulocal->Nxl*ulocal->Nyl, MPI_DOUBLE, MPI_COMM_WORLD); 
	   	MPI_Allgather(data_local_im, ulocal->Nxl*ulocal->Nyl, MPI_DOUBLE, data_global_im, ulocal->Nxl*ulocal->Nyl, MPI_DOUBLE, MPI_COMM_WORLD); 

		int size = mpi->getSize();

		for(int kk = 0; kk < size; kk++){
		
			int local_volume = ulocal->Nxl * ulocal->Nyl;

			#pragma omp parallel for simd collapse(2) default(shared)
			for(int xx = 0; xx < ulocal->Nxl; xx++){
				for(int yy = 0; yy < ulocal->Nyl; yy++){

					int i = xx*ulocal->Nyl + yy;

					//will only work for parallelization in y direction
					//int ii = xx*Nyg + (yy + kk*ulocal->Nyl);

					//will only work for parallelization in x direction
					int ii = (xx + kk*ulocal->Nxl)*Nyg + yy;

					this->u[k][ii] = std::complex<double>(data_global_re[i+kk*local_volume], data_global_im[i+kk*local_volume]);
					//this->u[k][i] = ulocal->u[k][i]; //data_global_re[i] + I*data_global_im[i];

				}
			}
		}
	}

//	free(data_local_re);
//	free(data_local_im);

//	free(data_global_re);
//	free(data_global_im);

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

  	    for(yy = 0; yy < Nyl; yy++){
		bufor_send_n[yy] = this->u[0][buf_pos(Nxl-1,yy)].real();
	    }

	    printf("X data exchange: rank %i sending to %i\n", mpi->getRank(), mpi->getXNeighbourNext());
	    printf("X data exchange: rank %i receiving from %i\n", mpi->getRank(), mpi->getXNeighbourNext());

	    MPI_Send(bufor_send_n, Nyl_buf, MPI_DOUBLE, mpi->getXNeighbourNext(), 11, MPI_COMM_WORLD);
    	    MPI_Recv(bufor_receive_n, Nyl_buf, MPI_DOUBLE, mpi->getXNeighbourPrevious(), 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  	    for(yy = 0; yy < Nyl; yy++){
		this->u[0][buf_pos_ex(0,yy)] = bufor_receive_n[yy];
	    }

   	    bufor_send_p = (double*) malloc(Nyl_buf*sizeof(double));
	    bufor_receive_p = (double*) malloc(Nyl_buf*sizeof(double));

	    for(yy = 0; yy < Nyl; yy++){
		bufor_send_p[yy] = this->u[0][buf_pos(0,yy)].real();
	    }
	
 	    printf("X data exchange: rank %i sending to %i\n", mpi->getRank(), mpi->getXNeighbourPrevious());
	    printf("X data exchange: rank %i receiving to %i\n", mpi->getRank(), mpi->getXNeighbourPrevious());

	    MPI_Send(bufor_send_p, Nyl_buf, MPI_DOUBLE, mpi->getXNeighbourPrevious(), 12, MPI_COMM_WORLD);
    	    MPI_Recv(bufor_receive_p, Nyl_buf, MPI_DOUBLE, mpi->getXNeighbourNext(), 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  	    for(yy = 0; yy < Nyl; yy++){
		this->u[0][buf_pos_ex(Nxl+1,yy)] = bufor_receive_p[yy];
	    }
    }

    if( mpi->getExchangeY() == 1 ){

	    int xx; 

	    bufor_send_n = (double*) malloc(Nxl_buf*sizeof(double));
	    bufor_receive_n = (double*) malloc(Nxl_buf*sizeof(double));

  	    for(xx = 0; xx < Nxl; xx++){
		bufor_send_n[xx] = this->u[0][buf_pos(xx,Nyl-1)].real();
	    }

	    printf("Y data exchange: rank %i sending to %i\n", mpi->getRank(), mpi->getYNeighbourNext());
	    printf("Y data exchange: rank %i receiving from %i\n", mpi->getRank(), mpi->getYNeighbourNext());

	    MPI_Send(bufor_send_n, Nxl_buf, MPI_DOUBLE, mpi->getYNeighbourNext(), 13, MPI_COMM_WORLD);
    	    MPI_Recv(bufor_receive_n, Nxl_buf, MPI_DOUBLE, mpi->getYNeighbourPrevious(), 13, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for(xx = 0; xx < Nxl; xx++){
		this->u[0][buf_pos_ex(xx,0)] = bufor_receive_n[xx];
	    }

	    bufor_send_p = (double*) malloc(Nxl_buf*sizeof(double));
	    bufor_receive_p = (double*) malloc(Nxl_buf*sizeof(double));

	    for(xx = 0; xx < Nxl; xx++){
		bufor_send_p[xx] = this->u[0][buf_pos(xx,0)].real();
	    }
	
 	    printf("Y data exchange: rank %i sending to %i\n", mpi->getRank(), mpi->getYNeighbourPrevious());
	    printf("Y data exchange: rank %i receiving to %i\n", mpi->getRank(), mpi->getYNeighbourPrevious());

	    MPI_Send(bufor_send_p, Nxl_buf, MPI_DOUBLE, mpi->getYNeighbourPrevious(), 14, MPI_COMM_WORLD);
    	    MPI_Recv(bufor_receive_p, Nxl_buf, MPI_DOUBLE, mpi->getYNeighbourNext(), 14, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  	    for(xx = 0; xx < Nxl; xx++){
		this->u[0][buf_pos_ex(xx,Nyl+1)] = bufor_receive_p[xx];
	    }
    }

return 1;
}

template<class T, int t> int lfield<T,t>::setMVModel(MV_class* MVconfig, rand_class* rr){

	if(t == 9){

	const double EPS = 10e-12;

//	#pragma omp parallel for simd default(shared)
	for(int i = 0; i < Nxl*Nyl; i++){

	        double n[8];

		for(int k = 0; k < 8; k++){
	         	n[k] = sqrt( pow(MVconfig->g_parameter,2.0) * pow(MVconfig->mu_parameter,2.0) / MVconfig->Ny_parameter ) * sqrt( -2.0 * log( EPS + rr->get() ) ) * cos( rr->get() * 2.0 * M_PI);
		}

	//these are the LAMBDAs and not the generators t^a = lambda/2.

	// 0 1 2
	// 3 4 5
	// 6 7 8

            //lambda_nr(1)%su3(1,2) =  runit
            //lambda_nr(1)%su3(2,1) =  runit
		this->u[1][i] += std::complex<double>(n[0],0.0);
		this->u[3][i] += std::complex<double>(n[0],0.0);


            //lambda_nr(2)%su3(1,2) = -iunit
            //lambda_nr(2)%su3(2,1) =  iunit
		this->u[1][i] += std::complex<double>(0.0,n[1]);
		this->u[3][i] -= std::complex<double>(0.0,n[1]);


            //lambda_nr(3)%su3(1,1) =  runit
            //lambda_nr(3)%su3(2,2) = -runit
		this->u[0][i] += std::complex<double>(n[2],0.0);
		this->u[4][i] -= std::complex<double>(n[2],0.0);


            //lambda_nr(4)%su3(1,3) =  runit
            //lambda_nr(4)%su3(3,1) =  runit
		this->u[2][i] += std::complex<double>(n[3],0.0);
		this->u[6][i] += std::complex<double>(n[3],0.0);


            //lambda_nr(5)%su3(1,3) = -iunit
            //lambda_nr(5)%su3(3,1) =  iunit
		this->u[2][i] += std::complex<double>(0.0,n[4]);
		this->u[6][i] -= std::complex<double>(0.0,n[4]);


            //lambda_nr(6)%su3(2,3) =  runit
            //lambda_nr(6)%su3(3,2) =  runit
		this->u[5][i] += std::complex<double>(n[5],0.0);
		this->u[7][i] += std::complex<double>(n[5],0.0);


            //lambda_nr(7)%su3(2,3) = -iunit
            //lambda_nr(7)%su3(3,2) =  iunit
		this->u[5][i] += std::complex<double>(0.0,n[6]);
		this->u[7][i] -= std::complex<double>(0.0,n[6]);


            //lambda_nr(8)%su3(1,1) =  cst8
            //lambda_nr(8)%su3(2,2) =  cst8
            //lambda_nr(8)%su3(3,3) =  -(two*cst8)

		this->u[0][i] += std::complex<double>(n[7]/sqrt(3.0),0.0);
		this->u[4][i] += std::complex<double>(n[7]/sqrt(3.0),0.0);
		this->u[8][i] += std::complex<double>(-2.0*n[7]/sqrt(3.0),0.0);
	}

//	fclose(f);

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

//				hh=sqrt(real(1.0*g_parameter**2*mu_parameter**2/Ny_parameter, kind=REALKND)) * &
//                              & sqrt((real(-2.0, kind=REALKND))*log(EPSI+real(ranvec(2*m-1),kind=REALKND))) * &
//                              & cos(real(ranvec(2*m),kind=REALKND) * real(TWOPI, kind=REALKND))

		for(int k = 0; k < 8; k++)
                	this->u[k][i] = sqrt( -2.0 * log( EPS + rr->get() ) ) * cos( rr->get() * 2.0 * M_PI);
		
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
    		//return distribution(*generator);

//        std::ranlux24_base rgenerator;
//        std::uniform_real_distribution<double> distribution{0.0,1.0};
//
//        rand_class(mpi_class *mpi, config *cnfg){
//
//        rgenerator.seed(cnfg->seed + 64*mpi->getRank() + omp_get_thread_num());
//
//        }


	    double n[8];

	    for(int k = 0; k < 8; k++)
                	n[k] = distribution(*generator); //sqrt( -2.0 * log( EPS + distribution(*generator) ) ) * cos( distribution(*generator) * 2.0 * M_PI);
//                	n[k] = sqrt( -2.0 * log( EPS + rr->get() ) ) * cos( rr->get() * 2.0 * M_PI);
	
   	    //these are the LAMBDAs and not the generators t^a = lambda/2.

            //lambda_nr(1)%su3(1,2) =  runit
            //lambda_nr(1)%su3(2,1) =  runit
		this->u[1][i] += std::complex<double>(n[0],0.0);
		this->u[3][i] += std::complex<double>(n[0],0.0);


            //lambda_nr(2)%su3(1,2) = -iunit
            //lambda_nr(2)%su3(2,1) =  iunit
		this->u[1][i] += std::complex<double>(0.0,n[1]);
		this->u[3][i] -= std::complex<double>(0.0,n[1]);


            //lambda_nr(3)%su3(1,1) =  runit
            //lambda_nr(3)%su3(2,2) = -runit
		this->u[0][i] += std::complex<double>(n[2],0.0);
		this->u[4][i] -= std::complex<double>(n[2],0.0);


            //lambda_nr(4)%su3(1,3) =  runit
            //lambda_nr(4)%su3(3,1) =  runit
		this->u[2][i] += std::complex<double>(n[3],0.0);
		this->u[6][i] += std::complex<double>(n[3],0.0);


            //lambda_nr(5)%su3(1,3) = -iunit
            //lambda_nr(5)%su3(3,1) =  iunit
		this->u[2][i] += std::complex<double>(0.0,n[4]);
		this->u[6][i] -= std::complex<double>(0.0,n[4]);


            //lambda_nr(6)%su3(2,3) =  runit
            //lambda_nr(6)%su3(3,2) =  runit
		this->u[5][i] += std::complex<double>(n[5],0.0);
		this->u[7][i] += std::complex<double>(n[5],0.0);


            //lambda_nr(7)%su3(2,3) = -iunit
            //lambda_nr(7)%su3(3,2) =  iunit
		this->u[5][i] += std::complex<double>(0.0,n[6]);
		this->u[7][i] -= std::complex<double>(0.0,n[6]);


            //lambda_nr(8)%su3(1,1) =  cst8
            //lambda_nr(8)%su3(2,2) =  cst8
            //lambda_nr(8)%su3(3,3) =  -(two*cst8)

		this->u[0][i] += std::complex<double>(n[7]/sqrt(3.0),0.0);
		this->u[4][i] += std::complex<double>(n[7]/sqrt(3.0),0.0);
		this->u[8][i] += std::complex<double>(-2.0*n[7]/sqrt(3.0),0.0);
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
			this->u[k][i] *= std::complex<double>(-1.0*g/(-mom->phat2(i) + mass*mass), 0.0);
		}
	}

return 1;
}

template<class T, int t > int lfield<T, t>::exponentiate(){

	#pragma omp parallel for simd default(shared)
	for(int i = 0; i < Nxl*Nyl; i++){
	
		su3_matrix<double> A;

		for(int k = 0; k < t; k++){

			A.m[k] = this->u[k][i];
		}
		
		A.exponentiate(1.0);

		for(int k = 0; k < t; k++){

			this->u[k][i] = A.m[k];
		}

	}

return 1;
}

template<class T, int t> int lfield<T, t>::exponentiate(double s){

	#pragma omp parallel for simd default(shared)
	for(int i = 0; i < Nxl*Nyl; i++){

		su3_matrix<double> A;

		for(int k = 0; k < t; k++){

			A.m[k] = s*(this->u[k][i]);
		}
		
		A.exponentiate(-1.0);

		for(int k = 0; k < t; k++){

			this->u[k][i] = A.m[k];
		}

	}

return 1;
}


template<class T, int t> int lfield<T,t>::setKernelPbarX(momenta* mom){

			//!pbar(dir,z,t)
                       	//tmpunit%su3(1,1) =  cmplx(0.0,&
                        //    &real(-1.0*(2.0*PI), kind=REALKND)*pbar(1,z+1,t+1)/phat2(z+1,t+1),kind=CMPLXKND)
                       	//tmpunit%su3(2,2) =  cmplx(0.0,&
                        //    &real(-1.0*(2.0*PI), kind=REALKND)*pbar(1,z+1,t+1)/phat2(z+1,t+1),kind=CMPLXKND)
                       	//tmpunit%su3(3,3) =  cmplx(0.0,&
                        //    &real(-1.0*(2.0*PI), kind=REALKND)*pbar(1,z+1,t+1)/phat2(z+1,t+1),kind=CMPLXKND)

                       	//tmpunita%su3 = matmul(tmpunit%su3, xi_local(ind,eo,2)%su3)
	if(t == 9){

	#pragma omp parallel for simd default(shared)
	for(int i = 0; i < Nxl*Nyl; i++){

		if( fabs(mom->phat2(i)) > 10e-9 ){

			this->u[0][i] = std::complex<double>(0.0, -2.0*M_PI*mom->pbarX(i)/mom->phat2(i));
			this->u[4][i] = this->u[0][i];
			this->u[8][i] = this->u[0][i];

		}else{

			this->u[0][i] = std::complex<double>(0.0, 0.0);
			this->u[4][i] = this->u[0][i];
			this->u[8][i] = this->u[0][i];

		}
	}

	}else{

		printf("Invalid lfield classes for setKernelPbarX function\n");

	}



return 1;
}

template<class T, int t> int lfield<T,t>::setKernelPbarY(momenta* mom){

			//!pbar(dir,z,t)
                       	//tmpunit%su3(1,1) =  cmplx(0.0,&
                        //    &real(-1.0*(2.0*PI), kind=REALKND)*pbar(1,z+1,t+1)/phat2(z+1,t+1),kind=CMPLXKND)
                       	//tmpunit%su3(2,2) =  cmplx(0.0,&
                        //    &real(-1.0*(2.0*PI), kind=REALKND)*pbar(1,z+1,t+1)/phat2(z+1,t+1),kind=CMPLXKND)
                       	//tmpunit%su3(3,3) =  cmplx(0.0,&
                        //    &real(-1.0*(2.0*PI), kind=REALKND)*pbar(1,z+1,t+1)/phat2(z+1,t+1),kind=CMPLXKND)

                       	//tmpunita%su3 = matmul(tmpunit%su3, xi_local(ind,eo,2)%su3)

	if(t == 9){

	#pragma omp parallel for simd default(shared)
	for(int i = 0; i < Nxl*Nyl; i++){

		if( fabs(mom->phat2(i)) > 10e-9 ){

			this->u[0][i] = std::complex<double>(0.0, -2.0*M_PI*mom->pbarY(i)/mom->phat2(i));
			this->u[4][i] = this->u[0][i];
			this->u[8][i] = this->u[0][i];

		}else{

			this->u[0][i] = std::complex<double>(0.0, 0.0);
			this->u[4][i] = this->u[0][i];
			this->u[8][i] = this->u[0][i];

		}
	}

	}else{

		printf("Invalid lfield classes for setKernelPbarY function\n");

	}



return 1;
}


template<class T, int t> int lfield<T,t>::setKernelPbarXWithCouplingConstant(momenta* mom){

			//!pbar(dir,z,t)
                       	//tmpunit%su3(1,1) =  cmplx(0.0,&
                        //    &real(-1.0*(2.0*PI), kind=REALKND)*pbar(1,z+1,t+1)/phat2(z+1,t+1),kind=CMPLXKND)
                       	//tmpunit%su3(2,2) =  cmplx(0.0,&
                        //    &real(-1.0*(2.0*PI), kind=REALKND)*pbar(1,z+1,t+1)/phat2(z+1,t+1),kind=CMPLXKND)
                       	//tmpunit%su3(3,3) =  cmplx(0.0,&
                        //    &real(-1.0*(2.0*PI), kind=REALKND)*pbar(1,z+1,t+1)/phat2(z+1,t+1),kind=CMPLXKND)

                       	//tmpunita%su3 = matmul(tmpunit%su3, xi_local(ind,eo,2)%su3)
	if(t == 9){


                        //coupling_constant = 4.0*real(PI,kind=REALKND)/((11.0-2.0*3.0/3.0)*&
                        //& log(((15.0**2/6.0**2)**(1.0/0.2) +&
                        //&(phat2(z+1,t+1)*zmax*zmax/(6.0**2))**(1.0/0.2))**(0.2)))


	#pragma omp parallel for simd default(shared)
	for(int i = 0; i < Nxl*Nyl; i++){

		if( mom->phat2(i) > 10e-9 ){

			double coupling_constant = 4.0*M_PI/( (11.0-2.0*3.0/3.0)*log( pow( pow(15.0*15.0/6.0/6.0,1.0/0.2) + pow((mom->phat2(i)*Nx*Ny)/6.0/6.0,1.0/0.2) , 0.2) ) );

			this->u[0][i] = std::complex<double>(0.0, -2.0*M_PI*sqrt(coupling_constant)*mom->pbarX(i)/mom->phat2(i));
			this->u[4][i] = this->u[0][i];
			this->u[8][i] = this->u[0][i];

		}else{

			this->u[0][i] = std::complex<double>(0.0, 0.0);
			this->u[4][i] = this->u[0][i];
			this->u[8][i] = this->u[0][i];

		}
	}

	}else{

		printf("Invalid lfield classes for setKernelPbarX function\n");

	}



return 1;
}

template<class T, int t> int lfield<T,t>::setKernelPbarYWithCouplingConstant(momenta* mom){

			//!pbar(dir,z,t)
                       	//tmpunit%su3(1,1) =  cmplx(0.0,&
                        //    &real(-1.0*(2.0*PI), kind=REALKND)*pbar(1,z+1,t+1)/phat2(z+1,t+1),kind=CMPLXKND)
                       	//tmpunit%su3(2,2) =  cmplx(0.0,&
                        //    &real(-1.0*(2.0*PI), kind=REALKND)*pbar(1,z+1,t+1)/phat2(z+1,t+1),kind=CMPLXKND)
                       	//tmpunit%su3(3,3) =  cmplx(0.0,&
                        //    &real(-1.0*(2.0*PI), kind=REALKND)*pbar(1,z+1,t+1)/phat2(z+1,t+1),kind=CMPLXKND)

                       	//tmpunita%su3 = matmul(tmpunit%su3, xi_local(ind,eo,2)%su3)

	if(t == 9){

	#pragma omp parallel for simd default(shared)
	for(int i = 0; i < Nxl*Nyl; i++){

		if( mom->phat2(i) > 10e-9 ){
	
			double coupling_constant = 4.0*M_PI/( (11.0-2.0*3.0/3.0)*log( pow( pow(15.0*15.0/6.0/6.0,1.0/0.2) + pow((mom->phat2(i)*Nx*Ny)/6.0/6.0,1.0/0.2) , 0.2) ) );

			this->u[0][i] = std::complex<double>(0.0, -2.0*M_PI*sqrt(coupling_constant)*mom->pbarY(i)/mom->phat2(i));
			this->u[4][i] = this->u[0][i];
			this->u[8][i] = this->u[0][i];

		}else{

			this->u[0][i] = std::complex<double>(0.0, 0.0);
			this->u[4][i] = this->u[0][i];
			this->u[8][i] = this->u[0][i];

		}
	}

	}else{

		printf("Invalid lfield classes for setKernelPbarY function\n");

	}



return 1;
}

template<class T, int t> int gfield<T,t>::setKernelXbarX(int x_global, int y_global, positions* pos){

	if(t == 9){

	//for(int i = 0; i < Nxl*Nyl; i++){
	#pragma omp parallel for simd collapse(2) default(shared)
	for(int xx = 0; xx < Nxg; xx++){
		for(int yy = 0; yy < Nyg; yy++){

			int i = xx*Nyg+yy;
		
			int ii = fabs(x_global - xx)*Nyg + fabs(y_global - yy);

                        //double dx2 = 0.5*Nxg*sin(2.0*M_PI*(x_global-xx)/Nxg)/M_PI;
                        //double dy2 = 0.5*Nyg*sin(2.0*M_PI*(y_global-yy)/Nyg)/M_PI;
/*
                        double dx = pos->xhatX(ii); //Nxg*sin(M_PI*(x_global-xx)/Nxg)/M_PI;
                        //double dy = pos->xbarY(ii); //Nyg*sin(M_PI*(y_global-yy)/Nyg)/M_PI;

                        double rrr = pos->xbar2(ii); //1.0*(dx2*dx2+dy2*dy2);
*/
	                                 double dx = x_global - xx;
                                         if( dx >= Nxg/2 )
                                                dx = dx - Nxg;
                                         if( dx < -Nxg/2 )
                                                dx = dx + Nxg;
                                         
                                         double dy = y_global - yy;
                                         if( dy >= Nyg/2 )
                                                dy = dy - Nyg;
                                         if( dy < -Nyg/2 )
                                                dy = dy + Nyg;

                                         double rrr = 1.0*(dx*dx+dy*dy);

			if( rrr > 10e-9 ){

				this->u[0][i] = std::complex<double>(dx/rrr, 0.0);
				this->u[4][i] = this->u[0][i];
				this->u[8][i] = this->u[0][i];

			}//else{
			//
			//	this->u[0][i] = std::complex<double>(0.0, 0.0);
			//	this->u[4][i] = this->u[0][i];
			//	this->u[8][i] = this->u[0][i];
			//
			//}
		}
	}

	}else{

		printf("Invalid lfield classes for setKernelPbarX function\n");

	}

return 1;
}

template<class T, int t> int gfield<T,t>::setKernelXbarY(int x_global, int y_global, positions* pos){

	if(t == 9){

	#pragma omp parallel for simd collapse(2) default(shared)
	for(int xx = 0; xx < Nxg; xx++){
		for(int yy = 0; yy < Nxg; yy++){

			int i = xx*Nyg+yy;

			int ii = fabs(x_global - xx)*Nyg + fabs(y_global - yy);

                        //double dx2 = 0.5*Nxg*sin(2.0*M_PI*(x_global-xx)/Nxg)/M_PI;
                        //double dy2 = 0.5*Nyg*sin(2.0*M_PI*(y_global-yy)/Nyg)/M_PI;
/*
                        //double dx = pos->xbarX(ii); //Nxg*sin(M_PI*(x_global-xx)/Nxg)/M_PI;
                        double dy = pos->xhatY(ii); //Nyg*sin(M_PI*(y_global-yy)/Nyg)/M_PI;

                        double rrr = pos->xbar2(ii); //1.0*(dx2*dx2+dy2*dy2);
*/
                                         double dx = x_global - xx;
                                         if( dx >= Nxg/2 )
                                                dx = dx - Nxg;
                                         if( dx < -Nxg/2 )
                                                dx = dx + Nxg;
                                         
                                         double dy = y_global - yy;
                                         if( dy >= Nyg/2 )
                                                dy = dy - Nyg;
                                         if( dy < -Nyg/2 )
                                                dy = dy + Nyg;

                                         double rrr = 1.0*(dx*dx+dy*dy);

			if( rrr > 10e-9 ){

				this->u[0][i] = std::complex<double>(dy/rrr, 0.0);
				this->u[4][i] = this->u[0][i];
				this->u[8][i] = this->u[0][i];

			}//else{
			//
			//	this->u[0][i] = std::complex<double>(0.0, 0.0);
			//	this->u[4][i] = this->u[0][i];
			//	this->u[8][i] = this->u[0][i];
			//
			//}
		}
	}

	}else{

		printf("Invalid lfield classes for setKernelPbarY function\n");

	}



return 1;
}


template<class T, int t> int gfield<T,t>::setKernelXbarXWithCouplingConstant(int x_global, int y_global, positions* postable){

	if(t == 9){

	//for(int i = 0; i < Nxl*Nyl; i++){
	#pragma omp parallel for simd collapse(2) default(shared)
	for(int xx = 0; xx < Nxg; xx++){
		for(int yy = 0; yy < Nyg; yy++){

			int i = xx*Nyg+yy;

                                         //coupling_constant = &
                                         //& 4.0*real(PI,kind=REALKND)/((11.0-2.0*3.0/3.0)*&
                                         //& log(((15.0**2/6.0**2)**(1.0/0.2) +&
                                         //& ((1.26)/(((1.0*dt**2+1.0*dz**2)/(1.0*zmax**2))*6.0**2)**(1.0/0.2)))**(0.2)))

                        double dx2 = 2.0*Nxg*sin(0.5*M_PI*(x_global-xx)/Nxg)/M_PI;
                        double dy2 = 2.0*Nyg*sin(0.5*M_PI*(y_global-yy)/Nyg)/M_PI;

                        double dx = Nxg*sin(M_PI*(x_global-xx)/Nxg)/M_PI;
                        double dy = Nyg*sin(M_PI*(y_global-yy)/Nyg)/M_PI;

			double coupling_constant = 4.0*M_PI/(  (11.0-2.0*3.0/3.0) * log( pow( pow(15.0*15.0/6.0/6.0,1.0/0.2) + 1.26/pow(6.0*6.0*(dx*dx+dy*dy)/Nxg/Nyg,1.0/0.2) , 0.2 ) ));
           	        double rrr = 1.0*(dx2*dx2+dy2*dy2);

			if( rrr > 10e-9 ){

				this->u[0][i] = std::complex<double>(0.0, sqrt(coupling_constant)*dx/rrr);
				this->u[4][i] = this->u[0][i];
				this->u[8][i] = this->u[0][i];

			}else{
	
				this->u[0][i] = std::complex<double>(0.0, 0.0);
				this->u[4][i] = this->u[0][i];
				this->u[8][i] = this->u[0][i];
	
			}
		}
	}

	}else{

		printf("Invalid lfield classes for setKernelPbarX function\n");

	}

return 1;
}

template<class T, int t> int gfield<T,t>::setKernelXbarYWithCouplingConstant(int x_global, int y_global, positions* postable){

	if(t == 9){

	#pragma omp parallel for simd collapse(2) default(shared)
	for(int xx = 0; xx < Nxg; xx++){
		for(int yy = 0; yy < Nxg; yy++){

			int i = xx*Nyg+yy;

	                double dx2 = 2.0*Nxg*sin(0.5*M_PI*(x_global-xx)/Nxg)/M_PI;
                        double dy2 = 2.0*Nyg*sin(0.5*M_PI*(y_global-yy)/Nyg)/M_PI;

                        double dx = Nxg*sin(M_PI*(x_global-xx)/Nxg)/M_PI;
                        double dy = Nyg*sin(M_PI*(y_global-yy)/Nyg)/M_PI;

			double coupling_constant = 4.0*M_PI/(  (11.0-2.0*3.0/3.0) * log( pow( pow(15.0*15.0/6.0/6.0,1.0/0.2) + 1.26/pow(6.0*6.0*(dx*dx+dy*dy)/Nxg/Nyg,1.0/0.2) , 0.2 ) ));
        
                        double rrr = 1.0*(dx2*dx2+dy2*dy2);

			if( rrr > 10e-9 ){

				this->u[0][i] = std::complex<double>(0.0, sqrt(coupling_constant)*dy/rrr);
				this->u[4][i] = this->u[0][i];
				this->u[8][i] = this->u[0][i];

			}else{

				this->u[0][i] = std::complex<double>(0.0, 0.0);
				this->u[4][i] = this->u[0][i];
				this->u[8][i] = this->u[0][i];

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

		result->u[0][i] = std::conj(this->u[0][i]);
		result->u[1][i] = std::conj(this->u[3][i]);
		result->u[2][i] = std::conj(this->u[6][i]);
		result->u[3][i] = std::conj(this->u[1][i]);
		result->u[4][i] = std::conj(this->u[4][i]);
		result->u[5][i] = std::conj(this->u[7][i]);
		result->u[6][i] = std::conj(this->u[2][i]);
		result->u[7][i] = std::conj(this->u[5][i]);
		result->u[8][i] = std::conj(this->u[8][i]);

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

		result->u[0][i] = std::conj(this->u[0][i]);
		result->u[1][i] = std::conj(this->u[3][i]);
		result->u[2][i] = std::conj(this->u[6][i]);
		result->u[3][i] = std::conj(this->u[1][i]);
		result->u[4][i] = std::conj(this->u[4][i]);
		result->u[5][i] = std::conj(this->u[7][i]);
		result->u[6][i] = std::conj(this->u[2][i]);
		result->u[7][i] = std::conj(this->u[5][i]);
		result->u[8][i] = std::conj(this->u[8][i]);

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
                        A.m[k] = this->u[k][i];
                }

		B.m[0] = std::conj(this->u[0][i]);
		B.m[1] = std::conj(this->u[3][i]);
		B.m[2] = std::conj(this->u[6][i]);
		B.m[3] = std::conj(this->u[1][i]);
		B.m[4] = std::conj(this->u[4][i]);
		B.m[5] = std::conj(this->u[7][i]);
		B.m[6] = std::conj(this->u[2][i]);
		B.m[7] = std::conj(this->u[5][i]);
		B.m[8] = std::conj(this->u[8][i]);

		//printf("A.m[0].r = %f, A.m[1].r = %f, A.m[2].r = %f\n", A.m[0].real(), A.m[1].real(), A.m[2].real());
		//printf("A.m[0].i = %f, A.m[1].i = %f, A.m[2].i = %f\n", A.m[0].imag(), A.m[1].imag(), A.m[2].imag());
		//printf("A.m[3].r = %f, A.m[4].r = %f, A.m[5].r = %f\n", A.m[3].real(), A.m[4].real(), A.m[5].real());
		//printf("A.m[3].i = %f, A.m[4].i = %f, A.m[5].i = %f\n", A.m[3].imag(), A.m[4].imag(), A.m[5].imag());
		//printf("A.m[6].r = %f, A.m[7].r = %f, A.m[8].r = %f\n", A.m[6].real(), A.m[7].real(), A.m[8].real());
		//printf("A.m[6].i = %f, A.m[7].i = %f, A.m[8].i = %f\n", A.m[6].imag(), A.m[7].imag(), A.m[8].imag());

		//printf("B.m[0].r = %f, B.m[4].r = %f, B.m[8].r = %f\n", B.m[0].real(), B.m[4].real(), B.m[8].real());
		//printf("B.m[0].i = %f, B.m[4].i = %f, B.m[8].i = %f\n", B.m[0].imag(), B.m[4].imag(), B.m[8].imag());

                C = A*B;

		//printf("C.m[0].r = %f, C.m[4].r = %f, C.m[8].r = %f\n", C.m[0].real(), C.m[4].real(), C.m[8].real());
		//printf("C.m[0].i = %f, C.m[4].i = %f, C.m[8].i = %f\n", C.m[0].imag(), C.m[4].imag(), C.m[8].imag());

                cc->u[0][i] = C.m[0] + C.m[4] + C.m[8];
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
                         cc->u[0][i] = this->u[0][i];

	}

	}else{

		printf("Invalid lfield classes for trace function\n");

	}


return 1;
}



template<class T, int t> int gfield<T,t>::average_and_symmetrize(void){

	printf("avegare_and_symmetrize: t = %i\n", t);

	gfield<T,t>* corr_tmp = new gfield(Nx,Ny);

	corr_tmp->setToZero();
	
	#pragma omp parallel for simd collapse(2) default(shared)
	for(int i = 0; i < Nx; i++){
		for(int j = 0; j < Ny; j++){
			corr_tmp->u[0][i*Ny+j] = 0.5*(this->u[0][i*Ny+j] + this->u[0][j*Ny+i]);
		}
	}

	#pragma omp parallel for simd collapse(2) default(shared)
	for(int i = 0; i < Nx; i++){
		for(int j = 0; j < Ny; j++){
			
			int a = abs(i-Nx)%Nx;
			int b = abs(j-Ny)%Ny;

			this->u[0][i*Ny+j] = 0.25*(	corr_tmp->u[0][i*Ny+j] +
							corr_tmp->u[0][a*Ny+j] + 
							corr_tmp->u[0][i*Ny+b] + 
							corr_tmp->u[0][a*Ny+b]);
		}
	}

	delete corr_tmp;

return 1;
}


template<class T, int t> lfield<T,t>* gfield<T,t>::reduce(int NNx, int NNy, mpi_class* mpi){

//	printf("Nxg = %i, Nyg = %i, NNx = %i, NNy = %i\n", Nxg, Nyg, NNx, NNy);
//	printf("pos_x = %i, pox_y = %i\n", mpi->getPosX(), mpi->getPosY());


	lfield<T,t>* corr_tmp = new lfield<T,t>(NNx,NNy);

//	std::complex<double> x = 0;
	corr_tmp->setToZero();	
//	printf("reduce: t = %i\n", t);

	#pragma omp parallel for simd collapse(2) default(shared)
	for(int i = 0; i < NNx; i++){
		for(int j = 0; j < NNy; j++){
			//x += this->u[(i+mpi->getPosX()*NNx)*Nyg+j+mpi->getPosY()*NNy][0];
			corr_tmp->u[0][i*NNy+j] = this->u[0][(i+mpi->getPosX()*NNx)*Ny+j+mpi->getPosY()*NNy];
		}
	}

//	printf("x = %e, %e\n", x.real(), x.imag());

return corr_tmp;
}

template<class T, int t> int lfield<T,t>::reduceAndSet(int x_local, int y_local, gfield<T,t>* f){

		for(int k = 0; k < t; k++){

		double sum_re = 0;
		double sum_im = 0;

		#pragma omp parallel for collapse(2) default(shared) reduction(+:sum_re) reduction(+:sum_im)
		for(int xx = 0; xx < f->Nxg; xx++){
			for(int yy = 0; yy < f->Nyg; yy++){

				sum_re += f->u[k][xx*f->Nyg+yy].real();		
				sum_im += f->u[k][xx*f->Nyg+yy].imag();		

			}
		}

		this->u[k][x_local*Nyl+y_local] = std::complex<double>(sum_re, sum_im);

		}

return 1;
}

template<class T, int t> int gfield<T,t>::setCorrelationsForCouplingConstant(){

	#pragma omp parallel for simd collapse(2) default(shared)
	for(int xx = 0; xx < Nxg; xx++){
		for(int yy = 0; yy < Nxg; yy++){

			int i = xx*Nyg+yy;

                        double dx = Nxg*sin(M_PI*(xx)/Nxg)/M_PI;
                        double dy = Nyg*sin(M_PI*(yy)/Nyg)/M_PI;

			double coupling_constant = 4.0*M_PI/(  (11.0-2.0*3.0/3.0) * log( pow( pow(15.0*15.0/6.0/6.0,1.0/0.2) + 1.26/pow(6.0*6.0*(dx*dx+dy*dy)/Nxg/Nyg,1.0/0.2) , 0.2 ) ));
        
			this->u[0][i] = coupling_constant;
		}
	}

return 1;
}

template<class T, int t> int lfield<T,t>::setCorrelationsForCouplingConstant(momenta* mom){

	#pragma omp parallel for simd collapse(2) default(shared)
	for(int xx = 0; xx < Nxl; xx++){
		for(int yy = 0; yy < Nxl; yy++){

			int i = xx*Nyl+yy;

			double coupling_constant = 4.0*M_PI/( (11.0-2.0*3.0/3.0)*log( pow( pow(15.0*15.0/6.0/6.0,1.0/0.2) + pow((mom->phat2(i)*Nx*Ny)/6.0/6.0,1.0/0.2) , 0.2) ) );
       
			this->u[0][i] = coupling_constant;
		}
	}

return 1;
}

template<class T, int t> int gfield<T,t>::multiplyByCholesky(gmatrix<T>* mm){

	gfield<T,t>* tmp = new gfield<T,t>(Nxg, Nyg); 

	for(int tt = 0; tt < t; tt++){
	
		for(int xx = 0; xx < Nxg; xx++){
			for(int yy = 0; yy < Nyg; yy++){
	
				tmp->u[tt][xx*Nyg+yy] = 0;

				for(int xxi = 0; xxi < Nxg; xxi++){
					for(int yyi = 0; yyi < Nyg; yyi++){
	
						tmp->u[tt][xx*Nyg+yy] += mm->u[(xx*Nyg+yy)*Nxg*Nyg + xxi*Nyg+yyi] * this->u[tt][xxi*Nyg+yyi];

					}
				}
			}
		}

		for(int xx = 0; xx < Nxg; xx++){
			for(int yy = 0; yy < Nyg; yy++){
	
				this->u[tt][xx*Nyg+yy] = tmp->u[tt][xx*Nyg+yy];

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
//			if((yy+xx)%2 == 0)       
			//printf("%f %f %f\n", Nx*Ny*(mom->phat2(i)), mom->phat2(i)*this->u[0][i].real(), mom->phat2(i)*this->u[0][i].imag());
			printf("%i %i %f %f\n", xx+1, yy+1, this->u[k][i].real(), this->u[k][i].imag());

		}
	}
/*
	for(int xx = 0; xx < Nxl; xx++){
		for(int yy = 0; yy < Nxl; yy++){

			int i = yy*Nyl+xx;
       			if((xx+yy)%2==1)
			//printf("%f %f %f\n", Nx*Ny*(mom->phat2(i)), mom->phat2(i)*this->u[0][i].real(), mom->phat2(i)*this->u[0][i].imag());
			printf("%i %i %f %f\n", xx+1, yy+1, this->u[k][i].real(), this->u[k][i].imag());

		}
	}
*/
	}
	
return 1;
}

template<class T, int t> int lfield<T,t>::print(momenta* mom, double x, mpi_class* mpi){

	for(int xx = 0; xx < Nxl; xx++){
		for(int yy = 0; yy < Nyl; yy++){

			int i = xx*Nyl+yy;
      
			if( fabs(xx + mpi->getPosX()*Nxl - yy - mpi->getPosY()*Nyl) <= 4 ){
 
				printf("%i %i %i %i %f %e %e\n", xx, mpi->getPosX(), yy, mpi->getPosY(), sqrt(mom->phat2(i)), x*(mom->phat2(i))*(this->u[0][i].real()), x*(mom->phat2(i))*(this->u[0][i].imag()));

			}
		}
	}

return 1;
}

template<class T, int t> int lfield<T,t>::printDebug(){

	for(int xx = 0; xx < Nxl; xx++){
		for(int yy = 0; yy < Nxl; yy++){

			int i = xx*Nyl+yy;
       
			printf("%i %i %f %f\n",xx, yy, this->u[0][i].real(), this->u[0][i].imag());
		}
	}

return 1;
}

template<class T, int t> int lfield<T,t>::printDebug(double x, mpi_class* mpi){

	for(int xx = 0; xx < Nxl; xx++){
		for(int yy = 0; yy < Nyl; yy++){

			int i = xx*Nyl+yy;
       
			printf("%i %i %f %f\n",xx+mpi->getPosX()*Nxl, yy+mpi->getPosY()*Nyl, x*(this->u[0][i].real()), x*(this->u[0][i].imag()));
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

			printf("%i %i %f %f\n",xx, yy, z*(this->u[0][i].real()), z*(this->u[0][i].imag()));
		}
	}

return 1;
}


#endif
