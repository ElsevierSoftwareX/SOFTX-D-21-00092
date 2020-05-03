#ifndef H_FIELD
#define H_FIELD

#include <iostream>
#include <stdlib.h>
#include <complex>
#include <complex.h>
#include "config.h"

#include <omp.h>

#include <mpi.h>
#include <math.h>

#include "mpi_class.h"

#include "MV_class.h"
#include "rand_class.h"

#include "momenta.h"

template<class T> class field {

	protected:

		std::complex<T>* u[9];		

		int Nxl, Nyl;

	public:

		field(int NNx, int NNy);

		~field(void);
};

template<class T> field<T>::field(int NNx, int NNy) {

	int i;

	for(i = 0; i < 9; i++){

		u[i] = (std::complex<T>*)malloc(NNx*NNy*sizeof(std::complex<T>));

	}

	Nxl = NNx;
	Nyl = NNy;
}

template<class T> field<T>::~field() {

	int i;

	for(i = 0; i < 9; i++){

		free(u[i]);

	}

}

template<class T> class lfield;


template<class T> class gfield: public field<T> {

	int Nxg, Nyg;

	public:

		//T getZero(void){ return this.u[0][0]; }

		int allgather(lfield<T>* ulocal);

		gfield(int NNx, int NNy) : field<T>{NNx, NNy} { Nxg = NNx; Nyg = NNy;};

		friend class fftw2D;
};


template<class T> class lfield: public field<T> {

	int Nxl, Nyl;
	int Nxl_buf, Nyl_buf;

	public:

		lfield(int NNx, int NNy) : field<T>{NNx, NNy} { Nxl = NNx; Nyl = NNy; };

		friend class fftw1D;

		int mpi_exchange_boundaries(mpi_class* mpi);

		friend class gfield<T>;

		int loc_pos(int x, int y){

			return y + Nyl*x;
		}

		int buf_pos(int x, int y){

			return ((y+(Nyl_buf-Nyl)/2) + Nyl_buf*(x+(Nxl_buf-Nxl)/2));
		}

		int buf_pos_ex(int x, int y){

			return (y + Nyl_buf*x);
		}

		int setMVModel(MV_class* MVconfig, rand_class* rr);

		int setGaussian(rand_class* rr);

		int solvePoisson(double mass, double g, momenta* momtable);

		int exponentiate();

		int operator *= ( lfield<T> &f ){

			for(int i = 0; i < Nxl*Nyl; i ++){
			
				su3_matrix<double> A,B,C;

				for(int k = 0; k < 9; k++){

					A.m[k] = this->u[k][i];
					B.m[k] = f.u[k][i];
				}
		
				C = A*B;

				for(int k = 0; k < 9; k++){

					this->u[k][i] = C.m[k];
				}
			}

		return 1;
		}
};


template<class T> int gfield<T>::allgather(lfield<T>* ulocal){


	T* data_local_re = (T*)malloc(ulocal->Nxl*ulocal->Nyl*sizeof(T));
	T* data_local_im = (T*)malloc(ulocal->Nxl*ulocal->Nyl*sizeof(T));

	T* data_global_re = (T*)malloc(Nxg*Nyg*sizeof(T));
	T* data_global_im = (T*)malloc(Nxg*Nyg*sizeof(T));

	int i,k;

	for(k = 0; k < 9; k++){

		for(i = 0; i < ulocal->Nxl*ulocal->Nyl; i++){

			data_local_re[i] = ulocal->u[k][i].real();
			data_local_im[i] = ulocal->u[k][i].imag();

		}

   		MPI_Allgather(data_local_re, ulocal->Nxl*ulocal->Nyl, MPI_DOUBLE, data_global_re, ulocal->Nxl*ulocal->Nyl, MPI_DOUBLE, MPI_COMM_WORLD); 
	   	MPI_Allgather(data_local_im, ulocal->Nxl*ulocal->Nyl, MPI_DOUBLE, data_global_im, ulocal->Nxl*ulocal->Nyl, MPI_DOUBLE, MPI_COMM_WORLD); 

		for(i = 0; i < Nxg*Nyg; i++){

			this->u[k][i] = data_global_re[i] + I*data_global_im[i];
	
		}
	}

	return 1;
}

template<class T> int lfield<T>::mpi_exchange_boundaries(mpi_class* mpi){

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

template<class T> int lfield<T>::setMVModel(MV_class* MVconfig, rand_class* rr){

	printf("SETTING MV MODEL: Nxl = %i, Nyl = %i\n", Nxl, Nyl);

	int myrank;
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

	const double EPS = 10e-12;

	// 0 1 2
	// 3 4 5
	// 6 7 8

        double n[8];

	for(int i = 0; i < Nxl*Nyl; i++){


//				hh=sqrt(real(1.0*g_parameter**2*mu_parameter**2/Ny_parameter, kind=REALKND)) * &
//                              & sqrt((real(-2.0, kind=REALKND))*log(EPSI+real(ranvec(2*m-1),kind=REALKND))) * &
//                              & cos(real(ranvec(2*m),kind=REALKND) * real(TWOPI, kind=REALKND))

		for(int k = 0; k < 8; k++)
                	n[k] = sqrt( pow(MVconfig->g_parameter,2.0) * pow(MVconfig->mu_parameter,2.0) / MVconfig->Ny_parameter ) * sqrt( -2.0 * log( EPS + rr->get() ) ) * cos( rr->get() * 2.0 * M_PI);
		
	 //these are the LAMBDAs and not the generators t^a = lambda/2.

            //lambda_nr(1)%su3(1,2) =  runit
            //lambda_nr(1)%su3(2,1) =  runit
		this->u[1][i] += std::complex<double>(n[0],0.0);
		this->u[3][i] += std::complex<double>(n[0],0.0);


            //lambda_nr(2)%su3(1,2) = -iunit
            //lambda_nr(2)%su3(2,1) =  iunit
		this->u[1][i] -= std::complex<double>(0.0,n[1]);
		this->u[3][i] += std::complex<double>(0.0,n[1]);


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
		this->u[2][i] -= std::complex<double>(0.0,n[4]);
		this->u[6][i] += std::complex<double>(0.0,n[4]);


            //lambda_nr(6)%su3(2,3) =  runit
            //lambda_nr(6)%su3(3,2) =  runit
		this->u[5][i] += std::complex<double>(n[5],0.0);
		this->u[7][i] += std::complex<double>(n[5],0.0);


            //lambda_nr(7)%su3(2,3) = -iunit
            //lambda_nr(7)%su3(3,2) =  iunit
		this->u[5][i] -= std::complex<double>(0.0,n[6]);
		this->u[7][i] += std::complex<double>(0.0,n[6]);


            //lambda_nr(8)%su3(1,1) =  cst8
            //lambda_nr(8)%su3(2,2) =  cst8
            //lambda_nr(8)%su3(3,3) =  -(two*cst8)

		this->u[0][i] += std::complex<double>(n[7]/sqrt(3.0),0.0);
		this->u[4][i] += std::complex<double>(n[7]/sqrt(3.0),0.0);
		this->u[8][i] += std::complex<double>(2.0*n[7]/sqrt(3.0),0.0);
	}

return 1;
}

template<class T> int lfield<T>::setGaussian(rand_class* rr){

	const double EPS = 10e-12;

	// 0 1 2
	// 3 4 5
	// 6 7 8

        double n[8];

	for(int i = 0; i < Nxl*Nyl; i++){


		double coupling_constant = 1.0;

//				hh=sqrt(real(coupling_constant, kind=REALKND)) * &
//                              & sqrt((real(-2.0, kind=REALKND))*log(EPSI+real(ranvec(2*m-1),kind=REALKND))) * &
//                              & cos(real(ranvec(2*m),kind=REALKND) * real(TWOPI, kind=REALKND))

		for(int k = 0; k < 8; k++)
                	n[k] = sqrt( coupling_constant ) * sqrt( -2.0 * log( EPS + rr->get() ) ) * cos( rr->get() * 2.0 * M_PI);
		
	 //these are the LAMBDAs and not the generators t^a = lambda/2.

            //lambda_nr(1)%su3(1,2) =  runit
            //lambda_nr(1)%su3(2,1) =  runit
		this->u[1][i] += std::complex<double>(n[0],0.0);
		this->u[3][i] += std::complex<double>(n[0],0.0);


            //lambda_nr(2)%su3(1,2) = -iunit
            //lambda_nr(2)%su3(2,1) =  iunit
		this->u[1][i] -= std::complex<double>(0.0,n[1]);
		this->u[3][i] += std::complex<double>(0.0,n[1]);


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
		this->u[2][i] -= std::complex<double>(0.0,n[4]);
		this->u[6][i] += std::complex<double>(0.0,n[4]);


            //lambda_nr(6)%su3(2,3) =  runit
            //lambda_nr(6)%su3(3,2) =  runit
		this->u[5][i] += std::complex<double>(n[5],0.0);
		this->u[7][i] += std::complex<double>(n[5],0.0);


            //lambda_nr(7)%su3(2,3) = -iunit
            //lambda_nr(7)%su3(3,2) =  iunit
		this->u[5][i] -= std::complex<double>(0.0,n[6]);
		this->u[7][i] += std::complex<double>(0.0,n[6]);


            //lambda_nr(8)%su3(1,1) =  cst8
            //lambda_nr(8)%su3(2,2) =  cst8
            //lambda_nr(8)%su3(3,3) =  -(two*cst8)

		this->u[0][i] += std::complex<double>(n[7]/sqrt(3.0),0.0);
		this->u[4][i] += std::complex<double>(n[7]/sqrt(3.0),0.0);
		this->u[8][i] += std::complex<double>(2.0*n[7]/sqrt(3.0),0.0);
	}

return 1;
}


template<class T> int lfield<T>::solvePoisson(double mass, double g, momenta* mom){

		//u(ind,eo)%su3 = cmplx(-1.0*g_parameter, 0.0, kind=CMPLXKND)*&
                //&u(ind,eo)%su3/(-phat2(z+1,t+1) + mass_parameter**2)

	for(int i = 0; i < Nxl*Nyl; i++){
	
		for(int k = 0; k < 9; k++){

			this->u[k][i] *= std::complex<double>(-1.0*g/(mom->phat2(i) + mass*mass), 0.0);

		}
	}

return 1;
}


template<class T> int lfield<T>::exponentiate(){


	for(int i = 0; i < Nxl*Nyl; i++){
	
		su3_matrix<double> A;

		for(int k = 0; k < 9; k++){

			A.m[k] = this->u[k][i];
		}
		
		A.exponentiate();

		for(int k = 0; k < 9; k++){

			this->u[k][i] = A.m[k];
		}

	}

return 1;
}

#endif
