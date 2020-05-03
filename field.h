#ifndef H_FIELD
#define H_FIELD

#include <iostream>
#include <stdlib.h>
#include <complex>
#include <complex.h>
#include "config.h"

#include "mpi_pos.h"

#include <omp.h>

#include <mpi.h>

#include "mpi_class.h"

template<class T> class field {

	protected:

		std::complex<T>* u[9];		

	public:

		field(int NNx, int NNy);

		~field(void);
};

template<class T> field<T>::field(int NNx, int NNy) {

	int i;

	for(i = 0; i < 9; i++){

		u[i] = (std::complex<T>*)malloc(NNx*NNy*sizeof(std::complex<T>));

	}
}

template<class T> field<T>::~field() {

	int i;

	for(i = 0; i < 9; i++){

		free(u[i]);

	}

}

template<class T> class lfield;


template<class T> class gfield: public field<T> {

	public:

		//T getZero(void){ return this.u[0][0]; }

		int allgather(lfield<T> ulocal);

		gfield(int NNx, int NNy) : field<T>{NNx, NNy} {};

};


template<class T> class lfield: public field<T> {

	public:

		lfield(int NNx, int NNy) : field<T>{NNx, NNy} {};

		friend class fftw;

		int mpi_exchange_boundaries(mpi_class* mpi);

		friend class gfield<T>;

};


template<class T> int gfield<T>::allgather(lfield<T> ulocal){


	static T data_local_re[Nxl*Nyl];
	static T data_local_im[Nxl*Nyl];

	static T data_global_re[Nx*Ny];
	static T data_global_im[Nx*Ny];

	int i,k;

	for(k = 0; k < 9; k++){

		for(i = 0; i < Nxl*Nyl; i++){

			data_local_re[i] = ulocal.u[k][i].real();
			data_local_im[i] = ulocal.u[k][i].imag();

		}

   		MPI_Allgather(data_local_re, Nxl*Nyl, MPI_DOUBLE, data_global_re, Nxl*Nyl, MPI_DOUBLE, MPI_COMM_WORLD); 
	   	MPI_Allgather(data_local_im, Nxl*Nyl, MPI_DOUBLE, data_global_im, Nxl*Nyl, MPI_DOUBLE, MPI_COMM_WORLD); 

		for(i = 0; i < Nx*Ny; i++){

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


    if( ExchangeX == 1 ){

	    int yy; 

	    bufor_send_n = (double*) malloc(Nyl_buf*sizeof(double));
	    bufor_receive_n = (double*) malloc(Nyl_buf*sizeof(double));

  	    for(yy = 0; yy < Nyl; yy++){
		bufor_send_n[yy] = this->u[0][buf_pos(Nxl-1,yy)].real();
	    }

	    printf("X data exchange: rank %i sending to %i\n", mpi->getRank(), XNeighbourNext);
	    printf("X data exchange: rank %i receiving from %i\n", mpi->getRank(), XNeighbourNext);

	    MPI_Send(bufor_send_n, Nyl_buf, MPI_DOUBLE, XNeighbourNext, 11, MPI_COMM_WORLD);
    	    MPI_Recv(bufor_receive_n, Nyl_buf, MPI_DOUBLE, XNeighbourPrevious, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  	    for(yy = 0; yy < Nyl; yy++){
		this->u[0][buf_pos_ex(0,yy)] = bufor_receive_n[yy];
	    }

   	    bufor_send_p = (double*) malloc(Nyl_buf*sizeof(double));
	    bufor_receive_p = (double*) malloc(Nyl_buf*sizeof(double));

	    for(yy = 0; yy < Nyl; yy++){
		bufor_send_p[yy] = this->u[0][buf_pos(0,yy)].real();
	    }
	
 	    printf("X data exchange: rank %i sending to %i\n", mpi->getRank(), XNeighbourPrevious);
	    printf("X data exchange: rank %i receiving to %i\n", mpi->getRank(), XNeighbourPrevious);

	    MPI_Send(bufor_send_p, Nyl_buf, MPI_DOUBLE, XNeighbourPrevious, 12, MPI_COMM_WORLD);
    	    MPI_Recv(bufor_receive_p, Nyl_buf, MPI_DOUBLE, XNeighbourNext, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  	    for(yy = 0; yy < Nyl; yy++){
		this->u[0][buf_pos_ex(Nxl+1,yy)] = bufor_receive_p[yy];
	    }
    }

    if( ExchangeY == 1 ){

	    int xx; 

	    bufor_send_n = (double*) malloc(Nxl_buf*sizeof(double));
	    bufor_receive_n = (double*) malloc(Nxl_buf*sizeof(double));

  	    for(xx = 0; xx < Nxl; xx++){
		bufor_send_n[xx] = this->u[0][buf_pos(xx,Nyl-1)].real();
	    }

	    printf("Y data exchange: rank %i sending to %i\n", mpi->getRank(), YNeighbourNext);
	    printf("Y data exchange: rank %i receiving from %i\n", mpi->getRank(), YNeighbourNext);

	    MPI_Send(bufor_send_n, Nxl_buf, MPI_DOUBLE, YNeighbourNext, 13, MPI_COMM_WORLD);
    	    MPI_Recv(bufor_receive_n, Nxl_buf, MPI_DOUBLE, YNeighbourPrevious, 13, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for(xx = 0; xx < Nxl; xx++){
		this->u[0][buf_pos_ex(xx,0)] = bufor_receive_n[xx];
	    }

	    bufor_send_p = (double*) malloc(Nxl_buf*sizeof(double));
	    bufor_receive_p = (double*) malloc(Nxl_buf*sizeof(double));

	    for(xx = 0; xx < Nxl; xx++){
		bufor_send_p[xx] = this->u[0][buf_pos(xx,0)].real();
	    }
	
 	    printf("Y data exchange: rank %i sending to %i\n", mpi->getRank(), YNeighbourPrevious);
	    printf("Y data exchange: rank %i receiving to %i\n", mpi->getRank(), YNeighbourPrevious);

	    MPI_Send(bufor_send_p, Nxl_buf, MPI_DOUBLE, YNeighbourPrevious, 14, MPI_COMM_WORLD);
    	    MPI_Recv(bufor_receive_p, Nxl_buf, MPI_DOUBLE, YNeighbourNext, 14, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  	    for(xx = 0; xx < Nxl; xx++){
		this->u[0][buf_pos_ex(xx,Nyl+1)] = bufor_receive_p[xx];
	    }
    }

return 1;
}

#endif
