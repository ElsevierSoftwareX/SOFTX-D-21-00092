#include <iostream>
#include <stdlib.h>
#include <complex>
#include <complex.h>
#include "config.h"

#include "mpi_pos.h"

#include <omp.h>

#include <mpi.h>


template<class T> class field_global{

	private:

		std::complex<T>* u[9];		

	public:

		field_global(int NNx, int NNy);

		~field_global(void);
};

template<class T> field_global<T>::field_global(int NNx, int NNy){

	int i;

	for(i = 0; i < 9; i++){

		u[i] = (std::complex<T>*)malloc(NNx*NNy*sizeof(std::complex<T>));

	}
}

template<class T> field_global<T>::~field_global(){

	int i;

	for(i = 0; i < 9; i++){

		free(u[i]);

	}

}


template<class T> class field {

	private:

		std::complex<T>* u[9];		

	public:

		field(int NNxl, int NNyl);

		~field(void);

		friend class fftw;

		int mpi_exchange_boundaries(void);

		int allgather(field_global<T> u);
		
};

template<class T> field<T>::field(int NNxl, int NNyl){

	int i;

	for(i = 0; i < 9; i++){

		u[i] = (std::complex<T>*)malloc(NNxl*NNyl*sizeof(std::complex<T>));

	}
}

template<class T> field<T>::~field(){

	int i;

	for(i = 0; i < 9; i++){

		free(u[i]);

	}

}


template<class T> int field<T>::allgather(field_global<T> u_global){


	T data_local_re[Nxl*Nyl];
	T data_local_im[Nxl*Nyl];

	T data_global_re[Nx*Ny];
	T data_global_im[Nx*Ny];


	int i,k;

	for(k = 0; k < 9; k++){

		for(i = 0; i < Nxl*Nyl; i++){

			data_local_re[i] = u[k][i].real();
			data_local_im[i] = u[k][i].imag();

		}

   		MPI_Allgather(data_local_re, Nxl*Nyl, MPI_DOUBLE, data_global_re, Nxl*Nyl, MPI_DOUBLE, MPI_COMM_WORLD); 
	   	MPI_Allgather(data_local_im, Nxl*Nyl, MPI_DOUBLE, data_global_im, Nxl*Nyl, MPI_DOUBLE, MPI_COMM_WORLD); 

		for(i = 0; i < Nx*Ny; i++){

			u_global.u[k][i] = data_global_re[i] + I*data_global_im[i];
	
		}
	}

	return 1;
}

template<class T> int field<T>::mpi_exchange_boundaries(void){

    int size, rank;

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double *bufor_send_n;
    double *bufor_receive_n;

    double *bufor_send_p;
    double *bufor_receive_p;


    if( ExchangeX == 1 ){

	    int yy; 

	    bufor_send_n = (double*) malloc(Nyl_buf*sizeof(double));
	    bufor_receive_n = (double*) malloc(Nyl_buf*sizeof(double));

  	    for(yy = 0; yy < Nyl; yy++){
		bufor_send_n[yy] = u[0][buf_pos(Nxl-1,yy)].real();
	    }

	    printf("X data exchange: rank %i sending to %i\n", rank, XNeighbourNext);
	    printf("X data exchange: rank %i receiving from %i\n", rank, XNeighbourNext);

	    MPI_Send(bufor_send_n, Nyl_buf, MPI_DOUBLE, XNeighbourNext, 11, MPI_COMM_WORLD);
    	    MPI_Recv(bufor_receive_n, Nyl_buf, MPI_DOUBLE, XNeighbourPrevious, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  	    for(yy = 0; yy < Nyl; yy++){
		u[0][buf_pos_ex(0,yy)] = bufor_receive_n[yy];
	    }

   	    bufor_send_p = (double*) malloc(Nyl_buf*sizeof(double));
	    bufor_receive_p = (double*) malloc(Nyl_buf*sizeof(double));

	    for(yy = 0; yy < Nyl; yy++){
		bufor_send_p[yy] = u[0][buf_pos(0,yy)].real();
	    }
	
 	    printf("X data exchange: rank %i sending to %i\n", rank, XNeighbourPrevious);
	    printf("X data exchange: rank %i receiving to %i\n", rank, XNeighbourPrevious);

	    MPI_Send(bufor_send_p, Nyl_buf, MPI_DOUBLE, XNeighbourPrevious, 12, MPI_COMM_WORLD);
    	    MPI_Recv(bufor_receive_p, Nyl_buf, MPI_DOUBLE, XNeighbourNext, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  	    for(yy = 0; yy < Nyl; yy++){
		u[0][buf_pos_ex(Nxl+1,yy)] = bufor_receive_p[yy];
	    }
    }

    if( ExchangeY == 1 ){

	    int xx; 

	    bufor_send_n = (double*) malloc(Nxl_buf*sizeof(double));
	    bufor_receive_n = (double*) malloc(Nxl_buf*sizeof(double));

  	    for(xx = 0; xx < Nxl; xx++){
		bufor_send_n[xx] = u[0][buf_pos(xx,Nyl-1)].real();
	    }

	    printf("Y data exchange: rank %i sending to %i\n", rank, YNeighbourNext);
	    printf("Y data exchange: rank %i receiving from %i\n", rank, YNeighbourNext);

	    MPI_Send(bufor_send_n, Nxl_buf, MPI_DOUBLE, YNeighbourNext, 13, MPI_COMM_WORLD);
    	    MPI_Recv(bufor_receive_n, Nxl_buf, MPI_DOUBLE, YNeighbourPrevious, 13, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for(xx = 0; xx < Nxl; xx++){
		u[0][buf_pos_ex(xx,0)] = bufor_receive_n[xx];
	    }

	    bufor_send_p = (double*) malloc(Nxl_buf*sizeof(double));
	    bufor_receive_p = (double*) malloc(Nxl_buf*sizeof(double));

	    for(xx = 0; xx < Nxl; xx++){
		bufor_send_p[xx] = u[0][buf_pos(xx,0)].real();
	    }
	
 	    printf("Y data exchange: rank %i sending to %i\n", rank, YNeighbourPrevious);
	    printf("Y data exchange: rank %i receiving to %i\n", rank, YNeighbourPrevious);

	    MPI_Send(bufor_send_p, Nxl_buf, MPI_DOUBLE, YNeighbourPrevious, 14, MPI_COMM_WORLD);
    	    MPI_Recv(bufor_receive_p, Nxl_buf, MPI_DOUBLE, YNeighbourNext, 14, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  	    for(xx = 0; xx < Nxl; xx++){
		u[0][buf_pos_ex(xx,Nyl+1)] = bufor_receive_p[xx];
	    }
    }

return 1;
}

