#ifndef H_SINGLE_FIELD
#define H_SINGLE_FIELD

#include <iostream>
#include <stdlib.h>
#include <complex>
#include <complex.h>
#include "config.h"

#include <omp.h>

#include <mpi.h>
#include <math.h>

#include "field.h"

#include "mpi_class.h"

template<class T> class sfield;

template<class T> class sfield {

	protected:
	
                int Nxl, Nyl;

        public:

                std::complex<T>* u;

                sfield(int NNx, int NNy);

                ~sfield(void);

		allgather(sfield<T> f);
};

template<class T> sfield<T>::sfield(int NNx, int NNy) {

        u = (std::complex<T>*)malloc(NNx*NNy*sizeof(std::complex<T>));

        Nxl = NNx;
        Nyl = NNy;
}

template<class T> sfield<T>::~sfield() {

        free(u);
}

template<class T> int sfield<T>::allgather(sfield<T>* ulocal){


        T* data_local_re = (T*)malloc(ulocal->Nxl*ulocal->Nyl*sizeof(T));
        T* data_local_im = (T*)malloc(ulocal->Nxl*ulocal->Nyl*sizeof(T));

        T* data_global_re = (T*)malloc(Nx*Ny*sizeof(T));
        T* data_global_im = (T*)malloc(Nx*Ny*sizeof(T));

        int i;

        for(i = 0; i < ulocal->Nxl*ulocal->Nyl; i++){

	        data_local_re[i] = ulocal->u[i].real();
                data_local_im[i] = ulocal->u[i].imag();

        }

        MPI_Allgather(data_local_re, ulocal->Nxl*ulocal->Nyl, MPI_DOUBLE, data_global_re, ulocal->Nxl*ulocal->Nyl, MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgather(data_local_im, ulocal->Nxl*ulocal->Nyl, MPI_DOUBLE, data_global_im, ulocal->Nxl*ulocal->Nyl, MPI_DOUBLE, MPI_COMM_WORLD);

       	for(i = 0; i < Nx*Ny; i++){

	       this->u[i] = data_global_re[i] + I*data_global_im[i];

        }

        return 1;
}


#endif
