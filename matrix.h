#ifndef H_MATRIX
#define H_MATRIX

#include <iostream>
#include <stdlib.h>
#include <complex>
#include <complex.h>
#include "config.h"

#include <omp.h>

#include <mpi.h>
#include <math.h>

#include "field.h"

template<class T> class matrix {

	public:

		std::complex<T>* u;		

		int Nxl, Nyl;

	public:

		matrix(int NNx, int NNy);

		~matrix(void);
};

template<class T> matrix<T>::matrix(int NNx, int NNy) {

	u = (std::complex<T>*)malloc(NNx*NNy*sizeof(std::complex<T>));

	Nxl = NNx;
	Nyl = NNy;
}

template<class T> matrix<T>::~matrix() {

	free(u);
}

template<class T> class lmatrix;


template<class T> class gmatrix: public matrix<T> {


	public:

		int Nxg, Nyg;

		gmatrix(int NNx, int NNy) : matrix<T>{NNx, NNy} { Nxg = NNx; Nyg = NNy;};

		gmatrix<T>& operator= ( const gmatrix<T>& f );

		int decompose(gfield<T,1>* corr);

};


template<class T> class lmatrix: public matrix<T> {

	public:

		int Nxl, Nyl;

		lmatrix(int NNx, int NNy) : matrix<T>{NNx, NNy} { Nxl = NNx; Nyl = NNy; };

		friend class gmatrix<T>;

		lmatrix<T>& operator= ( const lmatrix<T>& f );

};

template<class T> lmatrix<T>& lmatrix<T>::operator= ( const lmatrix<T>& f ){

			if( this != &f ){

			for(int i = 0; i < f.Nxl*f.Nyl; i ++){
			
					this->u[i] = f.u[i];
			}

			}

		return *this;
		}

template<class T> gmatrix<T>& gmatrix<T>::operator= ( const gmatrix<T>& f ){

			if( this != &f ){

			for(int i = 0; i < f.Nxg*f.Nyg; i ++){
			
					this->u[i] = f.u[i];
			}

			}

		return *this;
		}

template<class T> int gmatrix<T>::decompose(gfield<T,1>* corr){

    //Nxg and Nyg are sizes of the global matrix!!! -> Nxg = Nxg*Nyg
    for (int i = 0; i < Nxg; i++) { //x
        for (int j = 0; j <= i; j++) {  //y

            if (j == i) // summation for diagnols 
            { 

                std::complex<double> sum = 0; 

                for (int k = 0; k < i; k++){
                    sum += pow(this->u[i*Nyg+k], 2); 
		}

	        this->u[i*Nyg+i] = sqrt(corr->u[0][0] - sum);  //distance between i and j = 0

            } else { 
 
                std::complex<double> sum = 0; 

                // Evaluating L(i, j) using L(j, j) 
                for (int k = 0; k < j; k++){
                    sum += (this->u[i*Nyg+k] * this->u[j*Nyg+k]); 
		}

		int xi = i/Nyg;
		int yi = i - xi*Nyg;

		int xj = j/Nyg;
		int yj = j - xj*Nyg;

		int ii = abs(xi-xj)*Nyg + abs(yi-yj);

                this->u[i*Nyg+j] = (corr->u[0][ii] - sum) /  //distance between i and j
                                      this->u[j*Nyg+j]; 
            } 
        } 
    } 


return 1;
}

#endif