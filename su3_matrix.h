/********************************************//** 
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
 * File: su3_matrix.h
 * Authors: P. Korcyl
 * Contact: piotr.korcyl@uj.edu.pl
 * 
 * Version: 1.0
 * 
 * Description:
 * Adaptation of the numerical diagonalization of 3x3 matrices by Joachim Kopp (2006)
 * Functions for SU(3) multiplication and diagonalizations. Adaptation of open-source package https://www-numi.fnal.gov/offline_software/srt_public_context/WebDocs/doxygen/loon/html/zheevc3_8cxx.html#a9ac76ac4ddaf7e2a513107356d1ed85e and related files to C++ and C++ implementation of complex numbers.
 * 
 **************************************************/

#ifndef H_SU3_MATRIX
#define H_SU3_MATRIX

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>

#include <fstream>


using namespace std;

/* Complex datatype */
struct _dcomplex { double re, im; };
typedef struct _dcomplex dcomplex;

    // zgeev_ is a symbol in the LAPACK library files
extern "C" {
//    extern int zgeev_(char*,char*,int*,double*,int*,double*, double*, double*, int*, double*, int*, double*, int*, int*);
//      extern int zgeev_(char*,char*,int*,double*,int*,double*,double*,int*,double*,int*,double*,int*,double*,int*);
      extern int zgeev_(char* jobvl, char* jobvr, int* n, dcomplex* a,
                int* lda, dcomplex* w, dcomplex* vl, int* ldvl, dcomplex* vr, int* ldvr,
                dcomplex* work, int* lwork, double* rwork, int* info );

}


//#include "zheevh3.h"

#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <float.h>
#include "zheevh3-C-1.1/zheevc3.h"
#include "zheevh3-C-1.1/zheevq3.h"
//#include "zheevh3.h"

// Macros
#define SQR(x)      ((x)*(x))                        // x^2 
#define SQR_ABS(x)  (SQR(x.real()) + SQR(x.imag()))  // |x|^2



//#include "su3_complex.h"

template <class T> int function_zheevh3(std::complex<T> A[3][3], std::complex<T> Q[3][3], T w[3]);

template <class T>
class su3_matrix {
	public:
		std::complex<T> m[9];

		su3_matrix() {
			m[0] = 0.0;
			m[1] = 0.0;
			m[2] = 0.0;
			m[3] = 0.0;
			m[4] = 0.0;
			m[5] = 0.0;
			m[6] = 0.0;
			m[7] = 0.0;
			m[8] = 0.0;	
		}

		int exponentiate(double sign);
		int square_root(void);
		int inverse(void);

		std::complex<T> trace(su3_matrix<T> a);
		std::complex<T> determinant(su3_matrix<T> a);
		su3_matrix<T> su3_projection(void);

		int zheevh3(double *w){

			std::complex<T> A[3][3];
			std::complex<T> Q[3][3];
			double ww[3];

			A[0][0] = m[0];
			A[0][1] = m[1];
			A[0][2] = m[2];
			A[1][0] = m[3];
			A[1][1] = m[4]; 
			A[1][2] = m[5];
			A[2][0] = m[6];
			A[2][1] = m[7];
			A[2][2] = m[8];

			int st;
			st = function_zheevh3(A, Q, ww);

			w[0] = ww[0];
			w[1] = ww[1];
			w[2] = ww[2];

			return st;

		}

		void print() {
				for (int i = 0; i < 9; i++) {
					std::cerr<<"m["<<i<<"].r = "<<m[i].real()<<std::endl;
					std::cerr<<"m["<<i<<"].i = "<<m[i].imag()<<std::endl;
				}
		}

		void toVector(T* r) {
			for (int i = 0; i < 9; i++) {
				r[2 * i] = m[i].real();
				r[2 * i + 1] = m[i].imag();
			}
		}


		su3_matrix<T> operator*(T alpha) {

			int i;
			for(i=0;i<9;i++){
				this->m[i] *= alpha;
			}

			return *this;
		}

		su3_matrix<T> operator=(const su3_matrix<T>& a) {

			int i;
			for(i=0;i<9;i++){
				this->m[i] = a.m[i];
			}
			return *this;
		}

		su3_matrix<T> operator+(const su3_matrix<T>& b) {

			su3_matrix<T> c;
			int i;
			for(i=0;i<9;i++){
				c.m[i] = this->m[i] + b.m[i];
			}
			return c;
		}

		su3_matrix<T> operator-(const su3_matrix<T> &b) {
			su3_matrix<T> c;
			int i;
			for(i=0;i<9;i++){
				c.m[i] = this->m[i] - b.m[i];
			}
			return c;
		}

		su3_matrix<T> operator*(const su3_matrix<T> &b) {

			su3_matrix<T> c;
			int i,j,k;
			for(i=0;i<3;i++){
				for(j=0;j<3;j++){
					for(k=0;k<3;k++){
						c.m[i*3+j] += this->m[i*3+k] * b.m[k*3+j];
					}
				}
			}

			return c;
		}

		su3_matrix<T> operator^(const su3_matrix<T> &b) {

			su3_matrix<T> c;
			int i,j,k;
			for(i=0;i<3;i++){
				for(j=0;j<3;j++){
					for(k=0;k<3;k++){
						c.m[i*3+j] += std::conj(this->m[k*3+i]) * b.m[k*3+j];
					}
				}
			}

			return c;
		}

};

template<class T> int su3_matrix<T>::exponentiate(double sign){

			std::complex<T> A[3][3];
			std::complex<T> Q[3][3];
			double ww[3];
			std::complex<T> eig[3];

			A[0][0] = m[0];
			A[0][1] = m[1];
			A[0][2] = m[2];
			A[1][0] = m[3];
			A[1][1] = m[4]; 
			A[1][2] = m[5];
			A[2][0] = m[6];
			A[2][1] = m[7];
			A[2][2] = m[8];


			int st;
			st = function_zheevh3(A, Q, ww);

			eig[0] = exp(std::complex<double>(0.0, sign*ww[0]));
			eig[1] = exp(std::complex<double>(0.0, sign*ww[1]));
			eig[2] = exp(std::complex<double>(0.0, sign*ww[2]));

			for(int i = 0; i < 3; i++){
				for(int j = 0; j < 3; j++){
					
					m[i*3+j] = 0.0;
					
					for(int k = 0; k < 3; k++){

						m[i*3+j] += Q[i][k] * eig[k] * std::conj(Q[j][k]); 

					}
				}
			}

			return st;

}

template<class T> int su3_matrix<T>::square_root(void){

			std::complex<T> A[3][3];
			std::complex<T> Q[3][3];
			double ww[3];
			std::complex<T> eig[3];

			A[0][0] = m[0];
			A[0][1] = m[1];
			A[0][2] = m[2];
			A[1][0] = m[3];
			A[1][1] = m[4]; 
			A[1][2] = m[5];
			A[2][0] = m[6];
			A[2][1] = m[7];
			A[2][2] = m[8];


			int st;
			st = function_zheevh3(A, Q, ww);

			eig[0] = sqrt(fabs(ww[0]));
			eig[1] = sqrt(fabs(ww[1]));
			eig[2] = sqrt(fabs(ww[2]));

			for(int i = 0; i < 3; i++){
				for(int j = 0; j < 3; j++){
					
					m[i*3+j] = 0.0;
					
					for(int k = 0; k < 3; k++){

						m[i*3+j] += Q[i][k] * eig[k] * std::conj(Q[j][k]); 

					}
				}
			}

			return st;

}

template<class T> inline std::complex<T> su3_matrix<T>::trace(su3_matrix<T> a){
  std::complex<T> res;
  //a (a0 a1 a2                                                                                                                           
  //   a3 a4 a5                                                                                                                           
  //   a6 a7 a8)                                                                                                                          
  res=a.m[0] + a.m[4] + a.m[8];
  return res;
}

template<class T> inline std::complex<T> su3_matrix<T>::determinant(su3_matrix<T> a){
  std::complex<T> res;
  //a (a0 a1 a2                                                                                                                           
  //   a3 a4 a5                                                                                                                           
  //   a6 a7 a8)                                                                                                                          
  res = a.m[0]*a.m[4]*a.m[8] + a.m[3]*a.m[7]*a.m[2] + a.m[1]*a.m[5]*a.m[6] - a.m[2]*a.m[4]*a.m[6] - a.m[5]*a.m[7]*a.m[0] - a.m[1]*a.m[3]*a.m[8];
  return res;
}


template<class T> int diagonalize_unitary(su3_matrix<T> a){

      int n = 3;
      int m = 3;
      dcomplex *data;
      dcomplex *data_tmp;

	printf("Diagonalizing an unitary matrix\n");

	printf("Wypisujemy a.m[]\n");

	for(int i = 0; i < 3; i++){
		for(int j = 0; j < 3; j++){
			printf("( %1.4e %1.4e )", a.m[i*3+j].real(), a.m[i*3+j].imag());
		}
		printf("\n");
	}
	printf("\n");

	su3_matrix<T> b;

	b = a^a;

	printf("Wypisujemy a*a\n");

	for(int i = 0; i < 3; i++){
		for(int j = 0; j < 3; j++){
			printf("( %1.4e %1.4e )", b.m[i*3+j].real(), b.m[i*3+j].imag());
		}
		printf("\n");
	}
	printf("\n");

	printf("Wypisujemy data\n");

      data = new dcomplex[9];
      data_tmp = new dcomplex[9];

      for (int j=0;j<n;j++){
        for (int i=0;i<n;i++){
          data[j*n+i].re = a.m[i*n+j].real();
          data[j*n+i].im = a.m[i*n+j].imag();
	}
      }

      for(int j = 0; j < 3; j++){
	      for(int i = 0; i < 3; i++){
		      printf("( %1.4e, %1.4e )", data[j*n+i].re, data[j*n+i].im);
	      }
     	      printf("\n");
      }
      printf("\n");


      double *output = new double[18];

       for(int j = 0; j < 3; j++){
	      for(int i = 0; i < 3; i++){
		
		      	double re = 0;
			double im = 0;

			for(int k = 0; k < 3; k++){

				re += data[j*n+k].re * data[i*n+k].re + data[j*n+k].im * data[i*n+k].im;
				im += data[j*n+k].im * data[i*n+k].re - data[j*n+k].re * data[i*n+k].im;
			}

			output[(j*n+i)*2+0] = re;
			output[(j*n+i)*2+1] = im;
	      }
      }

      for(int j = 0; j < 3; j++){
	      for(int i = 0; i < 3; i++){
		      printf("( %1.4e, %1.4e )", output[(j*n+i)*2+0], output[(j*n+i)*2+1]);
	      }
     	      printf("\n");
      }
      printf("\n");

     // allocate data
      char Nchar='N';
      char Vchar='V';

      dcomplex *eig=new dcomplex[n];
      dcomplex *vl = new dcomplex[n*n];
      dcomplex *vr = new dcomplex[n*n];
      int one=1;
      int three=3;
      int lwork=2*6*n;
      dcomplex *work=new dcomplex[lwork];
      double *rwork=new double[2*lwork];

      int info;

      int nn = 2*n;

      //zgeev (jobvl, jobvr, n, a, lda, w, vl, ldvl, vr, ldvr, work, lwork, rwork, info);
//      extern int zgeev_(char* jobvl, char* jobvr, int* n, dcomplex* a,
//                int* lda, dcomplex* w, dcomplex* vl, int* ldvl, dcomplex* vr, int* ldvr,
 //               dcomplex* work, int* lwork, double* rwork, int* info );

      // calculate eigenvalues using the DGEEV subroutine
      zgeev_(&Vchar,&Vchar,&n,data,&n,eig,vl,&three,vr,&three,work,&lwork,rwork,&info);


      // check for errors
      if (info!=0){
        cout << "Error: dgeev returned error code " << info << endl;
        return -1;
      }

      // output eigenvalues to stdout
      cout << "--- Eigenvalues ---" << endl;
      for (int i=0;i<n;i++){
        cout << "( " << eig[i].re << " , " << eig[i].im << " )\n";
      }
      cout << endl;

      	// output eigenvectors to stdout
      	for(int jj = 0; jj < 3; jj++){
		
	        cout << "--- Right eigenvector " << jj << " ---" << endl;
	        for (int i=0;i<n;i++){
        
		      cout << "( " << vr[jj*n+i].re << " , " << vr[jj*n+i].im << " )\n";
      		}

      		cout << endl;
	}

      	// output eigenvectors to stdout
      	for(int jj = 0; jj < 3; jj++){
		
	        cout << "--- Left eigenvector " << jj << " ---" << endl;
	        for (int i=0;i<n;i++){
        
		      cout << "( " << vl[jj*n+i].re << " , " << vl[jj*n+i].im << " )\n";
      		}

      		cout << endl;
	}

	for(int j = 0; j < 3; j++){
		for(int i = 0; i < 3; i++){
		
		      	double re = 0;
			double im = 0;

			for(int k = 0; k < 3; k++){

				re += data[i*n+k].re * vr[j*n+k].re - data[i*n+k].im * vr[j*n+k].im;
				im += data[i*n+k].im * vr[j*n+k].re + data[i*n+k].re * vr[j*n+k].im;
			}

			printf(" re = %1.4e, im = %1.4e\n", (re*eig[i].re + im*eig[i].im) / (eig[i].re*eig[i].re + eig[i].im*eig[i].im), (-re*eig[i].im+im*eig[i].re) / (eig[i].re*eig[i].re+eig[i].im*eig[i].im) );
	      }
	}

	//calculate inverse
	//
			dcomplex feig[3];
			for(int kk = 0; kk < 3; kk++){
				//feig[kk].re = eig[kk].re; //eig[kk].re/(eig[kk].re*eig[kk].re + eig[kk].im*eig[kk].im);
				//feig[kk].im = eig[kk].im; //-eig[kk].im/(eig[kk].re*eig[kk].re + eig[kk].im*eig[kk].im);
				feig[kk].re = eig[kk].re/(eig[kk].re*eig[kk].re + eig[kk].im*eig[kk].im);
				feig[kk].im = -eig[kk].im/(eig[kk].re*eig[kk].re + eig[kk].im*eig[kk].im);

			}


                       for(int i = 0; i < 3; i++){
                                for(int j = 0; j < 3; j++){

                                        data_tmp[i*3+j].re = 0.0;
                                        data_tmp[i*3+j].im = 0.0;

                                        for(int k = 0; k < 3; k++){

						//Q[i][k] * eig[k] * std::conj(Q[j][k]);

						double tmp_re = vr[k*n+i].re * feig[k].re - vr[k*n+i].im * feig[k].im;
						double tmp_im = vr[k*n+i].im * feig[k].re + vr[k*n+i].re * feig[k].im;

						data_tmp[i*3+j].re += tmp_re * vr[k*n+j].re + tmp_im * vr[k*n+j].im;
						data_tmp[i*3+j].im += -tmp_re * vr[k*n+j].im + tmp_im * vr[k*n+j].re;

                                        }
                                }
                        }

	printf("1/A = \n");

       for(int j = 0; j < 3; j++){
	      for(int i = 0; i < 3; i++){
		      printf("( %1.4e, %1.4e )", data_tmp[j*n+i].re, data_tmp[j*n+i].im);
	      }
     	      printf("\n");
      }
      printf("\n");

	printf("A for comparison\n");

      for(int j = 0; j < 3; j++){
	      for(int i = 0; i < 3; i++){
		      printf("( %1.4e, %1.4e )", a.m[j*n+i].real(), a.m[j*n+i].imag());
	      }
     	      printf("\n");
      }
      printf("\n");


      for(int j = 0; j < 3; j++){
	      for(int i = 0; i < 3; i++){
		
		      	double re = 0;
			double im = 0;

			for(int k = 0; k < 3; k++){

				re += a.m[j*n+k].real() * data_tmp[k*n+i].re - a.m[j*n+k].imag() * data_tmp[k*n+i].im;
				im += a.m[j*n+k].imag() * data_tmp[k*n+i].re + a.m[j*n+k].real() * data_tmp[k*n+i].im;
			}

			output[(j*n+i)*2+0] = re;
			output[(j*n+i)*2+1] = im;
	      }
      }

	printf(" A * 1/A = \n");

      for(int j = 0; j < 3; j++){
	      for(int i = 0; i < 3; i++){
		      printf("( %1.4e, %1.4e )", output[(j*n+i)*2+0], output[(j*n+i)*2+1]);
	      }
     	      printf("\n");
      }
      printf("\n");




      // deallocate
      delete [] data;
      delete [] eig;
      delete [] work;
      delete [] vl;
      delete [] vr;
      delete [] rwork;


      return 0;
}

template<class T> int interpolate(su3_matrix<T> left, su3_matrix<T> right, su3_matrix<T> *output){

      	int n = 3;
      	int m = 3;
      	dcomplex *data;
	dcomplex *data_tmp;
	dcomplex *data_tmp2;
	dcomplex *data_tmp3;
	dcomplex *data_tmp4;
	dcomplex *data_tmp5;

	su3_matrix<T> left_inv;

        left_inv.m[0] = std::conj(left.m[0]);
        left_inv.m[1] = std::conj(left.m[3]);
        left_inv.m[2] = std::conj(left.m[6]);
        left_inv.m[3] = std::conj(left.m[1]);
        left_inv.m[4] = std::conj(left.m[4]);
        left_inv.m[5] = std::conj(left.m[7]);
        left_inv.m[6] = std::conj(left.m[2]);
        left_inv.m[7] = std::conj(left.m[5]);
        left_inv.m[8] = std::conj(left.m[8]);

        data = new dcomplex[9];
        data_tmp = new dcomplex[9];
        data_tmp2 = new dcomplex[9];
        data_tmp3 = new dcomplex[9];
        data_tmp4 = new dcomplex[9];
        data_tmp5 = new dcomplex[9];

	//B/A
        for(int j = 0; j < 3; j++){
	      	for(int i = 0; i < 3; i++){
		
		      	double re = 0;
			double im = 0;

			for(int k = 0; k < 3; k++){

				//re += right.m[j*n+k].real() * left_inv.m[k*n+i].real() - right.m[j*n+k].imag() * left_inv.m[k*n+i].imag();
				//im += right.m[j*n+k].imag() * left_inv.m[k*n+i].real() + right.m[j*n+k].real() * left_inv.m[k*n+i].imag();

				re += left_inv.m[j*n+k].real() * right.m[k*n+i].real() - left_inv.m[j*n+k].imag() * right.m[k*n+i].imag();
				im += left_inv.m[j*n+k].imag() * right.m[k*n+i].real() + left_inv.m[j*n+k].real() * right.m[k*n+i].imag();

			}

			data[j*n+i].re = re;
			data[j*n+i].im = im;
//			data_tmp4[j*n+i].re = re;
//			data_tmp4[j*n+i].im = im;

  		}
      }
/*
	printf("B/A =\n");

      for(int j = 0; j < 3; j++){
	      for(int i = 0; i < 3; i++){
		      printf("( %1.4e, %1.4e )", data[j*n+i].re, data[j*n+i].im);
	      }
     	      printf("\n");
      }
      printf("\n");

*/

      // allocate data
      char Nchar='N';
      char Vchar='V';

      dcomplex *eig=new dcomplex[n];
      dcomplex *vl = new dcomplex[n*n];
      dcomplex *vr = new dcomplex[n*n];
      int one=1;
      int three=3;
      int lwork=2*6*n;
      dcomplex *work=new dcomplex[lwork];
      double *rwork=new double[2*lwork];

      int info;

      int nn = 2*n;

      //zgeev (jobvl, jobvr, n, a, lda, w, vl, ldvl, vr, ldvr, work, lwork, rwork, info);
//      extern int zgeev_(char* jobvl, char* jobvr, int* n, dcomplex* a,
//                int* lda, dcomplex* w, dcomplex* vl, int* ldvl, dcomplex* vr, int* ldvr,
 //               dcomplex* work, int* lwork, double* rwork, int* info );

      // calculate eigenvalues using the DGEEV subroutine
      zgeev_(&Vchar,&Vchar,&n,data,&n,eig,vl,&three,vr,&three,work,&lwork,rwork,&info);


      // check for errors
      if (info!=0){
        cout << "Error: dgeev returned error code " << info << endl;
        return -1;
      }

      //calculate square root

			dcomplex feig[3];
			for(int kk = 0; kk < 3; kk++){
				//feig[kk].re = eig[kk].re; //eig[kk].re/(eig[kk].re*eig[kk].re + eig[kk].im*eig[kk].im);
				//feig[kk].im = eig[kk].im; //-eig[kk].im/(eig[kk].re*eig[kk].re + eig[kk].im*eig[kk].im);
				//feig[kk].re = eig[kk].re/(eig[kk].re*eig[kk].re + eig[kk].im*eig[kk].im);
				//feig[kk].im = -eig[kk].im/(eig[kk].re*eig[kk].re + eig[kk].im*eig[kk].im);
				
				double x = eig[kk].re;
				double y = eig[kk].im;

				double norm = x*x+y*y;
				double phi = atan(y/x);

				//feig[kk].re = 0.5*(x + sqrt(x*x+y*y));
				//feig[kk].im = 0.5*(-x + sqrt(x*x+y*y));

				feig[kk].re = sqrt(norm)*cos(0.5*phi);
				feig[kk].im = sqrt(norm)*sin(0.5*phi);

				double xx = feig[kk].re;
				double yy = feig[kk].im;	

				double square_re = xx*xx-yy*yy;
				double square_im = 2*xx*yy;

				if(square_re/x < 0){
					feig[kk].re = -yy;
					feig[kk].im = xx;
				}

				xx = feig[kk].re;
				yy = feig[kk].im;	

//				printf("eig.re = %e, eig.im = %e,     square_re = %e, square_im = %e\n", eig[kk].re, eig[kk].im, xx*xx-yy*yy, 2*xx*yy);
			}


                       for(int i = 0; i < 3; i++){
                                for(int j = 0; j < 3; j++){

                                        data_tmp[j*3+i].re = 0.0;
                                        data_tmp[j*3+i].im = 0.0;

                                        for(int k = 0; k < 3; k++){

						//Q[i][k] * eig[k] * std::conj(Q[j][k]);

						double tmp_re = vr[k*n+i].re * feig[k].re - vr[k*n+i].im * feig[k].im;
						double tmp_im = vr[k*n+i].im * feig[k].re + vr[k*n+i].re * feig[k].im;

						data_tmp[j*3+i].re += tmp_re * vr[k*n+j].re + tmp_im * vr[k*n+j].im;
						data_tmp[j*3+i].im += -tmp_re * vr[k*n+j].im + tmp_im * vr[k*n+j].re;

                                        }
                                }
                        }

                       for(int i = 0; i < 3; i++){
                                for(int j = 0; j < 3; j++){

                                        output->m[i*3+j] = std::complex<double>(data_tmp[i*3+j].re, data_tmp[i*3+j].im);

                                }
                        }
/*
      printf("B/A from diagonalization\n");

      for(int j = 0; j < 3; j++){
	      for(int i = 0; i < 3; i++){
		      printf("( %1.4e, %1.4e )", data_tmp[j*n+i].re, data_tmp[j*n+i].im);
	      }
     	      printf("\n");
      }
      printf("\n");



        //sqrt(B/A) * sqrt(B/A)
        for(int j = 0; j < 3; j++){
	      	for(int i = 0; i < 3; i++){
		
		      	double re = 0;
			double im = 0;

			for(int k = 0; k < 3; k++){

				//re += right.m[j*n+k].real() * left_inv.m[k*n+i].real() - right.m[j*n+k].imag() * left_inv.m[k*n+i].imag();
				//im += right.m[j*n+k].imag() * left_inv.m[k*n+i].real() + right.m[j*n+k].real() * left_inv.m[k*n+i].imag();

				re += data_tmp[j*n+k].re * data_tmp[k*n+i].re - data_tmp[j*n+k].im * data_tmp[k*n+i].im;
				im += data_tmp[j*n+k].im * data_tmp[k*n+i].re + data_tmp[j*n+k].re * data_tmp[k*n+i].im;

			}

			data_tmp5[j*n+i].re = re;
			data_tmp5[j*n+i].im = im;
	      }
      }

      printf("sqrt(B/A) * sqrt(B/A) = B/A\n");

      for(int j = 0; j < 3; j++){
	      for(int i = 0; i < 3; i++){
		      printf("( %1.4e, %1.4e )", data_tmp5[j*n+i].re, data_tmp5[j*n+i].im);
	      }
     	      printf("\n");
      }
      printf("\n");



	//A * sqrt(B/A)
        for(int j = 0; j < 3; j++){
	      	for(int i = 0; i < 3; i++){
		
		      	double re = 0;
			double im = 0;

			for(int k = 0; k < 3; k++){

				//re += right.m[j*n+k].real() * left_inv.m[k*n+i].real() - right.m[j*n+k].imag() * left_inv.m[k*n+i].imag();
				//im += right.m[j*n+k].imag() * left_inv.m[k*n+i].real() + right.m[j*n+k].real() * left_inv.m[k*n+i].imag();

				re += left.m[j*n+k].real() * data_tmp[k*n+i].re - left.m[j*n+k].imag() * data_tmp[k*n+i].im;
				im += left.m[j*n+k].imag() * data_tmp[k*n+i].re + left.m[j*n+k].real() * data_tmp[k*n+i].im;

			}

			data_tmp2[j*n+i].re = re;
			data_tmp2[j*n+i].im = im;
	      }
      }

        //A * sqrt(B/A) * sqrt(B/A)
        for(int j = 0; j < 3; j++){
	      	for(int i = 0; i < 3; i++){
		
		      	double re = 0;
			double im = 0;

			for(int k = 0; k < 3; k++){

				//re += right.m[j*n+k].real() * left_inv.m[k*n+i].real() - right.m[j*n+k].imag() * left_inv.m[k*n+i].imag();
				//im += right.m[j*n+k].imag() * left_inv.m[k*n+i].real() + right.m[j*n+k].real() * left_inv.m[k*n+i].imag();

				re += data_tmp2[j*n+k].re * data_tmp[k*n+i].re - data_tmp2[j*n+k].im * data_tmp[k*n+i].im;
				im += data_tmp2[j*n+k].im * data_tmp[k*n+i].re + data_tmp2[j*n+k].re * data_tmp[k*n+i].im;

			}

			data_tmp3[j*n+i].re = re;
			data_tmp3[j*n+i].im = im;
	      }
      }

      printf("A * sqrt(B/A) * sqrt(B/A) = B\n");

      for(int j = 0; j < 3; j++){
	      for(int i = 0; i < 3; i++){
		      printf("( %1.4e, %1.4e )", data_tmp3[j*n+i].re, data_tmp3[j*n+i].im);
	      }
     	      printf("\n");
      }
      printf("\n");


      printf("B again for comparison\n");

      for(int j = 0; j < 3; j++){
	      for(int i = 0; i < 3; i++){
		      printf("( %1.4e, %1.4e )", right.m[j*n+i].real(), right.m[j*n+i].imag());
	      }
     	      printf("\n");
      }
      printf("\n");

	//A * B/A
        for(int j = 0; j < 3; j++){
	      	for(int i = 0; i < 3; i++){
		
		      	double re = 0;
			double im = 0;

			for(int k = 0; k < 3; k++){

				//re += right.m[j*n+k].real() * left_inv.m[k*n+i].real() - right.m[j*n+k].imag() * left_inv.m[k*n+i].imag();
				//im += right.m[j*n+k].imag() * left_inv.m[k*n+i].real() + right.m[j*n+k].real() * left_inv.m[k*n+i].imag();

				re += left.m[j*n+k].real() * data_tmp4[k*n+i].re - left.m[j*n+k].imag() * data_tmp4[k*n+i].im;
				im += left.m[j*n+k].imag() * data_tmp4[k*n+i].re + left.m[j*n+k].real() * data_tmp4[k*n+i].im;

			}

			data_tmp2[j*n+i].re = re;
			data_tmp2[j*n+i].im = im;
	      }
      }

      printf("A * B/A = B\n");

      for(int j = 0; j < 3; j++){
	      for(int i = 0; i < 3; i++){
		      printf("( %1.4e, %1.4e )", data_tmp2[j*n+i].re, data_tmp2[j*n+i].im);
	      }
     	      printf("\n");
      }
      printf("\n");
*/

      // deallocate
      delete [] data;
      delete [] eig;
      delete [] work;
      delete [] vl;
      delete [] vr;
      delete [] rwork;


      return 0;
}


template<class T> inline su3_matrix<T> su3_matrix<T>::su3_projection(void){
  //iterative method of arXiv: hep-lat/0506008v1, eqs. 2.24 and 2.2
  su3_matrix<T> res, X;
  double tmp;

  tmp=0.0;

  for (int i = 0; i < 9; i++){
    tmp += this->m[i].real()*this->m[i].real() + this->m[i].imag()*this->m[i].imag();    //trace(WW^dagger)=sum_i(a_i*a_i^star)                                                 
  }
  tmp = 1.0/sqrt(tmp/3.0);

  for(int i = 0; i < 9; i++){
  	res.m[i] = tmp*this->m[i];
  }

  //4 iterations                                                                                                                          
  for (int i = 0; i < 4; i++){
    X=res^res; //Wdaggerr*W

    for(int k = 0; k < 9; k++){
  	X.m[k] = -0.5*X.m[k];
    }

    //--------------------------------------(3/2)*Id - -0.5*Wdagger*W                                                                     
  
    std::complex<T> ttmpa(X.m[0].real() + 1.5, X.m[0].imag());
    X.m[0] = ttmpa;
    std::complex<T> ttmpb(X.m[4].real() + 1.5, X.m[4].imag());
    X.m[4] = ttmpb;
    std::complex<T> ttmpc(X.m[8].real() + 1.5, X.m[8].imag());
    X.m[8] = ttmpc;

    //---------------------------------------
    //X=add_two(three_over_two,X);          
    X=res*X;                 //W*((3/2)*Id - -0.5*Wdagger*W)                                                                
    std::complex<T> tmp_complex(1.0, -0.33333*determinant(X).imag());
  
    for(int i = 0; i < 9; i++){
  	res.m[i] = X.m[i]*tmp_complex;
    }

//	cout << "projection res iteration = " << i << "\n";
//	cout << res.m[0].r << " " << res.m[1].r << " " << res.m[2].r << "\n" ;
//	cout << res.m[3].r << " " << res.m[4].r << " " << res.m[5].r << "\n" ;
//	cout << res.m[6].r << " " << res.m[7].r << " " << res.m[8].r << "\n" ;



  }

  return res;
}



template<class T> int su3_matrix<T>::inverse(void){

			std::complex<T> A[3][3];
			std::complex<T> Q[3][3];
			double ww[3];
			std::complex<T> eig[3];

			A[0][0] = m[0];
			A[0][1] = m[1];
			A[0][2] = m[2];
			A[1][0] = m[3];
			A[1][1] = m[4]; 
			A[1][2] = m[5];
			A[2][0] = m[6];
			A[2][1] = m[7];
			A[2][2] = m[8];


			int st;
			st = function_zheevh3(A, Q, ww);

			eig[0] = 1.0/(ww[0]);
			eig[1] = 1.0/(ww[1]);
			eig[2] = 1.0/(ww[2]);

			for(int i = 0; i < 3; i++){
				for(int j = 0; j < 3; j++){
					
					m[i*3+j] = 0.0;
					
					for(int k = 0; k < 3; k++){

						m[i*3+j] += Q[i][k] * eig[k] * std::conj(Q[j][k]); 

					}
				}
			}

			return st;

}




// ----------------------------------------------------------------------------
template<class T> int function_zheevh3(std::complex<T> A[3][3], std::complex<T> Q[3][3], T w[3])
/// ----------------------------------------------------------------------------
/// Calculates the eigenvalues and normalized eigenvectors of a hermitian 3x3
/// matrix A using Cardano's method for the eigenvalues and an analytical
/// method based on vector cross products for the eigenvectors. However,
/// if conditions are such that a large error in the results is to be
/// expected, the routine falls back to using the slower, but more
/// accurate QL algorithm. Only the diagonal and upper triangular parts of A need
/// to contain meaningful values. Access to A is read-only.
/// ----------------------------------------------------------------------------
/// Parameters:
///   A: The hermitian input matrix
///   Q: Storage buffer for eigenvectors
///   w: Storage buffer for eigenvalues
/// ----------------------------------------------------------------------------
/// Return value:
///   0: Success
///  -1: Error
/// ----------------------------------------------------------------------------
/// Dependencies:
///   zheevc3(), zhetrd3(), zheevq3()
/// ----------------------------------------------------------------------------
/// Version history:
///   v1.1: Simplified fallback condition --> speed-up
///   v1.0: First released version
/// ----------------------------------------------------------------------------
{
#ifndef EVALS_ONLY
  double norm;          /// Squared norm or inverse norm of current eigenvector
//  double n0, n1;      // Norm of first and second columns of A
  double error;         /// Estimated maximum roundoff error
  double t, u;          /// Intermediate storage
  int j;                /// Loop counter
#endif

  /// Calculate eigenvalues
  zheevc3(A, w);

#ifndef EVALS_ONLY
//  n0 = SQR(creal(A[0][0])) + SQR_ABS(A[0][1]) + SQR_ABS(A[0][2]);
//  n1 = SQR_ABS(A[0][1]) + SQR(creal(A[1][1])) + SQR_ABS(A[1][2]);
  
  t = fabs(w[0]);
  if ((u=fabs(w[1])) > t)
    t = u;
  if ((u=fabs(w[2])) > t)
    t = u;
  if (t < 1.0)
    u = t;
  else
    u = SQR(t);
  error = 256.0 * DBL_EPSILON * SQR(u);
//  error = 256.0 * DBL_EPSILON * (n0 + u) * (n1 + u);

  Q[0][1] = A[0][1]*A[1][2] - A[0][2]*A[1][1].real();
  Q[1][1] = A[0][2]*conj(A[0][1]) - A[1][2]*A[0][0].real();
  Q[2][1] = SQR_ABS(A[0][1]);

  /// Calculate first eigenvector by the formula
  ///   v[0] = conj( (A - w[0]).e1 x (A - w[0]).e2 )
  Q[0][0] = Q[0][1] + A[0][2]*w[0];
  Q[1][0] = Q[1][1] + A[1][2]*w[0];
  Q[2][0] = ((A[0][0].real()) - w[0]) * ((A[1][1].real()) - w[0]) - Q[2][1];
  norm    = SQR_ABS(Q[0][0]) + SQR_ABS(Q[1][0]) + SQR(Q[2][0].real());

  /// If vectors are nearly linearly dependent, or if there might have
  /// been large cancellations in the calculation of A(I,I) - W(1), fall
  /// back to QL algorithm
  /// Note that this simultaneously ensures that multiple eigenvalues do
  /// not cause problems: If W(1) = W(2), then A - W(1) * I has rank 1,
  /// i.e. all columns of A - W(1) * I are linearly dependent.
  if (norm <= error)
    return zheevq3(A, Q, w);
  else                      /// This is the standard branch
  {
    norm = sqrt(1.0 / norm);
    for (j=0; j < 3; j++)
      Q[j][0] = Q[j][0] * norm;
  }
  
  /// Calculate second eigenvector by the formula
  ///   v[1] = conj( (A - w[1]).e1 x (A - w[1]).e2 )
  Q[0][1]  = Q[0][1] + A[0][2]*w[1];
  Q[1][1]  = Q[1][1] + A[1][2]*w[1];
  Q[2][1]  = ((A[0][0].real()) - w[1]) * ((A[1][1].real()) - w[1]) - (Q[2][1].real());
  norm     = SQR_ABS(Q[0][1]) + SQR_ABS(Q[1][1]) + SQR((Q[2][1].real()));
  if (norm <= error)
    return zheevq3(A, Q, w);
  else
  {
    norm = sqrt(1.0 / norm);
    for (j=0; j < 3; j++)
      Q[j][1] = Q[j][1] * norm;
  }
  
  /// Calculate third eigenvector according to
  ///   v[2] = conj(v[0] x v[1])
  Q[0][2] = conj(Q[1][0]*Q[2][1] - Q[2][0]*Q[1][1]);
  Q[1][2] = conj(Q[2][0]*Q[0][1] - Q[0][0]*Q[2][1]);
  Q[2][2] = conj(Q[0][0]*Q[1][1] - Q[1][0]*Q[0][1]);
#endif

  return 0;
}


#endif
/// ----------------------------------------------------------------------------
/// Numerical diagonalization of 3x3 matrcies
/// Copyright (C) 2006  Joachim Kopp
/// ----------------------------------------------------------------------------
/// This library is free software; you can redistribute it and/or
/// modify it under the terms of the GNU Lesser General Public
/// License as published by the Free Software Foundation; either
/// version 2.1 of the License, or (at your option) any later version.
///
/// This library is distributed in the hope that it will be useful,
/// but WITHOUT ANY WARRANTY; without even the implied warranty of
/// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
/// Lesser General Public License for more details.
///
/// You should have received a copy of the GNU Lesser General Public
/// License along with this library; if not, write to the Free Software
/// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
/// ----------------------------------------------------------------------------

