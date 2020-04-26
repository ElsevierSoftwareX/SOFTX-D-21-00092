#ifndef H_SU3_MATRIX
#define H_SU3_MATRIX

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>

#include "su3_complex.h"

template <class T>
class su3_matrix_reduced;

template <class T>
class su3_matrix {
	public:
		su3_complex<T> m[9];

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

		void print() {
				for (int i = 0; i < 9; i++) {
					std::cerr<<"m["<<i<<"].r = "<<m[i].r<<std::endl;
					std::cerr<<"m["<<i<<"].i = "<<m[i].i<<std::endl;
				}
		}

		void toVector(T* r) {
			for (int i = 0; i < 9; i++) {
				r[2 * i] = m[i].r;
				r[2 * i + 1] = m[i].i;
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
						c.m[i*3+j] += this->m[i*3+k] + b.m[k*3+j];
					}
				}
			}

			return c;
		}



		void fromVector(T* r) {
			m[0].r = r[0];
			m[0].i = r[1];
			m[1].r = r[2];
			m[1].i = r[3];
			m[2].r = r[4];
			m[2].i = r[5];
			m[3].r = r[6];
			m[3].i = r[7];
			m[4].r = r[8];
			m[4].i = r[9];
			m[5].r = r[10];
			m[5].i = r[11];
			m[6].r = r[12];
			m[6].i = r[13];
			m[7].r = r[14];
			m[7].i = r[15];
			m[8].r = r[16];
			m[8].i = r[17];
		}

		void expand(su3_matrix_reduced<T>& rm) {

			m[1].r = rm.m[0];
			m[1].i = rm.m[1];
			m[2].r = rm.m[2];
			m[2].i = rm.m[3];

			m[6].r = rm.m[4];
			m[6].i = rm.m[5];

			T r1 = m[1].r*m[1].r;
			T r2 = m[2].r*m[2].r;
			T i1 = m[1].i*m[1].i;
			T i2 = m[2].i*m[2].i;

			T nn = r1 + i1 + r2 + i2; //m[1].r*m[1].r+m[1].i*m[1].i+m[2].r*m[2].r+m[2].i*m[2].i;

			su3_complex<T> a1,a2,a3,b1,c1;

			a2 = m[1];
			a3 = m[2];
			c1 = m[6];

			su3_complex<T> a1c,a2c,a3c,b1c,c1c;

			a2c = a2.conj();
			a3c = a3.conj();
			c1c = c1.conj();

			T a2r = a2.r*a2.r ;
			T a2i = a2.i*a2.i;
			T a3r = a3.r*a3.r;
			T a3i = a3.i*a3.i;

			T x = (a2r + a2i) + (a3r + a3i); // (a2.r*a2.r + a2.i*a2.i) + (a3.r*a3.r + a3.i*a3.i);

			T st = sqrt( 1 - x );

			m[0].r = st*rm.m[6];
			m[0].i = st*rm.m[7];

			T ss = sqrt( x - (c1.r*c1.r + c1.i*c1.i) );

			m[3].r = ss*rm.m[8];
			m[3].i = ss*rm.m[9];

			a1 = m[0];
			b1 = m[3];

			a1c = a1.conj();
			b1c = b1.conj();

			m[4].r = -((c1c * a3c) + ((a1c * b1) * a2)).r / nn;

			m[5].r = ((c1c * a2c) - ((a1c * b1) * a3)).r / nn;

			m[7].r = ((a3c * b1c) - ((a1c * c1) * a2)).r / nn;

			m[8].r = -(((c1 * a1c) * a3) + (b1c * a2c)).r / nn;

			m[4].i = -((c1c * a3c) + ((a1c * b1) * a2)).i / nn;

			m[5].i = ((c1c * a2c) - ((a1c * b1) * a3)).i / nn;

			m[7].i = ((a3c * b1c) - ((a1c * c1) * a2)).i / nn;

			m[8].i = -(((c1 * a1c) * a3) + (b1c * a2c)).i / nn;
		}
};

template <class T>
class su3_matrix_reduced {
	public:
		T m[10];
		
		template <class Ta>
		int reduce(su3_matrix<Ta> fm){
			m[0] = fm.m[1].r;
			m[1] = fm.m[1].i;
			m[2] = fm.m[2].r;
			m[3] = fm.m[2].i;
			m[4] = fm.m[6].r;
			m[5] = fm.m[6].i;

	 		T ss = sqrt(fm.m[0].i*fm.m[0].i+fm.m[0].r*fm.m[0].r);

			//m[6] = static_cast<T>(atan2((double)(fm.m[0].i/ss), (double)(fm.m[0].r/ss)));
			m[6] = fm.m[0].r/ss;
			m[7] = fm.m[0].i/ss;

			T tt = sqrt(fm.m[3].i*fm.m[3].i+fm.m[3].r*fm.m[3].r);

			// m[7] = static_cast<T>(atan2((double)(fm.m[3].i/tt), (double)(fm.m[3].r/tt)));
			m[8] = fm.m[3].r/tt;
			m[9] = fm.m[3].i/tt;
		}

		void toVector(T* r) {
			for (int i = 0; i < 10; i++) {
				r[i] = m[i];
		}

	}

};

#endif
