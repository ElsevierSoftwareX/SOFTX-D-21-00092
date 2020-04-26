#ifndef H_SU3_VECTOR
#define H_SU3_VECTOR

#include "su3_complex.h"

template <class T>
class su3_vector {
	public:
		su3_complex<T> v[3];

		void setZero() {
			v[0] = 0;
			v[1] = 0;
			v[2] = 0;	
		}

		su3_vector<T> operator!() {
			#pragma HLS inline
			su3_vector<T> c;
			
			for (int j = 0; j < 3; j++) {
			#pragma HLS unroll
				c.v[j].r = -v[j].i;
				c.v[j].i = v[j].r;
			}

			return c;
		}

		su3_vector<T> operator+(const su3_vector<T> &b) {
			#pragma HLS inline
			su3_vector<T> c;

			for (int j = 0; j < 3; j++) {
				#pragma HLS unroll
				c.v[j].i = v[j].i + b.v[j].i;
				c.v[j].r = v[j].r + b.v[j].r;
			}

			return c;
		}

		su3_vector<T> operator-(const su3_vector<T> &b) {
			#pragma HLS inline
			su3_vector<T> c;

			for (int j = 0; j < 3; j++) {
				#pragma HLS unroll
				c.v[j].i = v[j].i - b.v[j].i;
				c.v[j].r = v[j].r - b.v[j].r;
			}

			return c;
		}

		su3_vector plus_times_vector(su3_vector b) {
			#pragma HLS inline
			su3_vector c;

			for (int j = 0; j < 3; j++) {
			#pragma HLS unroll
				c.v[j].i = static_cast<T>(0.5/kappa)*b.v[j].i - static_cast<T>(0.5)*v[j].i;
				c.v[j].r = static_cast<T>(0.5/kappa)*b.v[j].r - static_cast<T>(0.5)*v[j].r;
			}

			return c;
		}

		void print() {
			for (int i = 0; i < 3; i++) {
				if (v[i].i*v[i].i + v[i].r*v[i].r > 10e-8)
					std::cerr<<v[i].i<<" "<<v[i].r<<std::endl;
			}
		}
};


// class su3_vector_high_type {
// public:
// 	complex_high_type v[3];

// 	su3_vector_high_type();

// 	void setZero();
// 	void setOne();
// 	su3_vector_high_type& operator=(const su3_vector_high_type &a){

// 	    if (this != &a) {
// 		this->v[0] = a.v[0];
// 		this->v[1] = a.v[1];
// 		this->v[2] = a.v[2];
// 	    }
// 	    return *this;
// 	}
// };

// su3_vector_high_type::su3_vector_high_type() {
// 	}

// void su3_vector_high_type::setZero() {
// 		v[0] = 0;
// 		v[1] = 0;
// 		v[2] = 0;
// 	}

// void su3_vector_high_type::setOne(){
// 		v[0] = 1;
// 		v[1] = 1;
// 		v[2] = 1;
// 	}


// class su3_vector_low_type {
// public:
// 	complex_low_type v[3];

// 	su3_vector_low_type();

// 	void setZero();
// 	void setOne();
// 	su3_vector_low_type& operator=(const su3_vector_low_type &a){

// 	    if (this != &a) {
// 		this->v[0] = a.v[0];
// 		this->v[1] = a.v[1];
// 		this->v[2] = a.v[2];
// 	    }
// 	    return *this;
// 	}
// };

// su3_vector_low_type::su3_vector_low_type() {
// 	}

// void su3_vector_low_type::setZero() {
// 		v[0] = 0;
// 		v[1] = 0;
// 		v[2] = 0;
// 	}

// void su3_vector_low_type::setOne(){
// 		v[0] = 1;
// 		v[1] = 1;
// 		v[2] = 1;
// 	}

#endif
