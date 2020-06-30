#include <stdio.h>
#include <complex>
#include <fftw3.h>
#include <fftw3-mpi.h>

#define size 10



int main(void){

std::complex<double> array[size];
fftw_complex table[size];

for(int i = 0; i < size; i++){

	array[i] = std::complex<double>(1.0*i,2.0*i);
	table[1][i] = 0.0;
	table[0][i] = 0.0;
}

fftw_complex* ptr;
//std::complex<double>* ptrc;

//ptrc = array;

for(int i = 0; i < size; i++){

	ptr = reinterpret_cast<fftw_complex*>(array);

	printf("array[%i] = %f %f\n", i, array->real(), array->imag());

	ptr++;

	printf("table[%i] = %f %f\n", i, (*ptr)[0], (*ptr)[1]);

	array++;

}

return 1;
}
