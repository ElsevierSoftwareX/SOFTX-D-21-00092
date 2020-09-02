#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <random>
#include <thread>
#include <omp.h>

int main(){

	int h[1000];

	for(int i = 0; i < 1000; i++)
		h[i] = 0;

        static __thread std::ranlux24* generator = nullptr;
        if (!generator){
                     std::hash<std::thread::id> hasher;
                     generator = new std::ranlux24(clock() + hasher(std::this_thread::get_id()));
        }   
        std::normal_distribution<double> distribution{0.0,1.0};

	for(int i = 0; i < 10000000; i++){
		int t = (int)(100*(2.0*distribution(*generator)))+500;
		if( t > 0 && t < 1000 )
			h[t]++;
	}
	
	for(int i = 0; i < 1000; i++)
		printf("%f %f\n", (i/100.0), (h[i]/10000000.0));


return 1;
}
