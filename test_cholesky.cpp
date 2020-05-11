#include <iostream>
#include <stdlib.h>
#include <math.h>

int main(){

int Nx = 2;
int Ny = 2;

int Nxg = Nx*Ny;
int Nyg = Nx*Ny;

double u[Nxg*Nyg], v[Nxg*Nyg];
double corr[Nx*Ny];

for(int x = 0; x < Nxg; x++){
for(int y = 0; y < Nyg; y++){

		u[x*Nyg+y] = 0.0;
		v[x*Nyg+y] = 0.0;

}
}

printf("printing corr vector\n");
for(int x = 0; x < Nx; x++){
for(int y = 0; y < Ny; y++){

	corr[x*Ny+y] = 1.0/(x*Ny+y+1);
	printf("%f ", corr[x*Ny+y]);
}
printf("\n");
}
printf("done\n");


    //Nxg and Nyg are sizes of the global matrix!!! -> Nxg = Nxg*Nyg
    for (int i = 0; i < Nxg; i++) { //x
        for (int j = 0; j <= i; j++) {  //y

	    printf("element i = %i, j = %i\n", i, j);

            if (j == i) // summation for diagnols 
            { 

	        printf("getting the diagonal element i = %i, j = %i\n", i, j);

                double sum = 0; 

                for (int k = 0; k < i; k++){
		    printf("gathering elements i = %i, k = %i: %f\n", i, k, u[i*Nyg+k]);
                    sum += pow(u[i*Nyg+k], 2); 
		}

	        u[i*Nyg+i] = sqrt(corr[0] - sum);  //distance between i and j = 0
		printf("resulting element: i = %i, i = %i: %f (corr[0] = %f, sum = %f)\n", i, i, u[i*Nyg+i], corr[0], sum);

            } else { 
 
                double sum = 0; 

	        printf("getting the off-diagonal element i = %i, j = %i\n", i, j);

                // Evaluating L(i, j) using L(j, j) 
                for (int k = 0; k < j; k++){
 		    printf("gathering elements i = %i, k = %i and j = %i, k = %i\n", i, k, j, k);
                    sum += (u[i*Nyg+k] * u[j*Nyg+k]); 
		}

		int xi = i/Nyg;
		int yi = i - xi*Nyg;

		int xj = j/Nyg;
		int yj = j - xj*Nyg;

		int ii = abs(xi-xj)*Nyg + abs(yi-yj);

		printf("expected result: %f\n", corr[ii]);
                u[i*Nyg+j] = (corr[ii] - sum) /  //distance between i and j
                                      u[j*Nyg+j]; 
            } 
        } 
    } 



printf("printing u matrix\n");
for(int x = 0; x < Nxg; x++){
for(int y = 0; y < Nyg; y++){

	printf("%f ", u[x*Nyg+y]);
}
printf("\n");
}
printf("done\n");


for(int x = 0; x < Nxg; x++){
for(int y = 0; y < Nyg; y++){


	for(int z = 0; z < Nyg; z++){
		v[x*Nyg+y] += u[x*Nyg+z] * u[y*Nyg+z];
	}
}
}

printf("printing v matrix\n");
for(int x = 0; x < Nxg; x++){
for(int y = 0; y < Nyg; y++){

	printf("%f ", v[x*Nyg+y]);
}
printf("\n");
}
printf("done\n");


printf("printing sigma matrix\n");
for(int x = 0; x < Nxg; x++){
for(int y = 0; y < Nyg; y++){

		int xi = x/Nyg;
		int yi = x - xi*Nyg;

		int xj = y/Nyg;
		int yj = y - xj*Nyg;

		int ii = abs(xi-xj)*Nyg + abs(yi-yj);

    		printf("%f ", corr[ii]);
}
printf("\n");
}
printf("done\n");


return 1;
}

