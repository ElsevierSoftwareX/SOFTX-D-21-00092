#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "config.h"

double theta(double x, double y){

	return x>0?y:0;
}

int max(int x, int y){

	if(x >= y)
		return x;
	else
		return y;
}

int kinematical_constraints(int initial_r_2, int max_evolution, double langevin_step, int *pos_vector, int *starting_points){

//	int Nx = 256;
//	int Ny = Nx;

	//dla sieci 256, potrzebujemy punktow w okolicy 0.03*L dla s=0.05
	//0.03*256 = 8
	//rr=64

	//dla sieci 64, potrzebujemy punktow w okolicy 0.03*L dla s=0.05
	//0.03*64 = 2
	//rr=4


	double rr;

//	int pos_vector[1000];
//	int starting_points[1000];


	int rap_vector[1000][1000];
	int rap_count[1000];
	pos_vector[0] = initial_r_2;
	rap_vector[0][0] = max_evolution;
	rap_count[0] = 1;
	for(int i = 1; i < 1000; i++)
		rap_count[i] = 0;
	int count = 1;


	int iter ;

	//double langevin_step = 0.0004;

for(iter = 0; iter < count; iter++){

	rr = pos_vector[iter];

	int max_rapidity = 0;
	for(int jj = 0; jj < rap_count[iter]; jj++)
		if( rap_vector[iter][jj] > max_rapidity )
			max_rapidity = rap_vector[iter][jj];


	for(int langevin = 0; langevin < max_rapidity; langevin++){

		//printf("size boundary: %f\n", rr*exp(langevin*langevin_step));

		//printf("\n");

		//trivial dependence on rapidity, we don't need to store that!!

//		//let's check if the rapidity is already onn the list
//		int flag_rap = 0;
//		for(int jj = 0; jj < rap_count[iter]; jj++){
//			if( langevin == rap_vector[iter][jj])
//				flag_rap = 1; //we already have it
//		}
//		if( flag_rap == 0 ){
//		//we didn't have it on our list, let's add it
//			rap_vector[iter][rap_count[iter]] = langevin;
//			rap_count[iter]++;
//		}

		//for each wilson line of the transverse plane
		for(int xx_global = 1; xx_global < 2; xx_global++){
			for(int yy_global = 1; yy_global < 2; yy_global++){

				//we calculate the kernel by integrating over z
				for(int xx = xx_global; xx < xx_global+sqrt(rr*exp(langevin*langevin_step)); xx++){
					for(int yy = yy_global; yy < yy_global+sqrt(rr*exp(langevin*langevin_step)); yy++){

			                        double dx = xx_global - xx;
		                                if( dx >= Nx/2 )
                		                      dx = dx - Nx;
                                		if( dx < -Nx/2 )
		                                      dx = dx + Nx;

        		                        double dy = yy_global - yy;
		                                if( dy >= Ny/2 )
                		                        dy = dy - Ny;
                                		if( dy < -Ny/2 )
		                                        dy = dy + Ny;

						double rho = log( (dx*dx + dy*dy) / (1.0*rr) );

						double Delta = theta( sqrt( dx*dx + dy*dy ) - sqrt(rr), rho);

						int RRint = max( dx*dx + dy*dy, rr );


						//double rho = log( ((xx_global - xx)*(xx_global - xx) + (yy_global - yy)*(yy_global - yy))/(1.0*rr));

						//double Delta = theta( sqrt((xx_global - xx)*(xx_global - xx) + (yy_global - yy)*(yy_global - yy)) - sqrt(rr), rho);

						//double R = max( sqrt((xx_global - xx)*(xx_global - xx) + (yy_global - yy)*(yy_global - yy)), sqrt(rr) );

						//kinematical constraint
				                if( langevin * langevin_step >= rho ){

							//printf("at step %i for wilson line at (%i %i) we sum over (%i %i) requires U at step %f and scale %f (%f)\n", langevin, xx_global, yy_global, xx, yy, (langevin*langevin_step - Delta)/langevin_step, R, R*R);

							int flag = 0;

							//we are evolving scale iter
							//in the evolution we find other scales; let's check if we've already seen that one
							for(int ii = 0; ii < count; ii++){
								//yes, we already have it
								if(pos_vector[ii] == RRint ){
									//we already have it
									flag = 1;
		
									if(ii != iter){

										printf("evolving scale %i at iter = %i (step %i) and searching for scale %i which is already on the list and looking for rapidity %i  (rap difference = %i)\n", pos_vector[iter], iter, langevin, pos_vector[ii], (int)((langevin*langevin_step - Delta)/langevin_step), langevin - (int)((langevin*langevin_step - Delta)/langevin_step));
								
										//let's check if the rapidity is already onn the list
										int flag_rap = 0;
										for(int jj = 0; jj < rap_count[ii]; jj++){
											if( (int)((langevin*langevin_step - Delta)/langevin_step) == rap_vector[ii][jj])
												flag_rap = 1; //we already have it
										}
										if( flag_rap == 0 ){
											//we didn't have it on our list, let's add it
											rap_vector[ii][rap_count[ii]] = (int)((langevin*langevin_step - Delta)/langevin_step);
											rap_count[ii]++;
										}
									}
								}
							}
							if(flag == 0){

								printf("evolving scale %i at iter = %i (step %i) and searching for scale %i which was NOT on the list and looking for rapidity %i  (rap difference = %i)\n", pos_vector[iter], iter, langevin, RRint, (int)((langevin*langevin_step - Delta)/langevin_step), langevin - (int)((langevin*langevin_step - Delta)/langevin_step));

								pos_vector[count] = RRint;
								starting_points[count] = langevin;
								rap_vector[count][0] = (int)((langevin*langevin_step - Delta)/langevin_step);
								rap_count[count]++;
								count++;
							}
						}
					}
				}

			}//yy_global
		}//xx_global

}//langevin

printf("analyzing %i out of %i\n", iter, count);

}//iter

	int sum = 0;

	for(int ii = 0; ii < count; ii++){
		printf("pos_vector[%i] = %i\n", ii, pos_vector[ii]);

		int max_rapidity = 0;
		for(int jj = 0; jj < rap_count[ii]; jj++)
			if( rap_vector[ii][jj] > max_rapidity )
				max_rapidity = rap_vector[ii][jj];

		if( max_rapidity + 1 == rap_count[ii] ){
			printf("all up to max_rap = %i\n", max_rapidity);

		sum += max_rapidity;

		}else{
			int up_to = 0;
			for(int jj = 1; jj < rap_count[ii]-1; jj++){
				printf("%i  ", rap_vector[ii][jj]);
				if(rap_vector[ii][jj+1] == rap_vector[ii][jj] + 1){
					up_to = jj+1;
				}else{
					printf("\n up to %i\n", up_to);
				}
			}
		}
		printf("\n");
	}

	printf("sum of all steps = %i\n", sum);
	printf("memory required = %f GB\n", 1.0*sum*Nx*Nx*9.0*2.0*8.0/1024.0/1024.0/1024.0);

	return count;
}
