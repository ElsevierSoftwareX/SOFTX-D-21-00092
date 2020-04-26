#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include "config.h"

int print(double *u){
	
    int xx,yy;

    int rank;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    for(xx = 0; xx < Nxl_buf; xx++){
    	for(yy = 0; yy < Nyl_buf; yy++){

        	printf("b (%i, [%i,%i] = %f ), ", rank, xx, yy, u[yy+Nyl_buf*xx]);

        }
        printf("\n");
    }

return 1;
}

int print_file(double *u){
	
    int xx,yy;

    int rank;

    FILE *f;

    char filename[100];

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    sprintf(filename,"output_rank%i.txt", rank);

    f = fopen(filename,"w+");

    for(xx = 0; xx < Nxl_buf; xx++){
	for(yy = 0; yy < Nyl_buf; yy++){

        	fprintf(f, "(%i, [%i,%i] = %f ), ", rank, xx, yy, u[yy+Nyl_buf*xx]);

        }
        fprintf(f, "\n");
    }

    fclose(f);

return 1;
}

