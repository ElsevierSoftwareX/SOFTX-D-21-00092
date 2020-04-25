#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include "config.h"

int print(double *u){
	
    int tt,xx,yy,zz;

    int rank;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    for(tt = 0; tt < Tt; tt++){
        for(xx = 0; xx < Nxl_buf; xx++){
            for(yy = 0; yy < Nyl_buf; yy++){
                for(zz = 0; zz < Nzl_buf; zz++){

                    printf("b (%i, [%i,%i,%i] = %f ), ", rank, xx, yy, zz, u[zz+Nzl_buf*yy+Nzl_buf*Nyl_buf*xx+tt*Nxl_buf*Nyl_buf*Nzl_buf]);

                }

                printf("\n");
            }

            printf("\n\n");
        }
    }

return 1;
}

int print_file(double *u){
	
    int tt,xx,yy,zz;

    int rank;

    FILE *f;

    char filename[100];

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    sprintf(filename,"output_rank%i.txt", rank);

    f = fopen(filename,"w+");

    for(tt = 0; tt < Tt; tt++){
        for(xx = 0; xx < Nxl_buf; xx++){
            for(yy = 0; yy < Nyl_buf; yy++){
                for(zz = 0; zz < Nzl_buf; zz++){

                    fprintf(f, "(%i, [%i,%i,%i] = %f ), ", rank, xx, yy, zz, u[zz+Nzl_buf*yy+Nzl_buf*Nyl_buf*xx+tt*Nxl_buf*Nyl_buf*Nzl_buf]);

                }

                fprintf(f, "\n");
            }

            fprintf(f, "\n\n");
        }
    }

    fclose(f);

return 1;
}

