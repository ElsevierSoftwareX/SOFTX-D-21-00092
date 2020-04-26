#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "config.h"

int mpi_allocate( void ) {

    int size, rank;

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if( Nxl == 0 || Nyl == 0 ){

                if( rank == 0 ){

                        printf("Local lattice dimensions not initialized\n");
                        printf("Aborting\n");

                }

                MPI_Finalize();

                exit(0);
    }

    if( Nxl_buf == 0 || Nyl_buf == 0 ){

                if( rank == 0 ){

                        printf("Local bufor sizes not initialized\n");
                        printf("Aborting\n");

                }

                MPI_Finalize();

                exit(0);
    }

    x = (double*) malloc(Nxl_buf*Nyl_buf*sizeof(double));
    p = (double*) malloc(Nxl_buf*Nyl_buf*sizeof(double));

return 1;
}

