#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "config.h"

int mpi_init( int proc_x, int proc_y, int proc_z ) {

    int size, rank;

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    procx = proc_x;
    procy = proc_y;
    procz = proc_z;

    if( proc_x * proc_y * proc_z != size ){

                if( rank == 0 ){

                        printf("Number of running processes does not match lattice subdivision.\n");
                        printf("Aborting\n");

                }

                MPI_Finalize();

                exit(0);

    }

    if( Nx % proc_x != 0 ){

                if( rank == 0 ){

                        printf("Dimension 0 is not divisible by the number of ranks in direction 0.\n");
                        printf("Aborting\n");

                }

                MPI_Finalize();

                exit(0);
    }

    if( Ny % proc_y != 0 ){

                if( rank == 0 ){

                        printf("Dimension 1 is not divisible by the number of ranks in direction 1.\n");
                        printf("Aborting\n");

                }

                MPI_Finalize();

                exit(0);
    }

    if( Nz % proc_z != 0 ){

                if( rank == 0 ){

                        printf("Dimension 2 is not divisible by the number of ranks in direction 2.\n");
                        printf("Aborting\n");

                }

                MPI_Finalize();

                exit(0);
    }


    Nxl = Nx/proc_x;
    Nyl = Ny/proc_y;
    Nzl = Nz/proc_z;

    if( Nxl == 1 ){

                if( rank == 0 ){

                        printf("Direction 0 has local lattice of size 1.\n");
                        printf("Aborting\n");

                }

                MPI_Finalize();

                exit(0);
    }

    if( Nyl == 1 ){

                if( rank == 0 ){

                        printf("Direction 1 has local lattice of size 1.\n");
                        printf("Aborting\n");

                }

                MPI_Finalize();

                exit(0);
    }

    if( Nzl == 1 ){

                if( rank == 0 ){

                        printf("Direction 2 has local lattice of size 1.\n");
                        printf("Aborting\n");

                }

                MPI_Finalize();

                exit(0);
    }

    printf("Running with local lattice size: %i %i %i\n", Nxl, Nyl, Nzl);

    if( proc_x == 1 ){
	Nxl_buf = Nx;
	ExchangeX = 0;
    }else{
	Nxl_buf = Nxl + 2;
	ExchangeX = 1;
    }

    if( proc_y == 1 ){
	Nyl_buf = Ny;
	ExchangeY = 0;
    }else{
	Nyl_buf = Nyl + 2;
	ExchangeY = 1;
    }
    if( proc_z == 1 ){
	Nzl_buf = Nz;
	ExchangeZ = 0;
    }else{
	Nzl_buf = Nzl + 2;
	ExchangeZ = 1;
    }



return 1;
}

