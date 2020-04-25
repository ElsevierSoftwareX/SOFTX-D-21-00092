#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include "config.h"
#include "mpi_init.h"
#include "mpi_allocate.h"
#include "mpi_exchange_grid.h"
#include "mpi_exchange_boundaries.h"
#include "mpi_pos.h"
#include "mpi_split.h"
#include "mpi_gather.h"
#include "utils.h"

int main(int argc, char *argv[]) {

    MPI_Init(NULL, NULL);

    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    printf("Welcome info: rank %d out of %d processors\n", rank, size);

    int proc_grid[3];
    proc_grid[0] = atoi(argv[1]);
    proc_grid[1] = atoi(argv[2]);
    proc_grid[2] = atoi(argv[3]);

    MPI_Barrier(MPI_COMM_WORLD);

    mpi_init(proc_grid[0], proc_grid[1], proc_grid[2]);

    mpi_exchange_grid();

    printf("Local variables set: Nxl = %i, Nyl = %i, Nzl = %i\n", Nxl, Nyl, Nzl);

    mpi_allocate();

    double XX[Nx*Ny*Nz*Tt];
    double YY[Nx*Ny*Nz*Tt];

    int ii;
    for(ii = 0; ii < Nx*Ny*Nz*Tt; ii++){
	XX[ii] = 0;
    }
    for(ii = 0; ii < Nx*Ny*Nz*Tt; ii++){
	YY[ii] = 0;
    }

    int tt, xx, yy, zz;

    for(tt = 0; tt < Tt; tt++){
        for(xx = 0; xx < Nxl; xx++){
            for(yy = 0; yy < Nyl; yy++){
    	        for(zz = 0; zz < Nzl; zz++){

		    x[buf_pos(tt, xx, yy, zz)] = rank*100+loc_pos(tt, xx, yy, zz) + 1;
		}
	    }
	}
    }

    for(tt = 0; tt < Tt; tt++){
        for(xx = 0; xx < Nx; xx++){
            for(yy = 0; yy < Ny; yy++){
    	        for(zz = 0; zz < Nz; zz++){

		    XX[zz + Nz*yy + Nz*Ny*xx + Nz*Ny*Nx*tt] = zz + Nz*yy + Nz*Ny*xx + Nz*Ny*Nx*tt + 1;
		}
	    }
	}
    }


    mpi_split(XX, p);

//    print_file(p);

    mpi_exchange_boundaries(p);

//    print_file(p);

    mpi_gather(YY, p);


    for(tt = 0; tt < Tt; tt++){
        for(xx = 0; xx < Nx; xx++){
            for(yy = 0; yy < Ny; yy++){
    	        for(zz = 0; zz < Nz; zz++){

		    printf("YY(%i)[%i + Nz*%i + Nz*Ny*%i + Nz*Ny*Nx*%i = %i] = %f\n", rank, zz, yy, xx, tt, zz + Nz*yy + Nz*Ny*xx + Nz*Ny*Nx*tt, YY[zz + Nz*yy + Nz*Ny*xx + Nz*Ny*Nx*tt]);
		}
	    }
	}
    }



    MPI_Finalize();

return 1;
}

