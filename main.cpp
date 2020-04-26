#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include <complex.h>

#include "su3_complex.h"
#include "su3_matrix.h"

#include "config.h"
#include "mpi_init.h"
#include "mpi_allocate.h"
#include "mpi_exchange_grid.h"
#include "mpi_exchange_boundaries.h"
#include "mpi_pos.h"
#include "mpi_split.h"
#include "mpi_gather.h"
#include "utils.h"

#include <fftw3.h>
#include <fftw3-mpi.h>

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

    MPI_Barrier(MPI_COMM_WORLD);

    mpi_init(proc_grid[0], proc_grid[1]);

    mpi_exchange_grid();

    printf("Local variables set: Nxl = %i, Nyl = %i\n", Nxl, Nyl);

    mpi_allocate();

    double XX[Nx*Ny];
    double YY[Nx*Ny];

    int ii;
    for(ii = 0; ii < Nx*Ny; ii++){
	XX[ii] = 0;
    }
    for(ii = 0; ii < Nx*Ny; ii++){
	YY[ii] = 0;
    }

    int xx, yy;

    for(xx = 0; xx < Nxl; xx++){
    	for(yy = 0; yy < Nyl; yy++){

		x[buf_pos(xx, yy)] = rank*100+loc_pos(xx, yy) + 1;
	}
    }

    for(xx = 0; xx < Nx; xx++){
    	for(yy = 0; yy < Ny; yy++){

		XX[yy + Ny*xx] = yy + Ny*xx + 1;
	}
    }


    mpi_split(XX, p);

//    print_file(p);

    mpi_exchange_boundaries(p);

//    print_file(p);

    mpi_gather(YY, p);

/*
    for(tt = 0; tt < Tt; tt++){
        for(xx = 0; xx < Nx; xx++){
            for(yy = 0; yy < Ny; yy++){
    	        for(zz = 0; zz < Nz; zz++){

		    printf("YY(%i)[%i + Nz*%i + Nz*Ny*%i + Nz*Ny*Nx*%i = %i] = %f\n", rank, zz, yy, xx, tt, zz + Nz*yy + Nz*Ny*xx + Nz*Ny*Nx*tt, YY[zz + Nz*yy + Nz*Ny*xx + Nz*Ny*Nx*tt]);
		}
	    }
	}
    }
*/

    const ptrdiff_t N0 = Nx, N1 = Ny;
    fftw_plan plan;
    fftw_complex *data;
    ptrdiff_t alloc_local, local_n0, local_0_start, i, j;

    fftw_mpi_init();

    /* get local data size and allocate */
    alloc_local = fftw_mpi_local_size_2d(N0, N1, MPI_COMM_WORLD,
                                         &local_n0, &local_0_start);
    data = fftw_alloc_complex(alloc_local);

    /* create plan for in-place forward DFT */
    plan = fftw_mpi_plan_dft_2d(N0, N1, data, data, MPI_COMM_WORLD,
                                FFTW_FORWARD, FFTW_ESTIMATE);    

    /* initialize data to some function my_function(x,y) */
    for (i = 0; i < local_n0; ++i) for (j = 0; j < N1; ++j){
       data[i*N1 + j][0] = 1.0*(local_0_start + i);
       data[i*N1 + j][1] = 1.0*j;
    }

    /* compute transforms, in-place, as many times as desired */
    fftw_execute(plan);

    fftw_destroy_plan(plan);

    MPI_Finalize();

return 1;
}

