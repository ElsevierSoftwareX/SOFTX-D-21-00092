#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include "config.h"
#include "mpi_pos.h"

int mpi_gather(double* global, double* local){

    int xx, yy;

    double tmp[Nxl*Nyl];

    for(xx = 0; xx < Nxl; xx++){
        for(yy = 0; yy < Nyl; yy++){
		tmp[yy+Nyl*xx] = local[buf_pos(xx,yy)];
	}
    }

//    MPI_Gather(tmp, Nxl*Nyl*Nzl*Tt, MPI_DOUBLE, global, Nxl*Nyl*Nzl*Tt, MPI_DOUBLE, 0, MPI_COMM_WORLD);

return 1;
}
