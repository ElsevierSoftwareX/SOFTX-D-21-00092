#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include "config.h"
#include "mpi_pos.h"

int mpi_gather(double* global, double* local){

    int tt, xx, yy, zz;

    double tmp[Nxl*Nyl*Nzl*Tt];

    for(tt = 0; tt < Tt; tt++){
	for(xx = 0; xx < Nxl; xx++){
	    for(yy = 0; yy < Nyl; yy++){
		for(zz = 0; zz < Nzl; zz++){

			tmp[zz+Nzl*yy+Nzl*Nyl*xx+Nzl*Nyl*Nxl*tt] = local[buf_pos(tt,xx,yy,zz)];

		}
	    }
	}
    }

    MPI_Gather(tmp, Nxl*Nyl*Nzl*Tt, MPI_DOUBLE, global, Nxl*Nyl*Nzl*Tt, MPI_DOUBLE, 0, MPI_COMM_WORLD);

return 1;
}
