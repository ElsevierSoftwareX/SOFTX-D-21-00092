#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include "config.h"
#include "mpi_pos.h"

int mpi_split(double* global, double* local){

/*
    int size, rank, tmprank;

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //proc_grid = z + procz * y + procz * procy * x
    int pos_x = rank/(procz*procy);
    tmprank = rank - pos_x * (procz*procy);
    int pos_y = tmprank/procz;
    tmprank = tmprank - pos_y*procz;
    int pos_z = tmprank;

    printf("rank %i has grid position (%i, %i, %i)\n", rank, pos_x, pos_y, pos_z);
*/

    double tmp[Nxl*Nyl*Nzl*Tt];

    MPI_Scatter(global, Nxl*Nyl*Nzl*Tt, MPI_DOUBLE, tmp, Nxl*Nyl*Nzl*Tt, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int tt, xx, yy, zz;

    for(tt = 0; tt < Tt; tt++){
        for(xx = 0; xx < Nxl; xx++){
            for(yy = 0; yy < Nyl; yy++){
                for(zz = 0; zz < Nzl; zz++){

                        local[buf_pos(tt,xx,yy,zz)] = tmp[zz+Nzl*yy+Nzl*Nyl*xx+Nzl*Nyl*Nxl*tt];

                }
            }
        }
    }

/*
    int tt, xx, yy, zz;
    
    for(tt = 0; tt < Tt; tt++){
	for(xx = 0; xx < Nxl; xx++){
	    for(yy = 0; yy < Nyl; yy++){
		for(zz = 0; zz < Nzl; zz++){

			local[buf_pos(tt,xx,yy,zz)] = global[(pos_z + procz*pos_y + procz*procy*pos_x)*Nxl*Nyl*Nzl*Tt + zz + Nzl*yy + Nzl*Nyl*xx + Nzl*Nyl*Nxl*tt];

		}
	    }
	}
    }
*/

return 1;
}
