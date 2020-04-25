#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "config.h"

int mpi_exchange_grid(void) {

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

    //next
    int n_pos_x = pos_x, n_pos_y = pos_y, n_pos_z = pos_z;
    //previous
    int p_pos_x = pos_x, p_pos_y = pos_y, p_pos_z = pos_z;

    if( ExchangeX == 1 ){

	n_pos_x = pos_x + 1;
	if(n_pos_x == procx)
		n_pos_x= 0;
	p_pos_x = pos_x - 1;
	if(p_pos_x == -1)
		p_pos_x = procx-1;

    }
    if( ExchangeY == 1 ){

	n_pos_y = pos_y + 1;
	if(n_pos_y == procy)
		n_pos_y = 0;
	p_pos_y = pos_y - 1;
	if(p_pos_y == -1)
		p_pos_y = procy-1;

    }
    if( ExchangeZ == 1 ){

	n_pos_z = pos_z + 1;
	if(n_pos_z == procz)
		n_pos_z = 0;
	p_pos_z = pos_z - 1;
	if(p_pos_z == -1)
		p_pos_z = procz-1;
    }

    XNeighbourNext     = pos_z + procz * pos_y + procz*procy * n_pos_x;
    XNeighbourPrevious = pos_z + procz * pos_y + procz*procy * p_pos_x;

    YNeighbourNext     = pos_z + procz * n_pos_y + procz*procy * pos_x;
    YNeighbourPrevious = pos_z + procz * p_pos_y + procz*procy * pos_x;

    ZNeighbourNext     = n_pos_z + procz * pos_y + procz*procy * pos_x;
    ZNeighbourPrevious = p_pos_z + procz * pos_y + procz*procy * pos_x;

 
    printf("rank %i has neighbours in X direction %i %i\n", rank, XNeighbourNext, XNeighbourPrevious);
    printf("rank %i has neighbours in Y direction %i %i\n", rank, YNeighbourNext, YNeighbourPrevious);
    printf("rank %i has neighbours in Z direction %i %i\n", rank, ZNeighbourNext, ZNeighbourPrevious);


return 1;
}

