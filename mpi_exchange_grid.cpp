#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "config.h"

#include "mpi_class.h"

int mpi_class::mpi_exchange_grid(void) {

    printf("rank %i has grid position (%i, %i)\n", rank, pos_x, pos_y);

    //next
    n_pos_x = pos_x, n_pos_y = pos_y;
    //previous
    p_pos_x = pos_x, p_pos_y = pos_y;

    if( ExchangeX == 1 ){

	n_pos_x = pos_x + 1;
	if(n_pos_x == proc_grid[0])
		n_pos_x = 0;
	p_pos_x = pos_x - 1;
	if(p_pos_x == -1)
		p_pos_x = proc_grid[0]-1;

    }
    if( ExchangeY == 1 ){

	n_pos_y = pos_y + 1;
	if(n_pos_y == proc_grid[1])
		n_pos_y = 0;
	p_pos_y = pos_y - 1;
	if(p_pos_y == -1)
		p_pos_y = proc_grid[1]-1;

    }

    XNeighbourNext     = pos_y + proc_grid[1] * n_pos_x;
    XNeighbourPrevious = pos_y + proc_grid[1] * p_pos_x;

    YNeighbourNext     = n_pos_y + proc_grid[1] * pos_x;
    YNeighbourPrevious = p_pos_y + proc_grid[1] * pos_x;


    printf("rank %i has neighbours in X direction %i %i\n", rank, XNeighbourNext, XNeighbourPrevious);
    printf("rank %i has neighbours in Y direction %i %i\n", rank, YNeighbourNext, YNeighbourPrevious);


return 1;
}

