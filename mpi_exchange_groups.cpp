#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "config.h"

int mpi_exchange_groups(MPI_Comm* row_comm, MPI_Comm* col_comm) {

    int size, rank, tmprank;

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
 
    int color_row = rank / procy;
    int color_col = rank % procy;

//    MPI_Comm row_comm;
    MPI_Comm_split(MPI_COMM_WORLD, color_row, rank, row_comm);

    int row_rank, row_size;
    MPI_Comm_rank(*row_comm, &row_rank);
    MPI_Comm_size(*row_comm, &row_size);

    printf("world rank/size: %d/%d \t row rank/size: %d/%d\n", rank, size, row_rank, row_size);


//    MPI_Comm col_comm;
    MPI_Comm_split(MPI_COMM_WORLD, color_col, rank, col_comm);

    int col_rank, col_size;
    MPI_Comm_rank(*col_comm, &col_rank);
    MPI_Comm_size(*col_comm, &col_size);

    printf("world rank/size: %d/%d \t col rank/size: %d/%d\n", rank, size, col_rank, col_size);

/*
    //proc_grid = y + procy * x
    int pos_x = rank/(procy);
    tmprank = rank - pos_x * (procy);
    int pos_y = tmprank;

    printf("rank %i has grid position (%i, %i)\n", rank, pos_x, pos_y);

    //next
    int n_pos_x = pos_x, n_pos_y = pos_y;
    //previous
    int p_pos_x = pos_x, p_pos_y = pos_y;

    if( ExchangeX == 1 ){

	n_pos_x = pos_x + 1;
	if(n_pos_x == procx)
		n_pos_x = 0;
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

    XNeighbourNext     = pos_y + procy * n_pos_x;
    XNeighbourPrevious = pos_y + procy * p_pos_x;

    YNeighbourNext     = n_pos_y + procy * pos_x;
    YNeighbourPrevious = p_pos_y + procy * pos_x;


    printf("rank %i has neighbours in X direction %i %i\n", rank, XNeighbourNext, XNeighbourPrevious);
    printf("rank %i has neighbours in Y direction %i %i\n", rank, YNeighbourNext, YNeighbourPrevious);
*/

return 1;
}

