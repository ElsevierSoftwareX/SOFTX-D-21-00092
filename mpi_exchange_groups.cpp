#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "config.h"

#include "mpi_class.h"

int mpi_class::mpi_exchange_groups(void) {


    int color_row = rank / proc_grid[1];
    int color_col = rank % proc_grid[1];

//    MPI_Comm row_comm;
    MPI_Comm_split(MPI_COMM_WORLD, color_row, rank, &row_comm);

    int row_rank, row_size;
    MPI_Comm_rank(row_comm, &row_rank);
    MPI_Comm_size(row_comm, &row_size);

    printf("world rank/size: %d/%d \t row rank/size: %d/%d\n", rank, size, row_rank, row_size);


//    MPI_Comm col_comm;
    MPI_Comm_split(MPI_COMM_WORLD, color_col, rank, &col_comm);

    int col_rank, col_size;
    MPI_Comm_rank(col_comm, &col_rank);
    MPI_Comm_size(col_comm, &col_size);

    printf("world rank/size: %d/%d \t col rank/size: %d/%d\n", rank, size, col_rank, col_size);

return 1;
}

