/* 
 * This file is part of the JIMWLK numerical solution package (https://github.com/piotrkorcyl/jimwlk).
 * Copyright (c) 2020 P. Korcyl
 * 
 * This program is free software: you can redistribute it and/or modify  
 * it under the terms of the GNU General Public License as published by  
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful, but 
 * WITHOUT ANY WARRANTY; without even the implied warranty of 
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License 
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 * 
 * File: mpi_exchange_groups.cpp
 * Authors: P. Korcyl
 * Contact: piotr.korcyl@uj.edu.pl
 * 
 * Version: 1.0
 * 
 * Description:
 * Class grouping mpi ranks to minimize global MPI calls
 * 
 */


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

