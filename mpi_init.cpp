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
 * File: mpi_init.cpp
 * Authors: P. Korcyl
 * Contact: piotr.korcyl@uj.edu.pl
 * 
 * Version: 1.0
 * 
 * Description:
 * Initialization of the MPI setup
 * 
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "mpi_class.h"

#include "config.h"

int mpi_class::mpi_init(config *cnfg) {

    int proc_x = proc_grid[0];
    int proc_y = proc_grid[1];


    //proc_grid = y + procy * x
    pos_x = rank/(proc_y);
    pos_y = rank - pos_x * (proc_y);


    if( proc_x * proc_y != size ){

                if( rank == 0 ){

                        printf("Number of running processes does not match lattice subdivision.\n");
                        printf("Aborting\n");

                }

                MPI_Finalize();

                exit(0);

    }

    if( Nx % proc_x != 0 ){

                if( rank == 0 ){

                        printf("Dimension 0 is not divisible by the number of ranks in direction 0.\n");
                        printf("Aborting\n");

                }

                MPI_Finalize();

                exit(0);
    }

    if( Ny % proc_y != 0 ){

                if( rank == 0 ){

                        printf("Dimension 1 is not divisible by the number of ranks in direction 1.\n");
                        printf("Aborting\n");

                }

                MPI_Finalize();

                exit(0);
    }

    cnfg->Nxl = Nx/proc_x;
    cnfg->Nyl = Ny/proc_y;

    if( cnfg->Nxl == 1 ){

                if( rank == 0 ){

                        printf("Direction 0 has local lattice of size 1.\n");
                        printf("Aborting\n");

                }

                MPI_Finalize();

                exit(0);
    }

    if( cnfg->Nyl == 1 ){

                if( rank == 0 ){

                        printf("Direction 1 has local lattice of size 1.\n");
                        printf("Aborting\n");

                }

                MPI_Finalize();

                exit(0);
    }

    printf("Running with local lattice size: %i %i\n", cnfg->Nxl, cnfg->Nyl);

    if( proc_x == 1 ){
	cnfg->Nxl_buf = Nx;
	ExchangeX = 0;
    }else{
	cnfg->Nxl_buf = cnfg->Nxl + 2;
	ExchangeX = 1;
    }

    if( proc_y == 1 ){
	cnfg->Nyl_buf = Ny;
	ExchangeY = 0;
    }else{
	cnfg->Nyl_buf = cnfg->Nyl + 2;
	ExchangeY = 1;
    }

return 1;
}

