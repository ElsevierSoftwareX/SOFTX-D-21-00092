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
 * File: mpi_class.h
 * Authors: P. Korcyl
 * Contact: piotr.korcyl@uj.edu.pl
 * 
 * Version: 1.0
 * 
 * Description:
 * Class containing mpi setup
 * 
 */

#ifndef H_MPI_CLASS
#define H_MPI_CLASS

#include <stdlib.h>
#include <iostream>

#include "config.h"

#include <mpi.h>

class mpi_class {

private: 

    int size, rank;

    int proc_grid[2];

    int seed;

    MPI_Comm row_comm;
    MPI_Comm col_comm;

    int XNeighbourNext, XNeighbourPrevious;
    int YNeighbourNext, YNeighbourPrevious;

    int ExchangeX, ExchangeY;

    //proc_grid = y + procy * x
    int pos_x;
    int pos_y;

    int n_pos_x, n_pos_y;
    int p_pos_x, p_pos_y;

public:

    mpi_class(int argc, char *argv[]){

    	printf("Class constructor\n");

  	int provided;
    	MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    	printf("MPI initialized\n");

    	MPI_Comm_size(MPI_COMM_WORLD, &size);

    	printf("MPI Comm size: %i \n", size);

    	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    	printf("MPI rank: %i \n", rank);

    	proc_grid[0] = atoi(argv[1]);
    	proc_grid[1] = 1;

    	//MPI_Barrier(MPI_COMM_WORLD);

    }

    mpi_class(){

    	printf("Null class constructor\n");

    }

    int mpi_init(config* cnfg);

    int mpi_exchange_grid();

    int mpi_exchange_groups();

    int getRank(){

	return rank;
    }

    int getSize(){

	return size;
    }

    int getSeed(){

	return seed;
    }


    MPI_Comm getRowComm(){

	return row_comm;
    }

    MPI_Comm getColComm(){

	return col_comm;
    }

    int getExchangeX(){
	
	return ExchangeX;

    }

    int getExchangeY(){
	
	return ExchangeY;

    }

    int getXNeighbourNext(){

	return XNeighbourNext;

    }	

    int getYNeighbourNext(){

	return YNeighbourNext;

    }	

    int getXNeighbourPrevious(){

	return XNeighbourPrevious;

    }	

    int getYNeighbourPrevious(){

	return YNeighbourPrevious;

    }	

    int getPosX(){

	return pos_x;
    }

    int getPosY(){

	return pos_y;
    }

};

#endif

