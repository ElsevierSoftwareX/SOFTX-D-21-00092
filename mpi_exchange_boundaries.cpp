/* 
 * This file is part of the JIMWLK numerical solution package (https://bitbucket.org/piotrekkorcyl/jimwlk.git).
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
 * File: mpi_exchange_boundaries.cpp
 * Authors: P. Korcyl
 * Contact: piotr.korcyl@uj.edu.pl
 * 
 * Version: 2.0
 * 
 * Description:
 * Functionality for exchanging boundary data between neighbouring MPI ranks, not used at the moment
 * 
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include "config.h"
#include "mpi_pos.h"
#include "mpi_class.h"

#include "field.h"

template<class T> int lfield<T>::mpi_exchange_boundaries(mpi_class* mpi){

    int size, rank;

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double *bufor_send_n;
    double *bufor_receive_n;

    double *bufor_send_p;
    double *bufor_receive_p;


    if( ExchangeX == 1 ){

	    int yy; 

	    bufor_send_n = (double*) malloc(Nyl_buf*sizeof(double));
	    bufor_receive_n = (double*) malloc(Nyl_buf*sizeof(double));

  	    for(yy = 0; yy < Nyl; yy++){
		bufor_send_n[yy] = u[buf_pos(Nxl-1,yy)];
	    }

	    printf("X data exchange: rank %i sending to %i\n", rank, XNeighbourNext);
	    printf("X data exchange: rank %i receiving from %i\n", rank, XNeighbourNext);

	    MPI_Send(bufor_send_n, Nyl_buf, MPI_DOUBLE, XNeighbourNext, 11, MPI_COMM_WORLD);
    	    MPI_Recv(bufor_receive_n, Nyl_buf, MPI_DOUBLE, XNeighbourPrevious, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  	    for(yy = 0; yy < Nyl; yy++){
		u[buf_pos_ex(0,yy)] = bufor_receive_n[yy];
	    }

   	    bufor_send_p = (double*) malloc(Nyl_buf*sizeof(double));
	    bufor_receive_p = (double*) malloc(Nyl_buf*sizeof(double));

	    for(yy = 0; yy < Nyl; yy++){
		bufor_send_p[yy] = u[buf_pos(0,yy)];
	    }
	
 	    printf("X data exchange: rank %i sending to %i\n", rank, XNeighbourPrevious);
	    printf("X data exchange: rank %i receiving to %i\n", rank, XNeighbourPrevious);

	    MPI_Send(bufor_send_p, Nyl_buf, MPI_DOUBLE, XNeighbourPrevious, 12, MPI_COMM_WORLD);
    	    MPI_Recv(bufor_receive_p, Nyl_buf, MPI_DOUBLE, XNeighbourNext, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  	    for(yy = 0; yy < Nyl; yy++){
		u[buf_pos_ex(Nxl+1,yy)] = bufor_receive_p[yy];
	    }
    }

    if( ExchangeY == 1 ){

	    int xx; 

	    bufor_send_n = (double*) malloc(Nxl_buf*sizeof(double));
	    bufor_receive_n = (double*) malloc(Nxl_buf*sizeof(double));

  	    for(xx = 0; xx < Nxl; xx++){
		bufor_send_n[xx] = u[buf_pos(xx,Nyl-1)];
	    }

	    printf("Y data exchange: rank %i sending to %i\n", rank, YNeighbourNext);
	    printf("Y data exchange: rank %i receiving from %i\n", rank, YNeighbourNext);

	    MPI_Send(bufor_send_n, Nxl_buf, MPI_DOUBLE, YNeighbourNext, 13, MPI_COMM_WORLD);
    	    MPI_Recv(bufor_receive_n, Nxl_buf, MPI_DOUBLE, YNeighbourPrevious, 13, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for(xx = 0; xx < Nxl; xx++){
		u[buf_pos_ex(xx,0)] = bufor_receive_n[xx];
	    }

	    bufor_send_p = (double*) malloc(Nxl_buf*sizeof(double));
	    bufor_receive_p = (double*) malloc(Nxl_buf*sizeof(double));

	    for(xx = 0; xx < Nxl; xx++){
		bufor_send_p[xx] = u[buf_pos(xx,0)];
	    }
	
 	    printf("Y data exchange: rank %i sending to %i\n", rank, YNeighbourPrevious);
	    printf("Y data exchange: rank %i receiving to %i\n", rank, YNeighbourPrevious);

	    MPI_Send(bufor_send_p, Nxl_buf, MPI_DOUBLE, YNeighbourPrevious, 14, MPI_COMM_WORLD);
    	    MPI_Recv(bufor_receive_p, Nxl_buf, MPI_DOUBLE, YNeighbourNext, 14, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  	    for(xx = 0; xx < Nxl; xx++){
		u[buf_pos_ex(xx,Nyl+1)] = bufor_receive_p[xx];
	    }
    }

return 1;
}

