#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include "config.h"
#include "mpi_pos.h"

int mpi_exchange_boundaries(double* u){

    int size, rank;

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double *bufor_send_n;
    double *bufor_receive_n;

    double *bufor_send_p;
    double *bufor_receive_p;


    if( ExchangeX == 1 ){

	    int tt, yy, zz; 

	    bufor_send_n = (double*) malloc(Nyl*Nzl*Tt*sizeof(double));
	    bufor_receive_n = (double*) malloc(Nyl*Nzl*Tt*sizeof(double));

	    for(tt = 0; tt < Tt; tt++){
  	        for(yy = 0; yy < Nyl; yy++){
		    for(zz = 0; zz < Nzl; zz++){
			bufor_send_n[yy+Nyl*zz+Nyl*Nzl*tt] = u[buf_pos(tt,Nxl-1,yy,zz)];
		    }
		}
	    }

	    printf("X data exchange: rank %i sending to %i\n", rank, XNeighbourNext);
	    printf("X data exchange: rank %i receiving from %i\n", rank, XNeighbourNext);

	    MPI_Send(bufor_send_n, Nyl*Nzl*Tt, MPI_DOUBLE, XNeighbourNext, 11, MPI_COMM_WORLD);
    	    MPI_Recv(bufor_receive_n, Nyl*Nzl*Tt, MPI_DOUBLE, XNeighbourPrevious, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	    for(tt = 0; tt < Tt; tt++){
  	        for(yy = 0; yy < Nyl; yy++){
		    for(zz = 0; zz < Nzl; zz++){
			u[buf_pos_ex(tt,0,yy,zz)] = bufor_receive_n[yy+Nyl*zz+Nyl*Nzl*tt];
		    }
		}
	    }

   	    bufor_send_p = (double*) malloc(Nyl*Nzl*Tt*sizeof(double));
	    bufor_receive_p = (double*) malloc(Nyl*Nzl*Tt*sizeof(double));

	    for(tt = 0; tt < Tt; tt++){
  	        for(yy = 0; yy < Nyl; yy++){
		    for(zz = 0; zz < Nzl; zz++){
			bufor_send_p[yy+Nyl*zz+Nyl*Nzl*tt] = u[buf_pos(tt,0,yy,zz)];
		    }
		}
	    }
	
 	    printf("X data exchange: rank %i sending to %i\n", rank, XNeighbourPrevious);
	    printf("X data exchange: rank %i receiving to %i\n", rank, XNeighbourPrevious);

	    MPI_Send(bufor_send_p, Nyl*Nzl*Tt, MPI_DOUBLE, XNeighbourPrevious, 12, MPI_COMM_WORLD);
    	    MPI_Recv(bufor_receive_p, Nyl*Nzl*Tt, MPI_DOUBLE, XNeighbourNext, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	    for(tt = 0; tt < Tt; tt++){
  	        for(yy = 0; yy < Nyl; yy++){
		    for(zz = 0; zz < Nzl; zz++){
			u[buf_pos_ex(tt,Nxl+1,yy,zz)] = bufor_receive_p[yy+Nyl*zz+Nyl*Nzl*tt];
		    }
		}
	    }
    }

    if( ExchangeY == 1 ){

	    int tt, xx, zz; 

	    bufor_send_n = (double*) malloc(Nxl*Nzl*Tt*sizeof(double));
	    bufor_receive_n = (double*) malloc(Nxl*Nzl*Tt*sizeof(double));

	    for(tt = 0; tt < Tt; tt++){
  	        for(xx = 0; xx < Nxl; xx++){
		    for(zz = 0; zz < Nzl; zz++){
			bufor_send_n[xx+Nxl*zz+Nxl*Nzl*tt] = u[buf_pos(tt,xx,Nyl-1,zz)];
		    }
		}
	    }

	    printf("Y data exchange: rank %i sending to %i\n", rank, YNeighbourNext);
	    printf("Y data exchange: rank %i receiving from %i\n", rank, YNeighbourNext);

	    MPI_Send(bufor_send_n, Nxl*Nzl*Tt, MPI_DOUBLE, YNeighbourNext, 13, MPI_COMM_WORLD);
    	    MPI_Recv(bufor_receive_n, Nxl*Nzl*Tt, MPI_DOUBLE, YNeighbourPrevious, 13, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	    for(tt = 0; tt < Tt; tt++){
  	        for(xx = 0; xx < Nxl; xx++){
		    for(zz = 0; zz < Nzl; zz++){
			u[buf_pos_ex(tt,xx,0,zz)] = bufor_receive_n[xx+Nxl*zz+Nxl*Nzl*tt];
		    }
		}
	    }

	    bufor_send_p = (double*) malloc(Nxl*Nzl*Tt*sizeof(double));
	    bufor_receive_p = (double*) malloc(Nxl*Nzl*Tt*sizeof(double));

	    for(tt = 0; tt < Tt; tt++){
  	        for(xx = 0; xx < Nxl; xx++){
		    for(zz = 0; zz < Nzl; zz++){
			bufor_send_p[xx+Nxl*zz+Nxl*Nzl*tt] = u[buf_pos(tt,xx,0,zz)];
		    }
		}
	    }
	
 	    printf("Y data exchange: rank %i sending to %i\n", rank, YNeighbourPrevious);
	    printf("Y data exchange: rank %i receiving to %i\n", rank, YNeighbourPrevious);

	    MPI_Send(bufor_send_p, Nxl*Nzl*Tt, MPI_DOUBLE, YNeighbourPrevious, 14, MPI_COMM_WORLD);
    	    MPI_Recv(bufor_receive_p, Nxl*Nzl*Tt, MPI_DOUBLE, YNeighbourNext, 14, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	    for(tt = 0; tt < Tt; tt++){
  	        for(xx = 0; xx < Nxl; xx++){
		    for(zz = 0; zz < Nzl; zz++){
			u[buf_pos_ex(tt,xx,Nyl+1,zz)] = bufor_receive_p[xx+Nxl*zz+Nxl*Nzl*tt];
		    }
		}
	    }
    }

    if( ExchangeZ == 1 ){

	    int tt, xx, yy; 

	    bufor_send_n = (double*) malloc(Nxl*Nyl*Tt*sizeof(double));
	    bufor_receive_n = (double*) malloc(Nxl*Nyl*Tt*sizeof(double));

	    for(tt = 0; tt < Tt; tt++){
  	        for(xx = 0; xx < Nxl; xx++){
		    for(yy = 0; yy < Nyl; yy++){
			bufor_send_n[xx+Nxl*yy+Nxl*Nyl*tt] = u[buf_pos(tt,xx,yy,Nzl-1)];
		    }
		}
	    }

	    printf("Z data exchange: rank %i sending to %i\n", rank, ZNeighbourNext);
	    printf("Z data exchange: rank %i receiving from %i\n", rank, ZNeighbourNext);

	    MPI_Send(bufor_send_n, Nxl*Nyl*Tt, MPI_DOUBLE, ZNeighbourNext, 15, MPI_COMM_WORLD);
    	    MPI_Recv(bufor_receive_n, Nxl*Nyl*Tt, MPI_DOUBLE, ZNeighbourPrevious, 15, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	    for(tt = 0; tt < Tt; tt++){
  	        for(xx = 0; xx < Nxl; xx++){
		    for(yy = 0; yy < Nyl; yy++){
			u[buf_pos_ex(tt,xx,yy,0)] = bufor_receive_n[xx+Nxl*yy+Nxl*Nyl*tt];
		    }
		}
	    }

	    bufor_send_p = (double*) malloc(Nxl*Nyl*Tt*sizeof(double));
	    bufor_receive_p = (double*) malloc(Nxl*Nyl*Tt*sizeof(double));

	    for(tt = 0; tt < Tt; tt++){
  	        for(xx = 0; xx < Nxl; xx++){
		    for(yy = 0; yy < Nyl; yy++){
			bufor_send_p[xx+Nxl*yy+Nxl*Nyl*tt] = u[buf_pos(tt,xx,yy,0)];
		    }
		}
	    }
	
 	    printf("Z data exchange: rank %i sending to %i\n", rank, ZNeighbourPrevious);
	    printf("Z data exchange: rank %i receiving to %i\n", rank, ZNeighbourPrevious);

	    MPI_Send(bufor_send_p, Nxl*Nyl*Tt, MPI_DOUBLE, ZNeighbourPrevious, 16, MPI_COMM_WORLD);
    	    MPI_Recv(bufor_receive_p, Nxl*Nyl*Tt, MPI_DOUBLE, ZNeighbourNext, 16, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	    for(tt = 0; tt < Tt; tt++){
  	        for(xx = 0; xx < Nxl; xx++){
		    for(yy = 0; yy < Nyl; yy++){
			u[buf_pos_ex(tt,xx,yy,Nzl+1)] = bufor_receive_p[xx+Nxl*yy+Nxl*Nyl*tt];
		    }
		}
	    }
    }

return 1;
}

