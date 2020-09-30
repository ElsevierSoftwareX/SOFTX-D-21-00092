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
 * File: positions.h
 * Authors: P. Korcyl
 * Contact: piotr.korcyl@uj.edu.pl
 * 
 * Version: 1.0
 * 
 * Description:
 * Class containing precomputed linear and sin positions
 * 
 */

#ifndef H_POSITIONS
#define H_POSITIONS

#include <stdlib.h>
#include <iostream>

#include "config.h"

#include <mpi.h>

class positions {

private:

	double* tbl_xhatx = NULL;
	double* tbl_xhaty = NULL;
	double* tbl_xbar2 = NULL;

	int Nxg, Nyg;
	int pos_x, pos_y;

public:

	positions(config* cnfg, mpi_class* mpi){

		tbl_xhatx = (double*)malloc(Nx*Ny*sizeof(double));
		tbl_xhaty = (double*)malloc(Nx*Ny*sizeof(double));
		tbl_xbar2 = (double*)malloc(Nx*Ny*sizeof(double));

  	        if(tbl_xhatx == NULL){
		    printf("positions constructor: malloc unsuccessful. Aborting.\n");
		    exit(0);
	        }
  	        if(tbl_xhaty == NULL){
		    printf("positions constructor: malloc unsuccessful. Aborting.\n");
		    exit(0);
	        }
  	        if(tbl_xbar2 == NULL){
		    printf("positions constructor: malloc unsuccessful. Aborting.\n");
		    exit(0);
	        }


		Nxg = Nx;
		Nyg = Ny;

		pos_x = mpi->getPosX();
		pos_y = mpi->getPosY();

	}

	positions(const positions &in){

		printf("Copy constructor with Nxl = %i and Nyl = %i\n", in.Nxg, in.Nyg);

		this->tbl_xhatx = (double*)malloc(in.Nxg*in.Nyg*sizeof(double));
		this->tbl_xhaty = (double*)malloc(in.Nxg*in.Nyg*sizeof(double));
		this->tbl_xbar2 = (double*)malloc(in.Nxg*in.Nyg*sizeof(double));

//		this->set();

		for(int i = 0; i < in.Nxg*in.Nyg; i++){
			this->tbl_xhatx[i] = in.tbl_xhatx[i];
			this->tbl_xhaty[i] = in.tbl_xhaty[i];
			this->tbl_xbar2[i] = in.tbl_xbar2[i];
		}

		this->Nxg = in.Nxg;
		this->Nyg = in.Nyg;

		this->pos_x = in.pos_x;
		this->pos_y = in.pos_y;

	}

	~positions(){

		free(tbl_xhatx);
		free(tbl_xhaty);
		free(tbl_xbar2);
	}

	int set();

	double xhatX(int i);
	double xhatY(int i);
	double xbar2(int i);


};

int positions::set(){

	int i,j;
	int ig, jg;
	double sargx, sargy;

	for(i = 0; i < Nx; i++){
		for(j = 0; j < Ny; j++){

			//ig = (i+pos_x*Nxl);
			//jg = (j+pos_y*Nyl);


                        //double dx2 = Nx*sin(M_PI*(x_global-xx)/Nx)/M_PI;
                        //double dy2 = Ny*sin(M_PI*(y_global-yy)/Ny)/M_PI;
                        //double dx = 0.5*Nx*sin(2.0*M_PI*(x_global-xx)/Nx)/M_PI;
                        //double dy = 0.5*Ny*sin(2.0*M_PI*(y_global-yy)/Ny)/M_PI;

		        //double rrr = 1.0*(dx2*dx2+dy2*dy2);
                      
                        //int ii = fabs(x_global - xx)*Nyg + fabs(y_global - yy);

                        //double dx = pos->xhatX(ii);
                        //double dy = pos->xhatY(ii);

                        //double rrr = pos->xbar2(ii);


			sargx = 0.5*Nx * sin( 2.0*M_PI * i / (1.0 * Nx) ) / M_PI;
			sargy = 0.5*Ny * sin( 2.0*M_PI * j / (1.0 * Ny) ) / M_PI;

			//tbl_xhat2[i*Nyl+j] = 4.0*pow(sargx,2.0) + 4.0*pow(sargy,2.0);
			tbl_xhatx[i*Ny+j] = sargx;
			tbl_xhaty[i*Ny+j] = sargy;

			sargx = Nx * sin( M_PI * i / (1.0 * Nx) ) / M_PI;
			sargy = Ny * sin( M_PI * j / (1.0 * Ny) ) / M_PI;

			tbl_xbar2[i*Ny+j] = pow(sargx,2.0) + pow(sargy,2.0);
			//tbl_xbarx[i*Nyl+j] = sargx;
			//tbl_xbary[i*Nyl+j] = sargy;
		}
	}

return 1;
}

//double positions::xhat2(int i){
//
//	return tbl_xhat2[i]; 
//}

//double positions::xbarX(int i){
//
//	return tbl_xbarx[i]; 
//}

//double positions::xbarY(int i){
//
//	return tbl_xbary[i]; 
//}

double positions::xbar2(int i){

	return tbl_xbar2[i]; 
}

double positions::xhatX(int i){

	return tbl_xhatx[i]; 
}

double positions::xhatY(int i){

	return tbl_xhaty[i]; 
}
#endif
