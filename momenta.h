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
 * File: momenta.h
 * Authors: P. Korcyl
 * Contact: piotr.korcyl@uj.edu.pl
 * 
 * Version: 1.0
 * 
 * Description:
 * Precalculating linear and sin based lattice momenta
 * 
 */


#ifndef H_MOMENTA
#define H_MOMENTA

#include <stdlib.h>
#include <iostream>

#include "config.h"

#include <mpi.h>

class momenta {

private:

	double* tbl_phat2;
	double* tbl_phatx;
	double* tbl_phaty;
	double* tbl_pbar2;
	double* tbl_pbarx;
	double* tbl_pbary;

	int Nxl, Nyl;
	int pos_x, pos_y;

public:

	momenta(config* cnfg, mpi_class* mpi){

		tbl_phat2 = (double*)malloc(cnfg->Nxl*cnfg->Nyl*sizeof(double));
		tbl_phatx = (double*)malloc(cnfg->Nxl*cnfg->Nyl*sizeof(double));
		tbl_phaty = (double*)malloc(cnfg->Nxl*cnfg->Nyl*sizeof(double));
		tbl_pbar2 = (double*)malloc(cnfg->Nxl*cnfg->Nyl*sizeof(double));
		tbl_pbarx = (double*)malloc(cnfg->Nxl*cnfg->Nyl*sizeof(double));
		tbl_pbary = (double*)malloc(cnfg->Nxl*cnfg->Nyl*sizeof(double));

		Nxl = cnfg->Nxl;
		Nyl = cnfg->Nyl;

		pos_x = mpi->getPosX();
		pos_y = mpi->getPosY();
	}

	~momenta(){

		free(tbl_phat2);
		free(tbl_phatx);
		free(tbl_phaty);
		free(tbl_pbar2);
		free(tbl_pbarx);
		free(tbl_pbary);

	}

	int set();

	double phat2(int i);
	double pbarX(int i);
	double pbarY(int i);

};

int momenta::set(){

	int i,j;
	int ig, jg;
	double sargx, sargy;

	for(i = 0; i < Nxl; i++){
		for(j = 0; j < Nyl; j++){

			ig = (i+pos_x*Nxl);
			jg = (j+pos_y*Nyl);


    			//constructs the different kinds of momenta that are needed: 
    			//phat2 = sum_\mu 4 sin(pi k_\mu / N_\mu)^2
    			//pbar2 = sum_\mu sin(2pi k_\mu / N_\mu)^2
    			//pbar_\mu = sin(2pi k_\mu / N_\mu)

			sargx = sin( M_PI * ig / (1.0 * Nx) );
			sargy = sin( M_PI * jg / (1.0 * Ny) );

			tbl_phat2[i*Nyl+j] = 4.0*pow(sargx,2.0) + 4.0*pow(sargy,2.0);
			tbl_phatx[i*Nyl+j] = sargx;
			tbl_phaty[i*Nyl+j] = sargy;

			sargx = sin( 2.0 * M_PI * ig / (1.0 * Nx) );
			sargy = sin( 2.0 * M_PI * jg / (1.0 * Ny) );

			tbl_pbar2[i*Nyl+j] = pow(sargx,2.0) + pow(sargy,2.0);
			tbl_pbarx[i*Nyl+j] = sargx;
			tbl_pbary[i*Nyl+j] = sargy;
		}
	}

return 1;
}

double momenta::phat2(int i){

	return tbl_phat2[i]; 
}

double momenta::pbarX(int i){

	return tbl_pbarx[i]; 
}

double momenta::pbarY(int i){

	return tbl_pbary[i]; 
}
#endif
