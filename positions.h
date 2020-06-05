#ifndef H_POSITIONS
#define H_POSITIONS

#include <stdlib.h>
#include <iostream>

#include "config.h"

#include <mpi.h>

class positions {

private:

	double* tbl_xhatx;
	double* tbl_xhaty;
	double* tbl_xbar2;

	int Nxl, Nyl;
	int pos_x, pos_y;

public:

	positions(config* cnfg, mpi_class* mpi){

		tbl_xhatx = (double*)malloc(cnfg->Nxl*cnfg->Nyl*sizeof(double));
		tbl_xhaty = (double*)malloc(cnfg->Nxl*cnfg->Nyl*sizeof(double));
		tbl_xbar2 = (double*)malloc(cnfg->Nxl*cnfg->Nyl*sizeof(double));

		Nxl = cnfg->Nxl;
		Nyl = cnfg->Nyl;

		pos_x = mpi->getPosX();
		pos_y = mpi->getPosY();

	}

	positions(const positions &in){

		this->tbl_xhatx = (double*)malloc(in.Nxl*in.Nyl*sizeof(double));
		this->tbl_xhaty = (double*)malloc(in.Nxl*in.Nyl*sizeof(double));
		this->tbl_xbar2 = (double*)malloc(in.Nxl*in.Nyl*sizeof(double));

		for(int i = 0; i < in.Nxl*in.Nyl; i++){
			this->tbl_xhatx[i] = in.tbl_xhatx[i];
			this->tbl_xhaty[i] = in.tbl_xhaty[i];
			this->tbl_xbar2[i] = in.tbl_xbar2[i];
		}

		this->Nxl = in.Nxl;
		this->Nyl = in.Nyl;

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

	for(i = 0; i < Nxl; i++){
		for(j = 0; j < Nyl; j++){

			ig = (i+pos_x*Nxl);
			jg = (j+pos_y*Nyl);


                        //double dx2 = Nx*sin(M_PI*(x_global-xx)/Nx)/M_PI;
                        //double dy2 = Ny*sin(M_PI*(y_global-yy)/Ny)/M_PI;
                        //double dx = 0.5*Nx*sin(2.0*M_PI*(x_global-xx)/Nx)/M_PI;
                        //double dy = 0.5*Ny*sin(2.0*M_PI*(y_global-yy)/Ny)/M_PI;

		        //double rrr = 1.0*(dx2*dx2+dy2*dy2);
                      
                        //int ii = fabs(x_global - xx)*Nyg + fabs(y_global - yy);

                        //double dx = pos->xhatX(ii);
                        //double dy = pos->xhatY(ii);

                        //double rrr = pos->xbar2(ii);


			sargx = 0.5*Nx * sin( 2.0*M_PI * ig / (1.0 * Nx) ) / M_PI;
			sargy = 0.5*Ny * sin( 2.0*M_PI * jg / (1.0 * Ny) ) / M_PI;

			//tbl_xhat2[i*Nyl+j] = 4.0*pow(sargx,2.0) + 4.0*pow(sargy,2.0);
			tbl_xhatx[i*Nyl+j] = sargx;
			tbl_xhaty[i*Nyl+j] = sargy;

			sargx = Nx * sin( M_PI * ig / (1.0 * Nx) ) / M_PI;
			sargy = Ny * sin( M_PI * jg / (1.0 * Ny) ) / M_PI;

			tbl_xbar2[i*Nyl+j] = pow(sargx,2.0) + pow(sargy,2.0);
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
