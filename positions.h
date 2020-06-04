#ifndef H_POSITIONS
#define H_POSITIONS

#include <stdlib.h>
#include <iostream>

#include "config.h"

#include <mpi.h>

class positions {

private:

	double* tbl_xhat2;
	double* tbl_xhatx;
	double* tbl_xhaty;
	double* tbl_xbar2;
	double* tbl_xbarx;
	double* tbl_xbary;

	int Nxl, Nyl;
	int pos_x, pos_y;

public:

	positions(config* cnfg, mpi_class* mpi){

		tbl_xhat2 = (double*)malloc(cnfg->Nxl*cnfg->Nyl*sizeof(double));
		tbl_xhatx = (double*)malloc(cnfg->Nxl*cnfg->Nyl*sizeof(double));
		tbl_xhaty = (double*)malloc(cnfg->Nxl*cnfg->Nyl*sizeof(double));
		tbl_xbar2 = (double*)malloc(cnfg->Nxl*cnfg->Nyl*sizeof(double));
		tbl_xbarx = (double*)malloc(cnfg->Nxl*cnfg->Nyl*sizeof(double));
		tbl_xbary = (double*)malloc(cnfg->Nxl*cnfg->Nyl*sizeof(double));

		Nxl = cnfg->Nxl;
		Nyl = cnfg->Nyl;

		pos_x = mpi->getPosX();
		pos_y = mpi->getPosY();
	}

	~positions(){

		free(tbl_xhat2);
		free(tbl_xhatx);
		free(tbl_xhaty);
		free(tbl_xbar2);
		free(tbl_xbarx);
		free(tbl_xbary);

	}

	int set();

	double xhat2(int i);
	double xbarX(int i);
	double xbarY(int i);
	double xbar2(int i);
	double xhatX(int i);
	double xhatY(int i);


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
