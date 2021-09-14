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
 * File: MV_class.h
 * Authors: P. Korcyl
 * Contact: piotr.korcyl@uj.edu.pl
 * 
 * Version: 1.0
 * 
 * Description:
 * Class aggregating gaussian model parameters
 * 
 */

#ifndef H_GAUSSIAN_CLASS
#define H_GAUSSIAN_CLASS

#include <iostream>
#include <stdlib.h>
#include <complex>
#include <complex.h>
#include "config.h"

#include <omp.h>

#include <mpi.h>

#include "mpi_class.h"

#include <random>


class gaussian_class{

	private:

	double R_parameter;
	double C_parameter;
	int Ny_parameter;

	public:

	gaussian_class(double R, double C, int N){

		R_parameter = R;
		C_parameter = C;
		Ny_parameter = N;
	}

	double RGet(){
		return R_parameter;
	}

	double CGet(){
		return C_parameter;
	}

	int NyGet(){
		return Ny_parameter;
	}
};

#endif
