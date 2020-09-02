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
 * File: rand_class.h
 * Authors: P. Korcyl
 * Contact: piotr.korcyl@uj.edu.pl
 * 
 * Version: 1.0
 * 
 * Description:
 * Wrapper for the random generator
 * 
 */


#ifndef H_RAND_CLASS
#define H_RAND_CLASS

#include <iostream>
#include <stdlib.h>
#include <complex>
#include <complex.h>
#include "config.h"

#include <omp.h>

#include <mpi.h>

#include "mpi_class.h"

#include <random>


class rand_class{

	public:

//	std::ranlux48 rgenerator;
	std::ranlux24_base rgenerator;
	std::uniform_real_distribution<double> distribution{0.0,1.0};

	rand_class(mpi_class *mpi, config *cnfg){

	rgenerator.seed(cnfg->seed + 64*mpi->getRank() + omp_get_thread_num());

	}

	

double get(){
	
		return distribution(rgenerator);
	
	}
};

#endif
