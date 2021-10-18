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
 * File: config.h
 * Authors: P. Korcyl
 * Contact: piotr.korcyl@uj.edu.pl
 * 
 * Version: 1.0
 * 
 * Description:
 * Class containing parameters and setup
 * 
 */

#ifndef H_CONFIG
#define H_CONFIG

#define Nx 1024
#define Ny Nx

#include <iostream>
#include <stdlib.h>
#include <string.h>

enum Evolution { POSITION_EVOLUTION, MOMENTUM_EVOLUTION, NO_EVOLUTION };

enum Coupling { SQRT_COUPLING_CONSTANT, NOISE_COUPLING_CONSTANT, HATTA_COUPLING_CONSTANT, NO_COUPLING_CONSTANT };

enum Kernel { LINEAR_KERNEL, SIN_KERNEL };

enum InitialCondition { GAUSSIAN_CONDITION, MV_CONDITION };

enum Choice { YES, NO };

class config{

public:

	int Nxl = 0, Nyl = 0;
	int Nxl_buf = 0, Nyl_buf = 0;

	int ExchangeX = 0, ExchangeY = 0;

	int XNeighbourNext = 0, XNeighbourPrevious = 0;
	int YNeighbourNext = 0, YNeighbourPrevious = 0;

	int proc_x, proc_y;

	int seed;

	int stat = 4;
	double R = 16.0;
	double C = 4.0;
	double mu = 30.72;
	double mass = 0.0001;
	int elementaryWilsonLines = 50;
	std::string file_name = "output";
	int langevin_steps = 100;
	int measurements = 100;
	double step = 0.0004;

	Evolution EvolutionChoice = MOMENTUM_EVOLUTION;

	Coupling CouplingChoice = NOISE_COUPLING_CONSTANT;

	Kernel KernelChoice = SIN_KERNEL;
	
	InitialCondition InitialConditionChoice = GAUSSIAN_CONDITION;

        Choice ContinuationChoice = NO;

        Choice ContinuationOutputChoice = NO;

        Choice AdaptiveResolutionChoice = NO;

	int read_config_from_file(std::string file_name_cnfg){

		FILE *f;
		float tmp;
		
		char str_file_name[200];
		char str_condition[200];
                char str_continuation[200];
                char str_continuation_output[200];
                char str_adaptive_resolution[200];


		f = fopen(file_name_cnfg.c_str(), "r+");		
	
		fscanf(f, "stat = %i\n", &stat);
		printf("SETUP: stat = %i\n", stat); 
		fscanf(f, "initial condition = %s\n", &str_condition[0]);
		printf("SETUP: initial condition = %s\n", str_condition);
		if(strcmp(str_condition, "GAUSSIAN_CONDITION") == 0){
			InitialConditionChoice = GAUSSIAN_CONDITION;
			fscanf(f, "R = %f\n", &tmp);
			R = tmp;
			printf("SETUP: R = %f\n", R); 
			fscanf(f, "C = %f\n", &tmp);
			C = tmp;
			printf("SETUP: C = %f\n", C); 
		}
		if(strcmp(str_condition, "MV_CONDITION") == 0){
			InitialConditionChoice = MV_CONDITION;

			fscanf(f, "mu = %f\n", &tmp);
			mu = tmp;
			printf("SETUP: mu = %f\n", mu); 
			fscanf(f, "mass = %f\n", &tmp);
			mass = tmp;
			printf("SETUP: mass = %f\n", mass); 
		}
		fscanf(f, "elementaryWilsonLines = %i\n", &elementaryWilsonLines);
		printf("SETUP: elementaryWilsonLines = %i\n", elementaryWilsonLines); 
		fscanf(f, "file_name = %s\n", &str_file_name[0]);

		
                file_name.assign(str_file_name);


                fscanf(f, "continuation = %s\n", &str_continuation[0]);
                printf("SETUP: continuation = %s\n", str_continuation);
                if(strcmp(str_continuation, "YES") == 0){
                        ContinuationChoice = YES;
                }
                if(strcmp(str_continuation, "NO") == 0){
                        ContinuationChoice = NO;
                }
                fscanf(f, "startingS = %i\n", &startingS);
                printf("SETUP: startingS = %i\n", startingS);
                fscanf(f, "continuationOutput = %s\n", &str_continuation_output[0]);
                printf("SETUP: continuationOutput = %s\n", str_continuation_output);
                if(strcmp(str_continuation_output, "YES") == 0){
                        ContinuationOutputChoice = YES;
                }
                if(strcmp(str_continuation_output, "NO") == 0){
                        ContinuationOutputChoice = NO;
                }
                fscanf(f, "adaptiveResolution = %s\n", &str_adaptive_resolution[0]);
                printf("SETUP: adaptiveResolution = %s\n", str_adaptive_resolution);
                if(strcmp(str_adaptive_resolution, "YES") == 0){
                        AdaptiveResolutionChoice = YES;
                }
                if(strcmp(str_adaptive_resolution, "NO") == 0){
                        AdaptiveResolutionChoice = NO;
                }


		fscanf(f, "langevin_steps = %i\n", &langevin_steps);
		printf("SETUP: langevin_steps = %i\n", langevin_steps); 
		fscanf(f, "measurements = %i\n", &measurements);
		printf("SETUP: measurements = %i\n", measurements); 
		fscanf(f, "step = %f\n", &tmp);
		step = tmp;	
		printf("SETUP: step = %f\n", step); 

		char evolution[100];
		char coupling[100];
		char kernel[100];

		fscanf(f, "EvolutionChoice = %s\n", &evolution[0]);
		fscanf(f, "CouplingChoice = %s\n", &coupling[0]);
		fscanf(f, "KernelChoice = %s\n", &kernel[0]);

		if(strcmp(evolution, "MOMENTUM_EVOLUTION") == 0){
			EvolutionChoice = MOMENTUM_EVOLUTION;
			printf("SETUP: MOMENTUM_EOLUTION\n");
		}
		else if(strcmp(evolution, "POSITION_EVOLUTION") == 0){
			EvolutionChoice = POSITION_EVOLUTION;
			printf("SETUP: POSITION_EVOLUTION\n");
		}
		else if(strcmp(evolution, "NO_EVOLUTION") == 0){
			EvolutionChoice = NO_EVOLUTION;
			printf("SETUP: NO_EVOLUTION\n");
		}else{
			printf("SETUP: EVOLUTION METHOD/NO_EVOLUTION OPTION UNRECOGNIZED; ABORTING\n");
			exit(0);
		}


		if(strcmp(coupling, "SQRT_COUPLING_CONSTANT") == 0){
			CouplingChoice = SQRT_COUPLING_CONSTANT;
			printf("SETUP: SQRT_COUPLING_CONSTANT\n");
		}
		else if(strcmp(coupling, "NOISE_COUPLING_CONSTANT") == 0){
			CouplingChoice = NOISE_COUPLING_CONSTANT;
			printf("SETUP: NOISE_COUPLING_CONSTANT\n");
		}
		else if(strcmp(coupling, "HATTA_COUPLING_CONSTANT") == 0){
			CouplingChoice = HATTA_COUPLING_CONSTANT;
			printf("SETUP: HATTA_COUPLING_CONSTANT\n");
		}
		else if(strcmp(coupling, "NO_COUPLING_CONSTANT") == 0){
			CouplingChoice = NO_COUPLING_CONSTANT;
			printf("SETUP: NO_COUPLING_CONSTANT\n");
		}else{
			printf("SETUP: COUPLING_CONSTANT/NO_COUPLING_CONSTANT OPTION UNRECOGNIZED; ABORTING\n"); 
			exit(0);
		}


		if(strcmp(kernel, "LINEAR_KERNEL") == 0){
			KernelChoice = LINEAR_KERNEL;
			printf("SETUP: LINEAR_KENREL\n");
		}
		else if(strcmp(kernel, "SIN_KERNEL") == 0){
			KernelChoice = SIN_KERNEL;
			printf("SETUP: SIN_KERNEL\n");
		}else{
			printf("SETUP: KERNEL UNRECOGNIZED\n");
			exit(0);
		}


		fclose(f);

		return 1;
	}

};

#endif
