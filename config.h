#ifndef H_CONFIG
#define H_CONFIG

#define Nx 64
#define Ny 64

#include <iostream>

enum Evolution { POSITION_EVOLUTION, MOMENTUM_EVOLUTION };

enum Coupling { SQRT_COUPLING_CONSTANT, NOISE_COUPLING_CONSTANT, HATTA_COUPLING_CONSTANT };

enum Kernel { LINEAR_KERNEL, SIN_KERNEL };

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


	int read_config_from_file(std::string file_name_cnfg){

		FILE *f;
		float tmp;

		f = fopen(file_name_cnfg.c_str(), "r+");		
	
		fscanf(f, "stat = %i\n", &stat);
		fscanf(f, "mu = %f\n", &tmp);
		mu = tmp;
		fscanf(f, "mass = %f\n", &tmp);
		mass = tmp;
		fscanf(f, "elementaryWilsonLines = %i\n", &elementaryWilsonLines);
		fscanf(f, "file_name = %s\n", &file_name[0]);
		fscanf(f, "langevin_steps = %i\n", &langevin_steps);
		fscanf(f, "measurements = %i\n", &measurements);
		fscanf(f, "step = %f\n", &tmp);
		step = tmp;	

		std::string evolution;
		std::string coupling;
		std::string kernel;

		fscanf(f, "EvolutionChoice = %s\n", &evolution[0]);
		fscanf(f, "CouplingChoice = %s\n", &coupling[0]);
		fscanf(f, "KernelChoice = %s\n", &kernel[0]);

		if(evolution.compare("MOMENTUM_EVOLUTION") == 0)
			EvolutionChoice = MOMENTUM_EVOLUTION;
		if(evolution.compare("POSITION_EVOLUTION") == 0)
			EvolutionChoice = POSITION_EVOLUTION;

		if(coupling.compare("SQRT_COUPLING_CONSTANT") == 0)
			CouplingChoice = SQRT_COUPLING_CONSTANT;
		if(coupling.compare("NOISE_COUPLING_CONSTANT") == 0)
			CouplingChoice = NOISE_COUPLING_CONSTANT;
		if(coupling.compare("HATTA_COUPLING_CONSTANT") == 0)
			CouplingChoice = HATTA_COUPLING_CONSTANT;

		if(kernel.compare("LINEAR_KERNEL") == 0)
			KernelChoice = LINEAR_KERNEL;
		if(kernel.compare("SIN_KERNEL") == 0)
			KernelChoice = SIN_KERNEL;


		fclose(f);
	}
};

#endif
