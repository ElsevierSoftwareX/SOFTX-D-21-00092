#ifndef H_CONFIG
#define H_CONFIG

#define Nx 32
#define Ny 32

#include <iostream>

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
const std::string file_name = "output";
const int langevin_steps = 100;
const int measurements = 100;
double step = 0.0004;


const int position_evolution = 0;
const int momentum_evolution = 1;
	
const int sqrt_coupling_constant = 1;
const int noise_coupling_constant = 0;
const int hatta_coupling_constant = 1;


};

#endif
