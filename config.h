#ifndef H_CONFIG
#define H_CONFIG

#define Nx 64
#define Ny 64

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

int stat;

};

#endif
