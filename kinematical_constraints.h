#ifndef KINEMATICAL_CONSTRAINTS
#define KINEMATICAL_CONSTRAINTS

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double theta(double x, double y);

int max(int x, int y);

int kinematical_constraints(int initial_r_2, int max_evolution, double langevin_step, int *pos, int *start);

#endif

