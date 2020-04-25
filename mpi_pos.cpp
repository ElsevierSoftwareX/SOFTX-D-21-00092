#include "config.h"

int loc_pos(int t, int x, int y, int z){

	return z + Nzl*y + Nzl*Nyl*x + Nzl*Nyl*Nxl*t;
}

int buf_pos(int t, int x, int y, int z){

	return (z+(Nzl_buf-Nzl)/2 + Nzl_buf*(y+(Nyl_buf-Nyl)/2) + Nzl_buf*Nyl_buf*(x+(Nxl_buf-Nxl)/2) + Nzl_buf*Nyl_buf*Nxl_buf*t);
}

int buf_pos_ex(int t, int x, int y, int z){

	return (z+ Nzl_buf*y + Nzl_buf*Nyl_buf*x + Nzl_buf*Nyl_buf*Nxl_buf*t);
}
