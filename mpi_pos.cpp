#include "config.h"

int loc_pos(int x, int y){

	return y + Nyl*x;
}

int buf_pos(int x, int y){

	return ((y+(Nyl_buf-Nyl)/2) + Nyl_buf*(x+(Nxl_buf-Nxl)/2));
}

int buf_pos_ex(int x, int y){

	return (y + Nyl_buf*x);
}
