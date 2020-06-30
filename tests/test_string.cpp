#include <iostream>
#include <string>
#include <stdio.h>
#include <stdlib.h>


int function(std::string const &s){

	std::cout << s << std::endl;

	char nazwa[100];

	sprintf(nazwa, "%s__plik.out", s.c_str());

	std::cout << nazwa << std::endl;

return 1;
}


int main(void){

	std::string s = "czesc";

	function(s);

return 1;
}
