CXX=mpicxx
INC=-I../zheevh3-C-1.1/ -I/home/pk/Desktop/nspt/fftw/install/include 
CXXFLAGS=-O3 -std=gnu++11 -fopenmp $(INC)
LIBS=/home/pk/Desktop/nspt/fftw/install/lib/libfftw3_mpi.a /home/pk/Desktop/nspt/fftw/install/lib/libfftw3.a /home/pk/Desktop/nspt/fftw/install/lib/libfftw3_omp.a 
LDFLAGS=$(LIBS) -lm 

default: 

all: test_position_evolution test_momentum_evolution

test_momentum_evolution: ../zheevh3-C-1.1/zheevh3.cpp ../zheevh3-C-1.1/zheevc3.cpp ../zheevh3-C-1.1/zheevq3.cpp ../zheevh3-C-1.1/zhetrd3.cpp ../mpi_init.cpp ../mpi_exchange_grid.cpp test_momentum_evolution.cpp
	$(CXX) $^ $(CXXFLAGS) $(LDFLAGS) -o $@

test_position_evolution: ../zheevh3-C-1.1/zheevh3.cpp ../zheevh3-C-1.1/zheevc3.cpp ../zheevh3-C-1.1/zheevq3.cpp ../zheevh3-C-1.1/zhetrd3.cpp ../mpi_init.cpp ../mpi_exchange_grid.cpp test_position_evolution.cpp
	$(CXX) $^ $(CXXFLAGS) $(LDFLAGS) -o $@


