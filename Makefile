##FFTW=/home/pk/jimwlk/source/fftw/install
FFTW=/net/people/plgpkorcyl/fftw/install_gcc3/
all: main_explicit.cpp main_optimized.cpp mpi_init.cpp mpi_exchange_grid.cpp mpi_exchange_groups.cpp su3_matrix.h
	mpicxx -g -std=gnu++11 zheevh3-C-1.1/zheevh3.cpp zheevh3-C-1.1/zheevc3.cpp zheevh3-C-1.1/zheevq3.cpp zheevh3-C-1.1/zhetrd3.cpp mpi_init.cpp mpi_exchange_grid.cpp main_explicit.cpp -o main_explicit $(FFTW)/lib/libfftw3_mpi.a $(FFTW)/lib/libfftw3.a $(FFTW)/lib/libfftw3_omp.a -I$(FFTW)/include -Izheevh3-C-1.1/  -fopenmp
	mpicxx -g -std=gnu++11 zheevh3-C-1.1/zheevh3.cpp zheevh3-C-1.1/zheevc3.cpp zheevh3-C-1.1/zheevq3.cpp zheevh3-C-1.1/zhetrd3.cpp mpi_init.cpp mpi_exchange_grid.cpp main_optimized.cpp -o main_optimized $(FFTW)/lib/libfftw3_mpi.a $(FFTW)/lib/libfftw3.a $(FFTW)/lib/libfftw3_omp.a -I$(FFTW)/include -Izheevh3-C-1.1/  -fopenmp
