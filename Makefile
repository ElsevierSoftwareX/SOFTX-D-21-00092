all: main_omp.cpp mpi_init.cpp mpi_exchange_grid.cpp mpi_exchange_groups.cpp su3_complex.h su3_matrix.h
	mpicxx zheevh3-C-1.1/zheevh3.cpp zheevh3-C-1.1/zheevc3.cpp zheevh3-C-1.1/zheevq3.cpp zheevh3-C-1.1/zhetrd3.cpp mpi_init.cpp mpi_exchange_grid.cpp mpi_exchange_groups.cpp main_omp.cpp -o main_omp /home/pk/Desktop/nspt/fftw/install/lib/libfftw3_mpi.a /home/pk/Desktop/nspt/fftw/install/lib/libfftw3.a /home/pk/Desktop/nspt/fftw/install/lib/libfftw3_omp.a -I/home/pk/Desktop/nspt/fftw/install/include -Izheevh3-C-1.1/  -fopenmp
#	mpicxx config.cpp mpi_init.cpp mpi_pos.cpp mpi_allocate.cpp mpi_exchange_grid.cpp mpi_exchange_boundaries.cpp utils.cpp mpi_split.cpp mpi_gather.cpp main.cpp -o main -lm -I/home/pk/Desktop/nspt/fftw/install/include -L/home/pk/Desktop/nspt/fftw/install/lib/ -lfftw3 -lfftw3_mpi


#rw------- 1 pk pk  5375 Apr 30 17:36 zheevh3.cpp
#rw------- 1 pk pk  3583 Apr 30 17:36 zheevc3.cpp
#rw------- 1 pk pk  1187 Apr 30 17:37 zheevc3.h
#rw------- 1 pk pk  1217 Apr 30 17:37 zheevq3.h
#rw------- 1 pk pk  4335 Apr 30 17:39 zheevq3.cpp
#rw------- 1 pk pk  1258 Apr 30 17:39 zhetrd3.h
#rw------- 1 pk pk  3690 Apr 30 17:42 zhetrd3.cpp




