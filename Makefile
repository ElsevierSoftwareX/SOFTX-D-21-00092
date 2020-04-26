all:  config.cpp main.cpp mpi_init.cpp mpi_pos.cpp mpi_exchange_grid.cpp mpi_exchange_boundaries.cpp mpi_allocate.cpp utils.cpp mpi_split.cpp mpi_gather.cpp su3_complex.h su3_matrix.h
	mpicxx config.cpp mpi_init.cpp mpi_pos.cpp mpi_allocate.cpp mpi_exchange_grid.cpp mpi_exchange_boundaries.cpp utils.cpp mpi_split.cpp mpi_gather.cpp main.cpp -o main -lm /home/pk/Desktop/nspt/fftw/install/lib/libfftw3_mpi.a /home/pk/Desktop/nspt/fftw/install/lib/libfftw3.a /home/pk/Desktop/nspt/fftw/install/lib/libfftw3_omp.a -I/home/pk/Desktop/nspt/fftw/install/include
#	mpicxx config.cpp mpi_init.cpp mpi_pos.cpp mpi_allocate.cpp mpi_exchange_grid.cpp mpi_exchange_boundaries.cpp utils.cpp mpi_split.cpp mpi_gather.cpp main.cpp -o main -lm -I/home/pk/Desktop/nspt/fftw/install/include -L/home/pk/Desktop/nspt/fftw/install/lib/ -lfftw3 -lfftw3_mpi




