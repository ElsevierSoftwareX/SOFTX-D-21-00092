all:  config.cpp main.cpp mpi_init.cpp mpi_pos.cpp mpi_exchange_grid.cpp mpi_exchange_boundaries.cpp mpi_allocate.cpp utils.cpp mpi_split.cpp mpi_gather.cpp
	mpicxx config.cpp mpi_init.cpp mpi_pos.cpp mpi_allocate.cpp mpi_exchange_grid.cpp mpi_exchange_boundaries.cpp utils.cpp mpi_split.cpp mpi_gather.cpp main.cpp -o main -lm


