CXX=mpicxx
INC=-I../zheevh3-C-1.1/ -I/home/pk/Desktop/nspt/fftw/install/include 
#INC=-I../zheevh3-C-1.1/ -I/net/people/plgpkorcyl/fftw/install_gcc3/include
CXXFLAGS=-g -O3 -std=gnu++11 -fopenmp $(INC)
LIBS=/home/pk/Desktop/nspt/fftw/install/lib/libfftw3_mpi.a /home/pk/Desktop/nspt/fftw/install/lib/libfftw3.a /home/pk/Desktop/nspt/fftw/install/lib/libfftw3_omp.a 
#LIBS=/net/people/plgpkorcyl/fftw/install_gcc3/lib/libfftw3_mpi.a /net/people/plgpkorcyl/fftw/install_gcc3/lib/libfftw3.a /net/people/plgpkorcyl/fftw/install_gcc3/lib/libfftw3_omp.a
LDFLAGS=$(LIBS) -lm 

.PHONY: clean test_position_evolution test_momentum_evolution test_position_evolution_coupling test_momentum_evolution_coupling_lambda test_initial_condition test_momentum_evolution_coupling_lambda_noise test_position_evolution_coupling_noise

##default: test_momentum_evolution_coupling_lambda_noise 
##default: test_momentum_evolution_coupling
##default: test_momentum_evolution_coupling2
##default: test_noise_vectors test_noise_vectors_fft test_momentum_coupling_fft test_fftw_execute_random test_fftw_execute_random_std test_noise_vectors_fft_A_local
default: test_position_evolution_coupling_noise

all: test_position_evolution test_momentum_evolution test_position_evolution_coupling test_initial_condition test_momentum_evolution_coupling_lambda test_momentum_evolution_coupling_lambda_noise

test_momentum_coupling: ../zheevh3-C-1.1/zheevh3.cpp ../zheevh3-C-1.1/zheevc3.cpp ../zheevh3-C-1.1/zheevq3.cpp ../zheevh3-C-1.1/zhetrd3.cpp ../mpi_init.cpp ../mpi_exchange_grid.cpp test_momentum_coupling.cpp
	$(CXX) $^ $(CXXFLAGS) $(LDFLAGS) -o $@

test_fftw_execute: ../zheevh3-C-1.1/zheevh3.cpp ../zheevh3-C-1.1/zheevc3.cpp ../zheevh3-C-1.1/zheevq3.cpp ../zheevh3-C-1.1/zhetrd3.cpp ../mpi_init.cpp ../mpi_exchange_grid.cpp test_fftw_execute.cpp
	$(CXX) $^ $(CXXFLAGS) $(LDFLAGS) -o $@

test_fftw_execute_random_std: ../zheevh3-C-1.1/zheevh3.cpp ../zheevh3-C-1.1/zheevc3.cpp ../zheevh3-C-1.1/zheevq3.cpp ../zheevh3-C-1.1/zhetrd3.cpp ../mpi_init.cpp ../mpi_exchange_grid.cpp test_fftw_execute_random_std.cpp
	$(CXX) $^ $(CXXFLAGS) $(LDFLAGS) -o $@


test_fftw_execute_random: ../zheevh3-C-1.1/zheevh3.cpp ../zheevh3-C-1.1/zheevc3.cpp ../zheevh3-C-1.1/zheevq3.cpp ../zheevh3-C-1.1/zhetrd3.cpp ../mpi_init.cpp ../mpi_exchange_grid.cpp test_fftw_execute_random.cpp
	$(CXX) $^ $(CXXFLAGS) $(LDFLAGS) -o $@


test_momentum_coupling_fft: ../zheevh3-C-1.1/zheevh3.cpp ../zheevh3-C-1.1/zheevc3.cpp ../zheevh3-C-1.1/zheevq3.cpp ../zheevh3-C-1.1/zhetrd3.cpp ../mpi_init.cpp ../mpi_exchange_grid.cpp test_momentum_coupling_fft.cpp
	$(CXX) $^ $(CXXFLAGS) $(LDFLAGS) -o $@


test_noise_vectors: ../zheevh3-C-1.1/zheevh3.cpp ../zheevh3-C-1.1/zheevc3.cpp ../zheevh3-C-1.1/zheevq3.cpp ../zheevh3-C-1.1/zhetrd3.cpp ../mpi_init.cpp ../mpi_exchange_grid.cpp test_noise_vectors.cpp
	$(CXX) $^ $(CXXFLAGS) $(LDFLAGS) -o $@

test_noise_vectors_fft: ../zheevh3-C-1.1/zheevh3.cpp ../zheevh3-C-1.1/zheevc3.cpp ../zheevh3-C-1.1/zheevq3.cpp ../zheevh3-C-1.1/zhetrd3.cpp ../mpi_init.cpp ../mpi_exchange_grid.cpp test_noise_vectors_fft.cpp
	$(CXX) $^ $(CXXFLAGS) $(LDFLAGS) -o $@

test_noise_vectors_fft_A_local: ../zheevh3-C-1.1/zheevh3.cpp ../zheevh3-C-1.1/zheevc3.cpp ../zheevh3-C-1.1/zheevq3.cpp ../zheevh3-C-1.1/zhetrd3.cpp ../mpi_init.cpp ../mpi_exchange_grid.cpp test_noise_vectors_fft_A_local.cpp
	$(CXX) $^ $(CXXFLAGS) $(LDFLAGS) -o $@



test_momentum_evolution: ../zheevh3-C-1.1/zheevh3.cpp ../zheevh3-C-1.1/zheevc3.cpp ../zheevh3-C-1.1/zheevq3.cpp ../zheevh3-C-1.1/zhetrd3.cpp ../mpi_init.cpp ../mpi_exchange_grid.cpp test_momentum_evolution.cpp
	$(CXX) $^ $(CXXFLAGS) $(LDFLAGS) -o $@

test_momentum_evolution_coupling: ../zheevh3-C-1.1/zheevh3.cpp ../zheevh3-C-1.1/zheevc3.cpp ../zheevh3-C-1.1/zheevq3.cpp ../zheevh3-C-1.1/zhetrd3.cpp ../mpi_init.cpp ../mpi_exchange_grid.cpp test_momentum_evolution_coupling.cpp
	$(CXX) $^ $(CXXFLAGS) $(LDFLAGS) -o $@

test_momentum_evolution_coupling2: ../zheevh3-C-1.1/zheevh3.cpp ../zheevh3-C-1.1/zheevc3.cpp ../zheevh3-C-1.1/zheevq3.cpp ../zheevh3-C-1.1/zhetrd3.cpp ../mpi_init.cpp ../mpi_exchange_grid.cpp test_momentum_evolution_coupling2.cpp
	$(CXX) $^ $(CXXFLAGS) $(LDFLAGS) -o $@

test_position_evolution: ../zheevh3-C-1.1/zheevh3.cpp ../zheevh3-C-1.1/zheevc3.cpp ../zheevh3-C-1.1/zheevq3.cpp ../zheevh3-C-1.1/zhetrd3.cpp ../mpi_init.cpp ../mpi_exchange_grid.cpp test_position_evolution.cpp
	$(CXX) $^ $(CXXFLAGS) $(LDFLAGS) -o $@

test_position_evolution_coupling: ../zheevh3-C-1.1/zheevh3.cpp ../zheevh3-C-1.1/zheevc3.cpp ../zheevh3-C-1.1/zheevq3.cpp ../zheevh3-C-1.1/zhetrd3.cpp ../mpi_init.cpp ../mpi_exchange_grid.cpp test_position_evolution_coupling.cpp
	$(CXX) $^ $(CXXFLAGS) $(LDFLAGS) -o $@

test_position_evolution_coupling_noise: ../zheevh3-C-1.1/zheevh3.cpp ../zheevh3-C-1.1/zheevc3.cpp ../zheevh3-C-1.1/zheevq3.cpp ../zheevh3-C-1.1/zhetrd3.cpp ../mpi_init.cpp ../mpi_exchange_grid.cpp test_position_evolution_coupling_noise.cpp
	$(CXX) $^ $(CXXFLAGS) $(LDFLAGS) -o $@

test_initial_condition: ../zheevh3-C-1.1/zheevh3.cpp ../zheevh3-C-1.1/zheevc3.cpp ../zheevh3-C-1.1/zheevq3.cpp ../zheevh3-C-1.1/zhetrd3.cpp ../mpi_init.cpp ../mpi_exchange_grid.cpp test_initial_condition.cpp
	$(CXX) $^ $(CXXFLAGS) $(LDFLAGS) -o $@

test_momentum_evolution_coupling_lambda: ../zheevh3-C-1.1/zheevh3.cpp ../zheevh3-C-1.1/zheevc3.cpp ../zheevh3-C-1.1/zheevq3.cpp ../zheevh3-C-1.1/zhetrd3.cpp ../mpi_init.cpp ../mpi_exchange_grid.cpp test_momentum_evolution_coupling_lambda.cpp
	$(CXX) $^ $(CXXFLAGS) $(LDFLAGS) -o $@

test_momentum_evolution_coupling_lambda_noise: ../zheevh3-C-1.1/zheevh3.cpp ../zheevh3-C-1.1/zheevc3.cpp ../zheevh3-C-1.1/zheevq3.cpp ../zheevh3-C-1.1/zhetrd3.cpp ../mpi_init.cpp ../mpi_exchange_grid.cpp test_momentum_evolution_coupling_lambda_noise.cpp
	$(CXX) $^ $(CXXFLAGS) $(LDFLAGS) -o $@

clean:
	rm test_initial_condition
	rm test_position_evolution 
	rm test_momentum_evolution 
	rm test_position_evolution_coupling
	rm test_momentum_evolution_coupling_lambda
	rm test_momentum_evolution_coupling_lambda_noise
