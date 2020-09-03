CXX=mpicxx
FFTW=/home/pk/jimwlk/source/fftw/install
INC=-I../zheevh3-C-1.1/ -I$(FFTW)/include 
#INC=-I../zheevh3-C-1.1/ -I/net/people/plgpkorcyl/fftw/install_gcc3/include
CXXFLAGS=-g -O3 -std=gnu++11 -fopenmp $(INC)
LIBS=$(FFTW)/lib/libfftw3_mpi.a $(FFTW)/lib/libfftw3.a $(FFTW)/lib/libfftw3_omp.a 
#LIBS=/net/people/plgpkorcyl/fftw/install_gcc3/lib/libfftw3_mpi.a /net/people/plgpkorcyl/fftw/install_gcc3/lib/libfftw3.a /net/people/plgpkorcyl/fftw/install_gcc3/lib/libfftw3_omp.a
LDFLAGS=$(LIBS) -lm 

.PHONY: clean test_position_evolution test_momentum_evolution test_position_evolution_coupling test_momentum_evolution_coupling_lambda test_initial_condition test_momentum_evolution_coupling_lambda_noise test_position_evolution_coupling_noise

##default: test_jimwlk_kernel
##default: test_momentum_evolution_coupling_lambda_noise 
#default: test_momentum_evolution
#default: test_momentum_evolution_coupling
##default: test_noise_vectors test_noise_vectors_fft test_momentum_coupling_fft test_fftw_execute_random test_fftw_execute_random_std test_noise_vectors_fft_A_local
##default: test_position_evolution_coupling_noise
##default: test_position_evolution_coupling
#default: test_position_evolution_hatta_coupling
#default: test_position_evolution
##default: test_noise_vectors_and_cholesky_decomposition

all: test_noise_vectors test_hermitian test_multiplication test_exponentiate test_cholesky test_accumulation test_symmetrize_average test_symmetrize_average2 test_generator test_position_evolution test_momentum_evolution test_position_evolution_coupling test_initial_condition test_momentum_evolution_coupling_lambda test_momentum_evolution_coupling_lambda_noise test_momentum_evolution_coupling test_noise_vectors test_noise_vectors_fft test_noise_vectors_fft_A_local test_noise_vectors_and_cholesky_decomposition


-rw-r--r-- 1 pk pk 7716 Sep  3 12:04 test_position_evolution_hatta_coupling.cpp
-rw-r--r-- 1 pk pk 8698 Sep  3 12:04 test_position_evolution_coupling_noise.cpp
-rw-r--r-- 1 pk pk 3016 Sep  3 12:04 test_noise_vectors_fft.cpp
-rw-r--r-- 1 pk pk 3394 Sep  3 12:04 test_noise_vectors_fft_A_local.cpp
-rw-r--r-- 1 pk pk 5559 Sep  3 12:04 test_noise_vectors_and_cholesky_decomposition.cpp
-rw-r--r-- 1 pk pk 3731 Sep  3 12:04 test_MVModel.cpp
-rw-r--r-- 1 pk pk 8563 Sep  3 12:04 test_momentum_evolution_coupling.cpp
-rw-r--r-- 1 pk pk 8247 Sep  3 12:04 test_momentum_evolution_coupling2.cpp
-rw-r--r-- 1 pk pk 2320 Sep  3 12:04 test_momentum_coupling_fft.cpp
-rw-r--r-- 1 pk pk 1938 Sep  3 12:04 test_momentum_coupling.cpp
-rw-r--r-- 1 pk pk 4133 Sep  3 12:04 test_initial_condition_fftw1d.cpp
-rw-r--r-- 1 pk pk 2024 Sep  3 12:04 test_fftw_execute_random_std.cpp
-rw-r--r-- 1 pk pk 1631 Sep  3 12:04 test_fftw_execute_random.cpp
-rw-r--r-- 1 pk pk 1386 Sep  3 12:04 test_fftw_execute.cpp
-rw-r--r-- 1 pk pk  606 Sep  3 12:04 test_fftw.cpp

test_symmetrize_average: ../zheevh3-C-1.1/zheevh3.cpp ../zheevh3-C-1.1/zheevc3.cpp ../zheevh3-C-1.1/zheevq3.cpp ../zheevh3-C-1.1/zhetrd3.cpp ../mpi_init.cpp ../mpi_exchange_grid.cpp test_symmetrize_average.cpp
	$(CXX) $^ $(CXXFLAGS) $(LDFLAGS) -o $@

test_multiplication: ../zheevh3-C-1.1/zheevh3.cpp ../zheevh3-C-1.1/zheevc3.cpp ../zheevh3-C-1.1/zheevq3.cpp ../zheevh3-C-1.1/zhetrd3.cpp ../mpi_init.cpp ../mpi_exchange_grid.cpp test_multiplication.cpp
	$(CXX) $^ $(CXXFLAGS) $(LDFLAGS) -o $@

test_hermitian: ../zheevh3-C-1.1/zheevh3.cpp ../zheevh3-C-1.1/zheevc3.cpp ../zheevh3-C-1.1/zheevq3.cpp ../zheevh3-C-1.1/zhetrd3.cpp ../mpi_init.cpp ../mpi_exchange_grid.cpp test_hermitian.cpp
	$(CXX) $^ $(CXXFLAGS) $(LDFLAGS) -o $@

test_symmetrize_average2: ../zheevh3-C-1.1/zheevh3.cpp ../zheevh3-C-1.1/zheevc3.cpp ../zheevh3-C-1.1/zheevq3.cpp ../zheevh3-C-1.1/zhetrd3.cpp ../mpi_init.cpp ../mpi_exchange_grid.cpp test_symmetrize_average2.cpp
	$(CXX) $^ $(CXXFLAGS) $(LDFLAGS) -o $@

test_accumulation: ../zheevh3-C-1.1/zheevh3.cpp ../zheevh3-C-1.1/zheevc3.cpp ../zheevh3-C-1.1/zheevq3.cpp ../zheevh3-C-1.1/zhetrd3.cpp ../mpi_init.cpp ../mpi_exchange_grid.cpp test_accumulation.cpp
	$(CXX) $^ $(CXXFLAGS) $(LDFLAGS) -o $@

test_cholesky: ../zheevh3-C-1.1/zheevh3.cpp ../zheevh3-C-1.1/zheevc3.cpp ../zheevh3-C-1.1/zheevq3.cpp ../zheevh3-C-1.1/zhetrd3.cpp ../mpi_init.cpp ../mpi_exchange_grid.cpp test_cholesky.cpp
	$(CXX) $^ $(CXXFLAGS) $(LDFLAGS) -o $@

test_exponentiate: ../zheevh3-C-1.1/zheevh3.cpp ../zheevh3-C-1.1/zheevc3.cpp ../zheevh3-C-1.1/zheevq3.cpp ../zheevh3-C-1.1/zhetrd3.cpp ../mpi_init.cpp ../mpi_exchange_grid.cpp test_exponentiate.cpp
	$(CXX) $^ $(CXXFLAGS) $(LDFLAGS) -o $@

test_generator: ../zheevh3-C-1.1/zheevh3.cpp ../zheevh3-C-1.1/zheevc3.cpp ../zheevh3-C-1.1/zheevq3.cpp ../zheevh3-C-1.1/zhetrd3.cpp ../mpi_init.cpp ../mpi_exchange_grid.cpp test_generator.cpp
	$(CXX) $^ $(CXXFLAGS) $(LDFLAGS) -o $@


test_momentum_coupling: ../zheevh3-C-1.1/zheevh3.cpp ../zheevh3-C-1.1/zheevc3.cpp ../zheevh3-C-1.1/zheevq3.cpp ../zheevh3-C-1.1/zhetrd3.cpp ../mpi_init.cpp ../mpi_exchange_grid.cpp test_momentum_coupling.cpp
	$(CXX) $^ $(CXXFLAGS) $(LDFLAGS) -o $@


test_noise_vectors_and_cholesky_decomposition: ../zheevh3-C-1.1/zheevh3.cpp ../zheevh3-C-1.1/zheevc3.cpp ../zheevh3-C-1.1/zheevq3.cpp ../zheevh3-C-1.1/zhetrd3.cpp ../mpi_init.cpp ../mpi_exchange_grid.cpp test_noise_vectors_and_cholesky_decomposition.cpp
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

test_momentum_evolution_coupling_linkernel: ../zheevh3-C-1.1/zheevh3.cpp ../zheevh3-C-1.1/zheevc3.cpp ../zheevh3-C-1.1/zheevq3.cpp ../zheevh3-C-1.1/zhetrd3.cpp ../mpi_init.cpp ../mpi_exchange_grid.cpp test_momentum_evolution_coupling_linkernel.cpp
	$(CXX) $^ $(CXXFLAGS) $(LDFLAGS) -o $@


test_momentum_evolution_coupling2: ../zheevh3-C-1.1/zheevh3.cpp ../zheevh3-C-1.1/zheevc3.cpp ../zheevh3-C-1.1/zheevq3.cpp ../zheevh3-C-1.1/zhetrd3.cpp ../mpi_init.cpp ../mpi_exchange_grid.cpp test_momentum_evolution_coupling2.cpp
	$(CXX) $^ $(CXXFLAGS) $(LDFLAGS) -o $@

test_position_evolution: ../zheevh3-C-1.1/zheevh3.cpp ../zheevh3-C-1.1/zheevc3.cpp ../zheevh3-C-1.1/zheevq3.cpp ../zheevh3-C-1.1/zhetrd3.cpp ../mpi_init.cpp ../mpi_exchange_grid.cpp test_position_evolution.cpp
	$(CXX) $^ $(CXXFLAGS) $(LDFLAGS) -o $@

test_position_evolution_coupling: ../zheevh3-C-1.1/zheevh3.cpp ../zheevh3-C-1.1/zheevc3.cpp ../zheevh3-C-1.1/zheevq3.cpp ../zheevh3-C-1.1/zhetrd3.cpp ../mpi_init.cpp ../mpi_exchange_grid.cpp test_position_evolution_coupling.cpp
	$(CXX) $^ $(CXXFLAGS) $(LDFLAGS) -o $@

test_position_evolution_hatta_coupling: ../zheevh3-C-1.1/zheevh3.cpp ../zheevh3-C-1.1/zheevc3.cpp ../zheevh3-C-1.1/zheevq3.cpp ../zheevh3-C-1.1/zhetrd3.cpp ../mpi_init.cpp ../mpi_exchange_grid.cpp test_position_evolution_hatta_coupling.cpp
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
