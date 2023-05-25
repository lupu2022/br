.PHONY: all

FLAGS = -Wall -O3 -Wno-maybe-uninitialized -fopenmp 
INC = -I../install/include/kernels \
	  -I../install/include/engine \
	  -I/home/teaonly/opt/nccl/include \
	  -I/home/teaonly/opt/blas/include \
	  -I/usr/local/cuda/include \
	  -I/usr/lib/x86_64-linux-gnu/openmpi/include \
	  -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi 

LINK = -L/home/teaonly/opt/nccl/lib -lnccl \
	   -L/home/teaonly/opt/blas/lib -lopenblas \
	   -L/usr/local/cuda/lib64 -lcudnn -lcudart -lcublas -lcublasLt \
	   -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi -lmpi_cxx \
	   -L../install/lib -lcuda_kernels -lengine

all:
	g++ $(FLAGS) -o test_tokenizer tokenizer.cpp $(INC) $(LINK)

clean:
	rm test_tokenizer
