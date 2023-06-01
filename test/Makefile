.PHONY: all

FLAGS = -Wall -O3 -Wno-maybe-uninitialized -fopenmp 
INC = -I../install/include/kernels \
	  -I../install/include/engine \
	  -I/home/teaonly/opt/nccl/include \
	  -I/home/teaonly/opt/blas/include \
	  -I/usr/local/cuda/include \
	  -I/usr/lib/x86_64-linux-gnu/openmpi/include \
	  -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi 

LINK = -L/home/teaonly/opt/nccl/lib \
	   -L/home/teaonly/opt/blas/lib \
	   -L/usr/local/cuda/lib64 \
	   -L/usr/lib/x86_64-linux-gnu/openmpi/lib \
	   -L../install/lib \
	   -lengine -lcuda_kernels -ltokenizers_c -lnccl -lopenblas -lcudnn -lcudart -lcublas -lcublasLt -lmpi -lmpi_cxx -lssl -lcrypto

repl: repl.cpp
	g++ $(FLAGS) -o $@ $< $(INC) $(LINK)

tokenizer: tokenizer.cpp
	g++ $(FLAGS) -o $@ $< $(INC) $(LINK)

train: train.cpp
	g++ $(FLAGS) -o $@ $< $(INC) $(LINK)

bloomz: bloomz.cpp
	g++ $(FLAGS) -o $@ $< $(INC) $(LINK)

run: bloomz
	mpirun -n 3 -stdin 0 bloomz

clean:
	rm -f repl tokenizer train bloomz
