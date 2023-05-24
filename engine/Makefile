.PHONY: all

TARGET = libblackriver.a 
all: $(TARGET) 

FLAGS = -Wall -O3 -Wno-maybe-uninitialized -fopenmp 
INC = -I. -I../kernels \
	  -I/home/teaonly/opt/nccl/include \
	  -I/home/teaonly/opt/blas/include \
	  -I/usr/local/cuda/include \
	  -I/usr/lib/x86_64-linux-gnu/openmpi/include \
	  -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi 

LINK = -L/home/teaonly/opt/nccl/lib -lnccl \
	   -L/home/teaonly/opt/blas/lib -lopenblas \
	   -L/usr/local/cuda/lib64 -lcudnn -lcudart -lcublas -lcublasLt \
	   -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi -lmpi_cxx

ENGINE_LIBS = ../kernels/build/libcuda_kernels.a

OBJS_DIR = ./objs
BR_SRCS = context.cpp tensortype.cpp cuda_tensor.cpp cpu_tensor.cpp dag.cpp nn.cpp 
BR_OBJS = ${BR_SRCS:%.cpp=$(OBJS_DIR)/%.o}

$(OBJS_DIR)/%.o : %.cpp
	@mkdir -p $(@D)
	g++ $(FLAGS) $(INC) -c -o $@ $< 

$(TARGET): $(BR_OBJS) 
	ar rcs $@ $(BR_OBJS) $(ENGINE_LIBS) 

clean:
	rm -rf $(OBJS_DIR)
	rm -f $(TARGET)

run: $(TARGET)
	mpirun -n 3 -stdin 1 train_bloom

