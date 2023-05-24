cmake_minimum_required(VERSION 3.18)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70 80 )
endif()
 
set(CMAKE_CXX_STANDARD 17) 
set(CMAKE_CUDA_STANDARD 17) 
set(CMAKE_CUDA_STANDARD_REQUIRED ON) 

project(tensortype LANGUAGES CUDA CXX)

set(tt_files
    context.cpp
    tensortype.cpp
    cuda_tensor.cpp
    cpu_tensor.cpp
    dag.cpp
    nn.cpp
    tokenizer.cpp)

add_library(tensortype STATIC ${tt_files})

include_directories(../install/include/kernels/)
include_directories(/home/teaonly/opt/nccl/include)
include_directories(/home/teaonly/opt/blas/include)
include_directories(/usr/lib/x86_64-linux-gnu/openmpi/include)
include_directories(/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi)
include_directories(/usr/local/cuda/include)
	
target_link_libraries(tensortype cudnn cublas cudart cublasLt nccl openbias)

install(TARGETS tensortype DESTINATION lib)
install(PROGRAMS 
        common.hpp
        context.hpp
        tensortype.hpp
        cpu_tensor.hpp
        cuda_tensor.hpp
        dag.hpp
        nn.hpp
        tokenizer.hpp
        DESTINATION include/tensortype )
