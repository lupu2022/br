cmake_minimum_required(VERSION 3.18)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70 80 )
endif()
 
set(CMAKE_CXX_STANDARD 17) 
set(CMAKE_CUDA_STANDARD 17) 
set(CMAKE_CUDA_STANDARD_REQUIRED ON) 

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

project(engine LANGUAGES CUDA CXX)

set(tt_files
    context.cpp
    common.cpp
    tensortype.cpp
    cuda_tensor.cpp
    cpu_tensor.cpp
    dag.cpp
    nn.cpp
    tokenizer.cpp)

add_library(engine STATIC ${tt_files})

include_directories(../install/include/kernels/)
include_directories(../install/include/tokenizers/)
include_directories(/usr/local/include)
include_directories(/usr/lib/x86_64-linux-gnu/openmpi/include)
include_directories(/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi)
include_directories(/usr/local/cuda/include)
	
target_link_libraries(engine cudnn cublas cudart cublasLt nccl openbias sentencepiece)

install(TARGETS engine DESTINATION lib)
install(PROGRAMS 
        common.hpp
        context.hpp
        operators.hpp
        tensortype.hpp
        cpu_tensor.hpp
        cuda_tensor.hpp
        dag.hpp
        nn.hpp
        tokenizer.hpp
        br_engine.hpp
        DESTINATION include/engine )
