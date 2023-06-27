#ifndef _CONTEXT_HPP_
#define _CONTEXT_HPP_

#include "common.hpp"

namespace br {

struct ComputingContext {
    static int cuda_device;
    static cudaStream_t cuda_stream;
    static cublasHandle_t cublas_handle;
    static cublasLtHandle_t cublasLt_handle;
    static cudnnHandle_t cudnn_handle;

    static void* cuda_workspace;
    static void* local_workspace;
    static size_t workspace_size;
    static std::mt19937* rng;

    static void boot(int cud);
    static void shutdown();
};

struct CollectiveContext {
    static int      current;
    static int      mpi_world;
    static int      mpi_rank;
    static int      pipe_world;
    static int      pipe_rank;
    static int*     pipe_fds;

    static ncclComm_t      nccl_comm;
    static ncclUniqueId    nccl_id;
    static int             nccl_rank;
    static int             nccl_world;

    static void boot_pipe(int gpus);
    static void boot_mpi(int argc, char* argv[], int gpus);
    static void shutdown();

    static int pipe_write(const int n, const void *buf, size_t nbyte);
    static int pipe_read(void* buf, size_t nbyte);

    static int now();
};

struct MemoryContext {
    static const size_t page_size;
    static void*    root;
    static size_t   total_size;
    static size_t   currentp;

    static void* alloc(size_t blk_size);
    static void boot(size_t total_bytes);
    static void shutdown();
};

} // end of namespace br


#endif
