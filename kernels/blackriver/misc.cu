#include <cuda.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <cstddef>
#include <cstdio>

namespace br { namespace cuda {

// just declare 
template <typename T>
int causal_mask(const int *mask, T *out,
                  const int batch,
                  const int tokens,
                  cudaStream_t stream);

template <typename T>
int alibi(T *out,
          const int heads,
          const int tokens,
          cudaStream_t stream);

__global__ void mask_float( const int *mask, float *out, const float minv, 
                        const int batch,
                        const int tokens) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if ( e >= batch * tokens ) {
        return;
    }

    int i = e / tokens;
    int m = e % tokens;
    const int* ms = &mask[i*tokens];
    float* m2d = &out[i * tokens * tokens];
    
    for (int n = 0; n < tokens; n++) {
        m2d[m * tokens + n] = minv;
    }
    
    for (int n = 0; n <=m; n++) {
        if ( ms[n] != 0) {
            m2d[m * tokens + n] = 0.0;
        } else {
            break;
        }
    }
}

// implement for float and __half
template <>
int causal_mask<float>(const int *mask, float *out,
                        const int batch,
                        const int tokens,
                        cudaStream_t stream) {

    int len = batch * tokens;

    dim3 block_size(256);
	dim3 num_of_blocks((len + block_size.x - 1) / block_size.x);

    float minv = -1.0 * std::numeric_limits<float>::max();

    mask_float <<< num_of_blocks, block_size, 0, stream >>> (mask, out, minv, batch, tokens);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(-1);
    }
   
    return 0;
}

__global__ void alibi_float(float *out,
                            const float base,
                            const int heads,
                            const int tokens) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if ( e >= heads * tokens ) {
        return;
    }

    int j = e / tokens;
    int k = e % tokens;
    float slope = pow(base, (j+1) * 1.0);
    out[e] = k * 1.0 * slope;
}


// implement for float and __half
template <>
int alibi<float>(float *out,
                 const int heads,
                 const int tokens,
                 cudaStream_t stream) {

    int len = heads * tokens;

    dim3 block_size(256);
	dim3 num_of_blocks((len + block_size.x - 1) / block_size.x);
    
    double base = 3 - log2(heads*1.0);
    base = -1 * pow(2.0, base);
    base = pow(2.0, base);

    alibi_float <<< num_of_blocks, block_size, 0, stream >>> (out, base, heads, tokens);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(-1);
    }
    return 0;
}


}}
