#include <cuda.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <cstddef>
#include <cstdio>

namespace br { namespace cuda {

// just declare
template <typename T>
int silu_product(const T *in_act, const T* in, T* out, const int items,
            cudaStream_t stream);

template <typename T>
int rotary_embed(const T *in, const T *cos_sin, const int* mask, T* out,
            const int bs, const int hnum, const int len, const int dims,
            cudaStream_t stream);

template <typename T>
int rsqrt(const T *in, T *out,
            const int len, float eps,
            cudaStream_t stream);

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
        fprintf(stderr, "Failed to launch mask_float kernel (error code %s)!\n", cudaGetErrorString(err));
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
        fprintf(stderr, "Failed to launch alibi_float kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(-1);
    }
    return 0;
}

//----------------
__global__ void rsqrt_float(const float *in, float *out, const int len, float eps) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if ( e >= len ) {
        return;
    }
    out[e] = sqrt(1.0 / (in[e]*in[e] + eps));
}

template<>
int rsqrt<float>(const float *in, float *out, const int len, float eps, cudaStream_t stream) {
    dim3 block_size(256);
	dim3 num_of_blocks((len + block_size.x - 1) / block_size.x);
   
    rsqrt_float <<< num_of_blocks, block_size, 0, stream >>> (in, out, len, eps);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch reverse_float kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(-1);
    }
    return 0;
}

//----------------
__global__ void rotary_embed_float(const float *in, const float *cos_sin, const int *mask, float *out, 
                                   const int bs, const int hnum, const int len, const int dims) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if ( e >= bs * hnum * len ) {
        return;
    }
    in = in + e * dims;
    out = out + e * dims;

    int b = e / (hnum * len);
    int l = e % len;

    int pos = l + b * len;
    if ( mask[pos] >= 1 ) { 
        cos_sin = cos_sin + l * dims * 2;
    }
    
    for (int i = 0; i < dims / 2; i++) {
        int ii = i + dims/2;
        float x = in[i];
        float y = in[ii];
        out[i] = cos_sin[i*2] * x - cos_sin[i*2+1] * y;
        out[ii] = cos_sin[ii*2] * y + cos_sin[ii*2+1] * x;
    }
}

template<>
int rotary_embed<float>( const float *in, const float *cos_sin, const int* mask, float *out, 
                         const int bs, const int hnum, const int len, const int dims, cudaStream_t stream) {
    dim3 block_size(256);
	dim3 num_of_blocks( (bs*hnum*len + block_size.x - 1) / block_size.x);
   
    rotary_embed_float <<< num_of_blocks, block_size, 0, stream >>> (in, cos_sin, mask, out, bs, hnum, len, dims);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch  rotary_embed_float kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(-1);
    }
    return 0;
}

//----------------
__global__ void silu_product_float(const float *in_act, const float *in, float *out, const int items) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= items ) {
        return;
    }
    float act = in_act[i];
    float in_ = in[i];
    out[i] = act / (1.f + __expf(-act)) * in_;
}

template<>
int silu_product<float>( const float *in_act, const float *in, float *out, 
                         const int items, cudaStream_t stream) {
    dim3 block_size(256);
	dim3 num_of_blocks((items + block_size.x - 1) / block_size.x);
   
    silu_product_float <<< num_of_blocks, block_size, 0, stream >>> (in_act, in, out, items);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch silu_product_float kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(-1);
    }
    return 0;
}


}}
