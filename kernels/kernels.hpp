#ifndef _ALL_KERNELS_HPP_
#define _ALL_KERNELS_HPP_

#include <cublasLt.h>
#include "lightseq/kernels.h"

namespace br { namespace cuda {

template <typename T>
int silu_product(const T *in_act, const T* in, T* out, const int items,
            cudaStream_t stream);

template <typename T>
int rotary_embed(const T *in, const T *cos_sin, const int* mask, T* out,
            const int bs, const int hnum, const int len, const int dims,
            cudaStream_t stream);

template <typename T>
int rsqrt(const T *in, T *out,
            const int len,
            float eps,
            cudaStream_t stream);

template <typename T>
int embed_forward(const int *ids, const T *table, T *out,
                  const int len,
                  const int hidden_size,
                  cudaStream_t stream);

template <typename T>
int causal_mask(const int *mask, T *out,
                  const int batch,
                  const int tokens,
                  cudaStream_t stream);

int gelu_forward(const float* src, float* target, int nElementNumber, cudaStream_t stream);
int gelu_backward(const float* out_g, const float* xi, float* x_g, int nElementNumber, cudaStream_t stream);
int nllloss_forward(const int* ids, const float* logsoftmax, float *output, float *dout, int n, int vocab, float loss_scale, cudaStream_t stream);
void LtSgemm(cublasLtHandle_t ltHandle,
             cublasOperation_t transa,
             cublasOperation_t transb,
             int m,
             int n,
             int k,
             const float *alpha, /* host pointer */
             const void *A, cudaDataType_t at,
             int lda,
             const void *B, cudaDataType_t bt,
             int ldb,
             const float *beta, /* host pointer */
             void *C, cudaDataType_t ct,
             int ldc,
             void *workspace,
             size_t workspaceSize);

void LtSgemmBatched(cublasLtHandle_t ltHandle,
             cublasOperation_t transa,
             cublasOperation_t transb,
             int m,
             int n,
             int k,
             const float *alpha, /* host pointer */
             const float *A,
             int lda,
             const float *B,
             int ldb,
             const float *beta, /* host pointer */
             float *C,
             int ldc,
             int batchCount,
             void *workspace,
             size_t workspaceSize);

}}

#endif
