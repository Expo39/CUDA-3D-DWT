#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cuda_runtime.h>
#include <cassert>

#include "utilities/utils.h"


// Kernel function prototypes
__global__ void dim0_kernel(float* data, float* temp, const float* lpf, const float* hpf, size_t filter_size, size_t depth, size_t rows, size_t cols);
__global__ void dim1_kernel(float* data, float* temp, const float* lpf, const float* hpf, size_t filter_size, size_t depth, size_t rows, size_t cols);
__global__ void dim2_kernel(float* data, float* temp, const float* lpf, const float* hpf, size_t filter_size, size_t depth, size_t rows, size_t cols);

// Utility function prototypes
__global__ void copy_transformed_data(float* d_transformed, float* d_final, size_t depth, size_t rows, size_t cols, size_t orig_depth, size_t orig_rows, size_t orig_cols);
__global__ void extract_lll_subband(float* d_data, float* d_temp, size_t depth, size_t rows, size_t cols);

#endif // KERNELS_CUH