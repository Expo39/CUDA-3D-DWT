#include "convolve.cuh"

/* 
 * Kernel to copy transformed data to the final array
 * Parameters:
 * - d_transformed: pointer to the transformed data on the device
 * - d_final: pointer to the final data on the device
 * - depth: depth of the transformed data
 * - rows: number of rows in the transformed data
 * - cols: number of columns in the transformed data
 * - orig_depth: original depth of the data
 * - orig_rows: original number of rows in the data
 * - orig_cols: original number of columns in the data
 */
__global__ void copy_transformed_data(float* d_transformed, float* d_final, size_t depth, size_t rows, size_t cols, size_t orig_depth, size_t orig_rows, size_t orig_cols) {
    int d = blockIdx.z * blockDim.z + threadIdx.z;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (d < depth && r < rows && c < cols) {
        d_final[d * orig_rows * orig_cols + r * orig_cols + c] = d_transformed[d * rows * cols + r * cols + c];
    }
}

/* 
 * Kernel to extract the LLL subband from the transformed data
 * Parameters:
 * - d_transformed: pointer to the transformed data on the device
 * - d_data: pointer to the data on the device
 * - depth: depth of the transformed data
 * - rows: number of rows in the transformed data
 * - cols: number of columns in the transformed data
 */
__global__ void extract_lll_subband(float* d_transformed, float* d_data, size_t depth, size_t rows, size_t cols) {
    int d = blockIdx.z * blockDim.z + threadIdx.z;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (d < depth / 2 && r < rows / 2 && c < cols / 2) {
        d_data[d * (rows / 2) * (cols / 2) + r * (cols / 2) + c] = d_transformed[d * rows * cols + r * cols + c];
    }
}

/* 
 * Kernel to perform convolution along the first dimension
 * Parameters:
 * - data: pointer to the input data on the device
 * - temp: pointer to the temporary data on the device
 * - lpf: pointer to the low-pass filter on the device
 * - hpf: pointer to the high-pass filter on the device
 * - filter_size: size of the filter
 * - depth_limit: depth limit of the data
 * - row_limit: row limit of the data
 * - col_limit: column limit of the data
 */
__global__ void dim0_kernel(float* data, float* temp, const float* lpf, const float* hpf, size_t filter_size, size_t depth_limit, size_t row_limit, size_t col_limit) {
    extern __shared__ float shared_mem[];
    float* shared_lpf = shared_mem;
    float* shared_hpf = shared_mem + filter_size;

    // Load filters into shared memory
    if (threadIdx.x < filter_size) {
        shared_lpf[threadIdx.x] = lpf[threadIdx.x];
        shared_hpf[threadIdx.x] = hpf[threadIdx.x];
    }
    __syncthreads();

    size_t d = blockIdx.z * blockDim.z + threadIdx.z;
    size_t c = blockIdx.y * blockDim.y + threadIdx.y;
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (d < depth_limit && c < col_limit && i < row_limit / 2) {
        float sum_low = 0.0f;
        float sum_high = 0.0f;

        // Perform convolution
        for (size_t j = 0; j < filter_size; ++j) {
            size_t index = (2 * i + j) % row_limit;
            size_t data_index = d * row_limit * col_limit + index * col_limit + c;

            float input_val = data[data_index];
            sum_low += shared_lpf[j] * input_val;
            sum_high += shared_hpf[j] * input_val;
        }

        size_t low_index = d * row_limit * col_limit + i * col_limit + c;
        size_t high_index = d * row_limit * col_limit + (i + row_limit / 2) * col_limit + c;

        temp[low_index] = sum_low;
        temp[high_index] = sum_high;
    }
}

/* 
 * Kernel to perform convolution along the second dimension
 * Parameters:
 * - data: pointer to the input data on the device
 * - temp: pointer to the temporary data on the device
 * - lpf: pointer to the low-pass filter on the device
 * - hpf: pointer to the high-pass filter on the device
 * - filter_size: size of the filter
 * - depth_limit: depth limit of the data
 * - row_limit: row limit of the data
 * - col_limit: column limit of the data
 */
__global__ void dim1_kernel(float* data, float* temp, const float* lpf, const float* hpf, size_t filter_size, size_t depth_limit, size_t row_limit, size_t col_limit) {
    extern __shared__ float shared_mem[];
    float* shared_lpf = shared_mem;
    float* shared_hpf = shared_mem + filter_size;

    // Load filters into shared memory
    if (threadIdx.x < filter_size) {
        shared_lpf[threadIdx.x] = lpf[threadIdx.x];
        shared_hpf[threadIdx.x] = hpf[threadIdx.x];
    }
    __syncthreads();

    size_t d = blockIdx.z * blockDim.z + threadIdx.z;
    size_t r = blockIdx.y * blockDim.y + threadIdx.y;
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (d < depth_limit && r < row_limit && i < col_limit / 2) {
        float sum_low = 0.0f;
        float sum_high = 0.0f;

        // Perform convolution
        for (size_t j = 0; j < filter_size; ++j) {
            size_t index = (2 * i + j) % col_limit;
            size_t data_index = d * row_limit * col_limit + r * col_limit + index;

            float input_val = data[data_index];
            sum_low += shared_lpf[j] * input_val;
            sum_high += shared_hpf[j] * input_val;
        }

        size_t low_index = d * row_limit * col_limit + r * col_limit + i;
        size_t high_index = d * row_limit * col_limit + r * col_limit + (i + col_limit / 2);

        temp[low_index] = sum_low;
        temp[high_index] = sum_high;
    }
}

/* 
 * Kernel to perform convolution along the third dimension
 * Parameters:
 * - data: pointer to the input data on the device
 * - temp: pointer to the temporary data on the device
 * - lpf: pointer to the low-pass filter on the device
 * - hpf: pointer to the high-pass filter on the device
 * - filter_size: size of the filter
 * - depth_limit: depth limit of the data
 * - row_limit: row limit of the data
 * - col_limit: column limit of the data
 */
__global__ void dim2_kernel(float* data, float* temp, const float* lpf, const float* hpf, size_t filter_size, size_t depth_limit, size_t row_limit, size_t col_limit) {
    extern __shared__ float shared_mem[];
    float* shared_lpf = shared_mem;
    float* shared_hpf = shared_mem + filter_size;

    // Load filters into shared memory
    if (threadIdx.x < filter_size) {
        shared_lpf[threadIdx.x] = lpf[threadIdx.x];
        shared_hpf[threadIdx.x] = hpf[threadIdx.x];
    }
    __syncthreads();

    size_t r = blockIdx.z * blockDim.z + threadIdx.z;
    size_t c = blockIdx.y * blockDim.y + threadIdx.y;
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < row_limit && c < col_limit && i < depth_limit / 2) {
        float sum_low = 0.0f;
        float sum_high = 0.0f;

        // Perform convolution
        for (size_t j = 0; j < filter_size; ++j) {
            size_t index = (2 * i + j) % depth_limit;
            size_t data_index = index * row_limit * col_limit + r * col_limit + c;

            float input_val = data[data_index];
            sum_low += shared_lpf[j] * input_val;
            sum_high += shared_hpf[j] * input_val;
        }

        size_t low_index = i * row_limit * col_limit + r * col_limit + c;
        size_t high_index = (i + depth_limit / 2) * row_limit * col_limit + r * col_limit + c;

        temp[low_index] = sum_low;
        temp[high_index] = sum_high;
    }
}