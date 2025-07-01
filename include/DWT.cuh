#ifndef DWT_H
#define DWT_H

#include "io.h"
#include "filters.h"
#include "inverse.h"
#include "convolve.cuh"
#include "utilities/utils.h"

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#ifdef DEBUG
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)
#else
#define CUDA_CHECK(call) call
#endif

// Function to perform the 3D wavelet transform
Array3D<float> dwt_3d(const float* lpf, const float* hpf, size_t filter_size, Array3D<float>& data, size_t depth, size_t rows, size_t cols, int levels);

// Function to process the wavelet transform
void process_wavelet(const std::string& binary_filename, const std::string& output_filename, const std::string& filter_type, int levels);

#endif // DWT_H