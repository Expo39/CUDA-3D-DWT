#include "inverse.h"

/* 
 * Constructor for the Inverse class
 * Parameters:
 * - lpf: low-pass filter
 * - hpf: high-pass filter
 * - filter_size: size of the filter
 */
Inverse::Inverse(const float* lpf, const float* hpf, size_t filter_size)
    : lpf(lpf), hpf(hpf), filter_size(filter_size) {}

/* 
 * Perform inverse convolution along the first dimension
 * Parameters:
 * - data: 3D array of data to be transformed
 * - depth_limit: depth limit of the data
 * - row_limit: row limit of the data
 * - col_limit: column limit of the data
 */
void Inverse::dim0(Array3D<float>& data, size_t depth_limit, size_t row_limit, size_t col_limit) const {
    Array3D<float> temp(data);

    for (size_t d = 0; d < depth_limit; ++d) {
        for (size_t c = 0; c < col_limit; ++c) {
            // Initialize the data array to zero
            for (size_t i = 0; i < row_limit; ++i) {
                data(d, i, c) = 0.0f;
            }
            // Perform inverse convolution
            for (size_t i = 0; i < row_limit / 2; ++i) {
                float low_val = temp(d, i, c);
                float high_val = temp(d, i + row_limit / 2, c);

                for (size_t j = 0; j < filter_size; ++j) {
                    size_t index = (2 * i + j) % row_limit;
                    data(d, index, c) += (lpf[j] * low_val) + (hpf[j] * high_val);
                }
            }
        }
    }
}

/* 
 * Perform inverse convolution along the second dimension
 * Parameters:
 * - data: 3D array of data to be transformed
 * - depth_limit: depth limit of the data
 * - row_limit: row limit of the data
 * - col_limit: column limit of the data
 */
void Inverse::dim1(Array3D<float>& data, size_t depth_limit, size_t row_limit, size_t col_limit) const {
    Array3D<float> temp(data);

    for (size_t d = 0; d < depth_limit; ++d) {
        for (size_t r = 0; r < row_limit; ++r) {
            // Initialize the data array to zero
            for (size_t i = 0; i < col_limit; ++i) {
                data(d, r, i) = 0.0f;
            }
            // Perform inverse convolution
            for (size_t i = 0; i < col_limit / 2; ++i) {
                float low_val = temp(d, r, i);
                float high_val = temp(d, r, i + col_limit / 2);

                for (size_t j = 0; j < filter_size; ++j) {
                    size_t index = (2 * i + j) % col_limit;
                    data(d, r, index) += (lpf[j] * low_val) + (hpf[j] * high_val);
                }
            }
        }
    }
}

/* 
 * Perform inverse convolution along the third dimension
 * Parameters:
 * - data: 3D array of data to be transformed
 * - depth_limit: depth limit of the data
 * - row_limit: row limit of the data
 * - col_limit: column limit of the data
 */
void Inverse::dim2(Array3D<float>& data, size_t depth_limit, size_t row_limit, size_t col_limit) const {
    Array3D<float> temp(data);

    for (size_t r = 0; r < row_limit; ++r) {
        for (size_t c = 0; c < col_limit; ++c) {
            // Initialize the data array to zero
            for (size_t i = 0; i < depth_limit; ++i) {
                data(i, r, c) = 0.0f;
            }
            // Perform inverse convolution
            for (size_t i = 0; i < depth_limit / 2; ++i) {
                float low_val = temp(i, r, c);
                float high_val = temp(i + depth_limit / 2, r, c);

                for (size_t j = 0; j < filter_size; ++j) {
                    size_t index = (2 * i + j) % depth_limit;
                    data(index, r, c) += (lpf[j] * low_val) + (hpf[j] * high_val);
                }
            }
        }
    }
}

/* 
 * Perform the inverse 3D wavelet transform
 * Parameters:
 * - data: 3D array of data to be transformed
 * - levels: number of decomposition levels
 * Returns:
 * - 3D array of reconstructed data
 */
Array3D<float> Inverse::inverse_dwt_3d(Array3D<float>& data, int levels) const {
    // Get the initial dimensions of the data
    size_t depth = data.get_depth();
    size_t rows = data.get_rows();
    size_t cols = data.get_cols();

    // Adjust the dimensions for the number of levels
    vector<size_t> depth_levels(levels);
    vector<size_t> row_levels(levels);
    vector<size_t> col_levels(levels);

    depth_levels[0] = depth;
    row_levels[0] = rows;
    col_levels[0] = cols;

    for (int i = 1; i < levels; ++i) {
        depth_levels[i] = (depth_levels[i-1]) / 2;
        row_levels[i] = (row_levels[i-1]) / 2;
        col_levels[i] = (col_levels[i-1]) / 2;
    }

    // Perform inverse convolution for each level
    for (int level = levels - 1; level >= 0; --level) {
        dim2(data, depth_levels[level], row_levels[level], col_levels[level]);
        dim1(data, depth_levels[level], row_levels[level], col_levels[level]);
        dim0(data, depth_levels[level], row_levels[level], col_levels[level]);
    }

    // Return the reconstructed data
    return data;
}