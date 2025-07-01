#ifndef INVERSE_H
#define INVERSE_H

#include "utilities/utils.h"
#include "utilities/jbutil.h"
#include "filters.h"

class Inverse {
public:
    // Constructor
    Inverse(const float* lpf, const float* hpf, size_t filter_size);

    // Inverse Convolution functions
    void dim0(Array3D<float>& data, size_t depth_limit, size_t row_limit, size_t col_limit) const;
    void dim1(Array3D<float>& data, size_t depth_limit, size_t row_limit, size_t col_limit) const;
    void dim2(Array3D<float>& data, size_t depth_limit, size_t row_limit, size_t col_limit) const;

    // Inverse DWT function
    Array3D<float> inverse_dwt_3d(Array3D<float>& data, int levels) const;

private:
    // Filter coefficients
    const float* lpf;
    const float* hpf;
    size_t filter_size;

};

#endif // INVERSE_H