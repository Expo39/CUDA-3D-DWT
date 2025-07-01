## Overview

This project implements the Mallat algorithm for 3D DWT decomposition using CUDA for maximum performance. It supports multiple wavelet filters and multi-level decomposition, designed for processing medical imaging data from the CHAOS (Combined Healthy Abdominal Organ Segmentation) dataset.

### Key Features

- **Multi-Level 3D DWT**: Support for multiple decomposition levels
- **Multiple Wavelets**: Support for multiple filters
- **Medical Data Support**: Optimised for CT and MR imaging datasets
- **CUDA Acceleration**: GPU-parallelised implementation for high performance
- **Memory Efficient**: Optimised memory access patterns for 3D data

## Building the Project

```bash
make
```

## Usage

```bash
./DWT --file-number 1 --dataset-type CT --filter-type db4 --levels 2
```

**Parameters:**
- `file_number`: Dataset identifier 
- `dataset_type`: CT or MR
- `filter_type`: db1, db2 or many more
- `levels`: Number of decomposition levels 
- `mr_type`: T1DUAL or T2SPIR (for MR datasets)
- `phase_type`: InPhase or OutPhase (for MR datasets)

## Algorithm Details

The implementation uses the Mallat algorithm for DWT decomposition:

1. **3D Convolution**: Applies low-pass and high-pass filters across each dimension
2. **Downsampling**: Reduces data size by factor of 2 in each dimension
3. **Multi-Level**: Recursively applies transform to low-frequency coefficients
4. **GPU Optimisation**: Parallel processing of independent convolution operations

## Dependencies

- NVIDIA CUDA Toolkit (11.0+)
- Boost Program Options
- C++17 compatible compiler

## Data Preparation

The project processes medical imaging data from DICOM files, which must be converted to binary format before processing:

### DICOM to Binary Conversion
The CHAOS dataset DICOM files are converted to binary format using the provided Python utilities:

```bash
cd python/
python create.py  # Converts DICOM files to binary format with shape metadata
```

This creates:
- `.bin` files: Raw binary data (float32 format)
- `.shape` files: Dimension metadata (width, height, depth)


## Testing and Validation

The implementation is validated using multiple approaches:

### 1. Serial Inverse Transform Validation
A serial C++ implementation (`inverse.cpp`) of the inverse DWT is used to reconstruct the original data:

```bash
# Run forward DWT
./DWT --file-number 1 --dataset-type CT --filter-type db4 --levels 2

# Validate with inverse transform using Python notebook
cd python/
jupyter notebook inverse.ipynb
```

### 2. PyWavelets Library Comparison
Python validation using the PyWavelets (pywt) library for reference comparison in the Testing notebook:

```bash
cd python/
jupyter notebook Testing.ipynb
```

This validates:
- **Coefficient accuracy**: Comparison with PyWavelets reference implementation
- **Reconstruction quality**: Mean Squared Error (MSE) analysis

