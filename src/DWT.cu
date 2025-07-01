#include "DWT.cuh"

/* 
 * Perform the multi level 3D wavelet transform
 * Parameters:
 * - lpf: low-pass filter
 * - hpf: high-pass filter
 * - filter_size: size of the filter
 * - data: 3D array of data to be transformed
 * - depth: depth of the 3D data array
 * - rows: number of rows in the 3D data array
 * - cols: number of columns in the 3D data array
 * - levels: number of decomposition levels
 * Returns:
 * - 3D array of transformed data
 */
Array3D<float> dwt_3d(const float* lpf, const float* hpf, size_t filter_size, Array3D<float>& data, size_t depth, size_t rows, size_t cols, int levels) {
    nvtxRangeId_t rangeId = nvtxRangeStartA("DWT 3D Transform");

    // Allocate memory for the input data on the device
    float* d_data;
    size_t data_size = depth * rows * cols * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_data, data_size));
    CUDA_CHECK(cudaMemcpy(d_data, data.get_data().data(), data_size, cudaMemcpyHostToDevice));

    // Allocate memory for the final transformed data on the device
    float* d_final;
    size_t orig_depth = depth;
    size_t orig_rows = rows;
    size_t orig_cols = cols;
    size_t final_data_size = orig_depth * orig_rows * orig_cols * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_final, final_data_size));
    CUDA_CHECK(cudaMemset(d_final, 0, final_data_size)); // Initialize to zero

    // Allocate memory for the filters on the device
    float* d_lpf;
    float* d_hpf;
    size_t filter_size_bytes = filter_size * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_lpf, filter_size_bytes));
    CUDA_CHECK(cudaMalloc(&d_hpf, filter_size_bytes));
    CUDA_CHECK(cudaMemcpy(d_lpf, lpf, filter_size_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_hpf, hpf, filter_size_bytes, cudaMemcpyHostToDevice));

    for (int level = 0; level < levels; ++level) {
        // Allocate memory for temporary data on the device
        float* d_temp;
        CUDA_CHECK(cudaMalloc(&d_temp, data_size));

        dim3 blockDim(16, 8, 8);
        dim3 gridDim0((rows + blockDim.x - 1) / blockDim.x, (cols + blockDim.y - 1) / blockDim.y, (depth + blockDim.z - 1) / blockDim.z);
        dim3 gridDim1((cols + blockDim.x - 1) / blockDim.x, (rows + blockDim.y - 1) / blockDim.y, (depth + blockDim.z - 1) / blockDim.z);
        dim3 gridDim2((depth + blockDim.x - 1) / blockDim.x, (cols + blockDim.y - 1) / blockDim.y, (rows + blockDim.z - 1) / blockDim.z);

        // Perform convolution along the first dimension
        nvtxRangePushA("Perform convolution along the first dimension");
        dim0_kernel<<<gridDim0, blockDim, filter_size * sizeof(float) * 2>>>(d_data, d_temp, d_lpf, d_hpf, filter_size, depth, rows, cols);
        CUDA_CHECK(cudaDeviceSynchronize());
        nvtxRangePop();

        // Perform convolution along the second dimension
        nvtxRangePushA("Perform convolution along the second dimension");
        dim1_kernel<<<gridDim1, blockDim, filter_size * sizeof(float) * 2>>>(d_temp, d_data, d_lpf, d_hpf, filter_size, depth, rows, cols);
        CUDA_CHECK(cudaDeviceSynchronize());
        nvtxRangePop();

        // Perform convolution along the third dimension
        nvtxRangePushA("Perform convolution along the third dimension");
        dim2_kernel<<<gridDim2, blockDim, filter_size * sizeof(float) * 2>>>(d_data, d_temp, d_lpf, d_hpf, filter_size, depth, rows, cols);
        CUDA_CHECK(cudaDeviceSynchronize());
        nvtxRangePop();

        // Copy transformed data to the final array
        nvtxRangePushA("Copy transformed data to final array");
        dim3 gridDim((cols + blockDim.x - 1) / blockDim.x, (rows + blockDim.y - 1) / blockDim.y, (depth + blockDim.z - 1) / blockDim.z);
        copy_transformed_data<<<gridDim, blockDim>>>(d_temp, d_final, depth, rows, cols, orig_depth, orig_rows, orig_cols);
        CUDA_CHECK(cudaDeviceSynchronize());
        nvtxRangePop();

        // Update dimensions for the next level
        size_t new_depth = depth / 2;
        size_t new_rows = rows / 2;
        size_t new_cols = cols / 2;
        size_t new_data_size = new_depth * new_rows * new_cols * sizeof(float);

        // Allocate memory for the new data on the device
        float* d_new_data;
        CUDA_CHECK(cudaMalloc(&d_new_data, new_data_size));

        // Extract the LLL subband
        nvtxRangePushA("Extract LLL subband");
        extract_lll_subband<<<gridDim, blockDim>>>(d_temp, d_new_data, depth, rows, cols);
        CUDA_CHECK(cudaDeviceSynchronize());
        nvtxRangePop();

        // Free the old data and update pointers
        CUDA_CHECK(cudaFree(d_data));
        d_data = d_new_data;
        data_size = new_data_size;

        depth = new_depth;
        rows = new_rows;
        cols = new_cols;

        // Free the temporary data
        CUDA_CHECK(cudaFree(d_temp));
    }

    // Copy the final transformed data back to the host
    Array3D<float> result(orig_depth, orig_rows, orig_cols);
    CUDA_CHECK(cudaMemcpy(result.get_data().data(), d_final, final_data_size, cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_final));
    CUDA_CHECK(cudaFree(d_lpf));
    CUDA_CHECK(cudaFree(d_hpf));

    nvtxRangeEnd(rangeId);

    return result;
}

/* 
 * Process the wavelet transform
 * Parameters:
 * - binary_filename: the name of the binary file containing the input data
 * - output_filename: the name of the binary file to write the transformed data to
 * - filter_type: the type of wavelet filter to use (e.g., "haar", "db1")
 * - levels: the number of levels of decomposition
 */
void process_wavelet(const std::string& binary_filename, const std::string& output_filename, const std::string& filter_type, int levels) {
    const float* lpf;
    const float* hpf;
    const float* Ilpf;
    const float* Ihpf;
    size_t filter_size;

    // Determine the shape file based on the binary file name
    std::string shape_filename = binary_filename.substr(0, binary_filename.find_last_of('.')) + "_shape.txt";

    if (!get_filters(filter_type, lpf, hpf, Ilpf, Ihpf, filter_size)) {
        std::cerr << "Failed to get filters for type: " << filter_type << std::endl;
        return;
    }

    try {
        // Read the DICOM data into an array
        Array3D<float> dicom_data = IO::read(binary_filename, shape_filename);

        std::cout << "\nData read from " << binary_filename << " successfully.\n" << std::endl;

        // Print the characteristics of the input data
        std::cout << "Filter type: " << filter_type << std::endl;
        std::cout << "Filter size: " << filter_size << std::endl;
        std::cout << "Levels: " << levels << std::endl;


        /**********************Perform the Forward 3D DWT**********************/

        // Measure the time taken for the 3D wavelet transform
        double start_time = jbutil::gettime();


        // Perform the 3D wavelet transform directly on the GPU for all levels
        Array3D<float> final_result = dwt_3d(lpf, hpf, filter_size, dicom_data, dicom_data.get_depth(), dicom_data.get_rows(), dicom_data.get_cols(), levels);    


        double end_time = jbutil::gettime();
        double elapsed_time = end_time - start_time;

        /********************************************************************/


        std::cout << "Time taken for 3D Wavelet Transform: " << elapsed_time << " seconds\n" << std::endl;

        // Export the transformed data to a binary file
        IO::export_data(final_result, output_filename);

        std::cout << "Data exported to " << output_filename << " successfully.\n" << std::endl;

        // Perform the inverse 3D wavelet transform
        std::cout << "Performing Inverse 3D Wavelet Transform..." << std::endl;
        Inverse inverse(Ilpf, Ihpf, filter_size);
        Array3D<float> reconstructed_data = inverse.inverse_dwt_3d(final_result, levels);

        // Determine the inverse output filename
        std::string inverse_output_filename = "data/outputs/inverse_" + output_filename.substr(output_filename.find_last_of('/') + 1);

        // Export the reconstructed data to a binary file
        IO::export_inverse(reconstructed_data, inverse_output_filename);

        std::cout << "Inverse 3D Wavelet Transform completed successfully." << std::endl;
        std::cout << "Data exported to " << inverse_output_filename << " successfully." << std::endl;

    } catch (const std::runtime_error& e) {
        std::cerr << "Runtime error: " << e.what() << std::endl;
        return;
    }
}