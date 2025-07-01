#include "io.h"

/* 
 * Function to read the data from a binary file and return it as a 3D array
 * Parameters:
 * - filename: the name of the binary file to read
 * - shape_filename: the name of the shape file that contains the dimensions of the 3D array
 * Returns:
 * - A 3D array of float values read from the binary file
 */
Array3D<float> IO::read(const std::string& filename, const std::string& shape_filename) {
    // Check if the file exists
    if (!std::filesystem::exists(filename)) {
        throw std::runtime_error("File does not exist: " + filename);
    }
    // Read the shape information from the shape file
    std::vector<size_t> shape = read_shape(shape_filename);

    // Check if the shape information is valid
    if (shape.size() != 3) {
        throw std::runtime_error("Invalid shape information");
    }

    size_t depth = shape[0];
    size_t rows = shape[1];
    size_t cols = shape[2];

    // Create a 3D array with the dimensions
    Array3D<float> data(depth, rows, cols);
    std::ifstream file(filename, std::ios::binary);

    // Check if the file was opened successfully
    if (!file) {
        throw std::runtime_error("Error opening file: " + filename);
    }

    // Read the data from the file into the 3D array
    for (size_t d = 0; d < depth; ++d) {
        for (size_t r = 0; r < rows; ++r) {
            file.read(reinterpret_cast<char*>(&data(d, r, 0)), cols * sizeof(float));
            if (!file) {
                throw std::runtime_error("Error reading row " + std::to_string(r) + " of depth " + std::to_string(d) + " from file: " + filename);
            }
        }
    }

    file.close(); 
    return data; 
}

/*
 * Function to read the shape information from a shape file
 * Parameters:
 * - shape_filename: the name of the shape file that contains the dimensions of the 3D array
 * Returns:
 * - A vector of size_t values representing the dimensions of the 3D array
 */
std::vector<size_t> IO::read_shape(const std::string& shape_filename) {
    std::vector<size_t> shape;
    std::ifstream file(shape_filename);
    
    // Check if the shape file was opened successfully
    if (!file) {
        throw std::runtime_error("Error opening shape file: " + shape_filename);
    }

    // Read the shape information from the file
    std::string line;
    std::getline(file, line);
    std::stringstream ss(line);
    std::string item;
    
    while (std::getline(ss, item, ',')) {
        shape.push_back(std::stoul(item));
    }

    file.close();
    return shape;
}

/* Function to export the 3D array data to a binary file
 * Parameters:
 * - data: the 3D array of data to be exported
 * - filename: the name of the binary file to write to
 */
void IO::export_data(const Array3D<float>& data, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    
    // Check if the file was opened successfully
    if (!file) {
        throw std::runtime_error("Error opening file for writing: " + filename);
    }

    size_t depth = data.get_depth();
    size_t rows = data.get_rows();
    size_t cols = data.get_cols();

    // Define the dimensions of each sub-band
    size_t sub_depth = depth / 2;
    size_t sub_rows = rows / 2;
    size_t sub_cols = cols / 2;

    // Lambda function to export a sub-band of the 3D array
    auto export_subband = [&](size_t offset_depth, size_t offset_rows, size_t offset_cols) {
        file.write(reinterpret_cast<const char*>(&sub_depth), sizeof(sub_depth));
        file.write(reinterpret_cast<const char*>(&sub_rows), sizeof(sub_rows));
        file.write(reinterpret_cast<const char*>(&sub_cols), sizeof(sub_cols));

        for (size_t d = 0; d < sub_depth; ++d) {
            for (size_t r = 0; r < sub_rows; ++r) {
                file.write(reinterpret_cast<const char*>(&data(offset_depth + d, offset_rows + r, offset_cols)), sub_cols * sizeof(float));
            }
        }
    };

    // Export each sub-band
    export_subband(0, 0, 0); // LLL
    export_subband(0, 0, sub_cols); // LLH
    export_subband(0, sub_rows, 0); // LHL
    export_subband(0, sub_rows, sub_cols); // LHH
    export_subband(sub_depth, 0, 0); // HLL
    export_subband(sub_depth, 0, sub_cols); // HLH
    export_subband(sub_depth, sub_rows, 0); // HHL
    export_subband(sub_depth, sub_rows, sub_cols); // HHH

    file.close();
}

/* Function to construct filenames based on input parameters
 * Parameters:
 * - file_number: the file number
 * - dataset_type: the dataset type (CT/MR)
 * - mr_type: the MR type (T1DUAL/T2SPIR)
 * - phase_type: the phase type (InPhase/OutPhase)
 * Returns:
 * - A tuple containing the binary filename, shape filename, and output filename
 */
std::tuple<std::string, std::string, std::string> IO::construct_filenames(const std::string& file_number, const std::string& dataset_type, const std::string& mr_type, const std::string& phase_type, const std::string& filter_type, int levels) {
    std::string binary_filename = "data/inputs/" + file_number + "_" + dataset_type + (dataset_type == "MR" ? "_" + mr_type + (mr_type == "T1DUAL" ? "_" + phase_type : "") : "") + ".bin";
    std::string shape_filename = "data/inputs/" + file_number + "_" + dataset_type + (dataset_type == "MR" ? "_" + mr_type + (mr_type == "T1DUAL" ? "_" + phase_type : "") : "") + "_shape.txt";
    std::string output_filename = "data/outputs/" + file_number + "_" + dataset_type + (dataset_type == "MR" ? "_" + mr_type + (mr_type == "T1DUAL" ? "_" + phase_type : "") : "") + "_" + filter_type + "_" + std::to_string(levels) + ".bin";

    return std::make_tuple(binary_filename, shape_filename, output_filename);
}


// Function to export the inverse transform data to a binary file
bool IO::export_inverse(const Array3D<float>& data, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    
    // Check if the file was opened successfully
    if (!file) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return false;
    }

    size_t depth = data.get_depth();
    size_t rows = data.get_rows();
    size_t cols = data.get_cols();

    // Write the data to the file without the dimensions
    for (size_t d = 0; d < depth; ++d) {
        for (size_t r = 0; r < rows; ++r) {
            file.write(reinterpret_cast<const char*>(&data(d, r, 0)), cols * sizeof(float));
        }
    }

    file.close();
    return true;
}