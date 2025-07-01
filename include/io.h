#ifndef IO_H
#define IO_H

#include "utilities/utils.h"
#include <iostream>
#include <sstream>
#include <fstream>
#include <filesystem>
#include <tuple>
#include <vector>

class IO {
public:
    // Read the data from a binary file and return it as a 3D array
    static Array3D<float> read(const std::string& filename, const std::string& shape_filename);

    // Export the data to a binary file
    static void export_data(const Array3D<float>& data, const std::string& filename);

    // Construct filenames based on input parameters
    static std::tuple<std::string, std::string, std::string> construct_filenames(const std::string& file_number, const std::string& dataset_type, const std::string& mr_type, const std::string& phase_type, const std::string& filter_type, int levels);

    static bool export_inverse(const Array3D<float>& data, const std::string& filename);

private:
    // Read the shape information from a shape file
    static std::vector<size_t> read_shape(const std::string& shape_filename);
};

#endif // IO_H