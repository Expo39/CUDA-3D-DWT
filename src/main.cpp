#include <iostream>
#include <stdexcept>
#include <string>
#include <filesystem>
#include <boost/program_options.hpp>

#include "io.h"
#include "DWT.cuh"

int main(int argc, char* argv[]) {
    try {
        // Define the options
        boost::program_options::options_description desc("Allowed options");
        desc.add_options()
            ("help", "produce help message")
            ("file-number", boost::program_options::value<std::string>()->required(), "file number (required)")
            ("dataset-type", boost::program_options::value<std::string>()->required(), "dataset type (CT/MR) (required)")
            ("filter-type", boost::program_options::value<std::string>()->required(), "filter type (required)")
            ("levels", boost::program_options::value<int>()->required(), "levels (required)")
            ("mr-type", boost::program_options::value<std::string>(), "MR type (T1DUAL/T2SPIR) (required if dataset type is MR)")
            ("phase-type", boost::program_options::value<std::string>(), "Phase type (InPhase/OutPhase) (required if dataset type is MR)");

        // Parse the command-line arguments
        boost::program_options::variables_map vm;
        boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);

        if (vm.count("help")) {
            std::cout << desc << "\n";
            return 1;
        }

        boost::program_options::notify(vm);

        // Get the values of the options
        std::string file_number = vm["file-number"].as<std::string>();
        std::string dataset_type = vm["dataset-type"].as<std::string>();
        std::string filter_type = vm["filter-type"].as<std::string>();
        int levels = vm["levels"].as<int>();

        if (dataset_type == "MR") {
            if (!vm.count("mr-type") || !vm.count("phase-type")) {
                std::cerr << "Error: Both 'mr-type' and 'phase-type' are required when 'dataset-type' is MR.\n";
                return 1;
            }
        }

        std::string mr_type = vm.count("mr-type") ? vm["mr-type"].as<std::string>() : "";
        std::string phase_type = vm.count("phase-type") ? vm["phase-type"].as<std::string>() : "";

        // Construct filenames based on input parameters
        auto [binary_filename, shape_filename, output_filename] = IO::construct_filenames(file_number, dataset_type, mr_type, phase_type, filter_type, levels);

        // Create the outputs directory if it does not exist
        std::filesystem::create_directories("data/outputs");

        // Perform the 3D wavelet transform
        process_wavelet(binary_filename, output_filename, filter_type, levels);

    } catch (const boost::program_options::error &e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    } catch (const std::invalid_argument& e) {
        std::cerr << "Invalid argument: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}