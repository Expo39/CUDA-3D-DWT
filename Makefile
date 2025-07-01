# Compiler
CXX = g++
NVCC = nvcc

# CUDA paths
CUDA_INSTALL_DIR = /usr/local/cuda
CUDA_INCLUDE_PATH = $(CUDA_INSTALL_DIR)/include
CUDA_LIB_PATH = $(CUDA_INSTALL_DIR)/lib64

# Boost paths
BOOST_INCLUDE_PATH = /usr/include
BOOST_LIB_PATH = /usr/lib/x86_64-linux-gnu

# Compiler flags
CXXFLAGS = -Iinclude -Iinclude/utilities -I$(CUDA_INCLUDE_PATH) -I$(BOOST_INCLUDE_PATH)
NVCCFLAGS = -Iinclude -Iinclude/utilities -I$(CUDA_INCLUDE_PATH) -arch=sm_86

# Debug build flags
DEBUG_FLAGS = -g -O0 -DDEBUG -Wall -Wextra -Wpedantic
NVCC_DEBUG_FLAGS = -G -O0 -DDEBUG

# Release build flags
RELEASE_FLAGS = -O3 -DNDEBUG -Wall -Wextra -Wpedantic
NVCC_RELEASE_FLAGS = -O3 -DNDEBUG

# Linker flags
LDFLAGS = -L$(CUDA_LIB_PATH) -lcudart -L$(BOOST_LIB_PATH) -lboost_program_options -lboost_system -lboost_filesystem

# Target executable names
DEBUG_TARGET = DEBUG
RELEASE_TARGET = DWT

# Source files
SRCS = src/main.cpp src/io.cpp src/filters.cpp src/inverse.cpp 
CUDA_SRCS = src/convolve.cu src/DWT.cu

# Object files
DEBUG_OBJS = $(addprefix build/debug/, $(notdir $(SRCS:.cpp=.o))) build/debug/convolve.o build/debug/DWT.o
RELEASE_OBJS = $(addprefix build/release/, $(notdir $(SRCS:.cpp=.o))) build/release/convolve.o build/release/DWT.o

# Default target
all: release

# Debug build target
debug: $(DEBUG_TARGET)

# Release build target
release: $(RELEASE_TARGET)

# Link the debug target executable
$(DEBUG_TARGET): $(DEBUG_OBJS)
	@echo "Debug object files: $(DEBUG_OBJS)"
	$(NVCC) -o $@ $^ $(LDFLAGS)

# Link the release target executable
$(RELEASE_TARGET): $(RELEASE_OBJS)
	@echo "Release object files: $(RELEASE_OBJS)"
	$(NVCC) -o $@ $^ $(LDFLAGS)

# Compile source files into debug object files
build/debug/%.o: src/%.cpp
	@mkdir -p $(dir $@)
	@echo "Compiling $< to $@"
	$(CXX) $(CXXFLAGS) $(DEBUG_FLAGS) -c $< -o $@

# Compile source files into release object files
build/release/%.o: src/%.cpp
	@mkdir -p $(dir $@)
	@echo "Compiling $< to $@"
	$(CXX) $(CXXFLAGS) $(RELEASE_FLAGS) -c $< -o $@

# Compile CUDA source files into debug object files
build/debug/%.o: src/%.cu
	@mkdir -p $(dir $@)
	@echo "Compiling $< to $@"
	$(NVCC) $(NVCCFLAGS) $(NVCC_DEBUG_FLAGS) -c $< -o $@

# Compile CUDA source files into release object files
build/release/%.o: src/%.cu
	@mkdir -p $(dir $@)
	@echo "Compiling $< to $@"
	$(NVCC) $(NVCCFLAGS) $(NVCC_RELEASE_FLAGS) -c $< -o $@

# Clean up build artifacts
clean:
	rm -rf build $(DEBUG_TARGET) $(RELEASE_TARGET)

.PHONY: all debug release clean