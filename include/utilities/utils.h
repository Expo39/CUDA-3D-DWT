#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <cassert>

using namespace std;

// Template class for a custom 3D array
template <class T>
class Array3D {
private:
    vector<T> data; 
    size_t depth, rows, cols; 

public:
    // Default constructor
    Array3D() : depth(0), rows(0), cols(0) {}

    // Constructor to initialize the 3D array with given dimensions
    Array3D(size_t d, size_t r, size_t c) : data(d * r * c), depth(d), rows(r), cols(c) {}

    // Access the underlying data as a 1D array
    T& operator[](size_t index) {
        assert(index < data.size());
        return data[index];
    }

    // Const access to the underlying data as a 1D array
    const T& operator[](size_t index) const {
        assert(index < data.size());
        return data[index];
    }

    // Non-const element access operator
    T& operator()(size_t d, size_t r, size_t c) {
        assert(d < depth && r < rows && c < cols); 
        return data[d * rows * cols + r * cols + c]; // Calculate the 1D index and return the element
    }

    // Const element access operator
    const T& operator()(size_t d, size_t r, size_t c) const {
        assert(d < depth && r < rows && c < cols); 
        return data[d * rows * cols + r * cols + c]; // Calculate the 1D index and return the element
    }

    // Get the depth of the 3D array
    size_t get_depth() const { return depth; }

    // Get the number of rows in the 3D array
    size_t get_rows() const { return rows; }

    // Get the number of columns in the 3D array
    size_t get_cols() const { return cols; }

    // Get the total number of elements in the 3D array
    size_t size() const { return data.size(); }

    // Get a reference to the underlying data vector
    vector<T>& get_data() { return data; }

    // Get a const reference to the underlying data vector
    const vector<T>& get_data() const { return data; }

    // Resize the 3D array
    void resize(size_t new_depth, size_t new_rows, size_t new_cols) {
        depth = new_depth;
        rows = new_rows;
        cols = new_cols;
        data.resize(depth * rows * cols);
    }
};

#endif // UTILS_H