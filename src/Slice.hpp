#ifndef __SLICE_HPP__
#define __SLICE_HPP__

#include "tthresh.hpp"
#include <string>
#include <sstream>

using namespace std;

class Slice {

public:

    int32_t points[3] = {INT32_MAX, INT32_MAX, 1};
    int32_t max_upper = INT32_MAX;
    bool downsample = false;

    Slice(int32_t lower, int32_t stride, int32_t upper, bool downsample);  // Create a slice from its data
    Slice(string description); // Create a slice from its NumPy-like description
    const uint32_t get_size(); // Number of elements encompassed by the slice
    const bool is_standard(); // Whether it is the (0,1,-1) slice (equivalent to doing nothing)

    friend ostream& operator<<(ostream& os, const Slice& slice);
    void update(uint32_t size);
};

Slice::Slice(int32_t lower, int32_t upper, int32_t stride, bool downsample=false) {
       points[0] = lower;
       points[1] = upper;
       points[2] = stride;
       downsample = downsample; // If true, new samples are computed as averages; otherwise, decimation is used
}

Slice::Slice(string description) {
    char delim = 0;
    if (description.find(':') != string::npos)
        delim = ':';
    if (description.find('/') != string::npos) {
        if (delim == ':')
            display_error("Slicing argument \""+description+"\" not understood");
        delim = '/';
        downsample = true;
    }
    stringstream ss1(description);
    string token;
    uint8_t n_parts = 0; // Should become 1, 2 or 3
    while(getline(ss1, token, delim)) {
        n_parts++;
        if (n_parts > 3)
            display_error("Slicing argument \""+description+"\" not understood");
        stringstream ss2(token);
        int32_t point = INT32_MAX; // Default value, used to detect missing parts (e.g. "::")
        ss2 >> point;
        if (point != INT32_MAX)
            points[n_parts-1] = point;
    }
    if (n_parts == 1 and description[description.size()-1] != delim) // E.g. "3"; indicates a single slice
        points[1] = points[0]+1;
    if (points[2] < 0) {
        if (points[0] == INT32_MAX) points[0] = -1;
        if (points[1] == INT32_MAX) points[1] = 0;
    }
    else if (points[2] > 0) {
        if (points[0] == INT32_MAX) points[0] = 0;
        if (points[1] == INT32_MAX) points[1] = -1;
    }
    else
        display_error("Slicing argument \""+description+"\" not understood");
}

const uint32_t Slice::get_size() {
    assert(points[1] > 0);
    return ceil((points[1]-points[0])/double(points[2]));
}

const bool Slice::is_standard() {
    return points[0] == 0 and (points[1] == -1 or points[1] == max_upper) and points[2] == 1;
}

ostream& operator<<(ostream& os, const Slice& slice)
{
    char delim = ':';
    if (slice.downsample) delim = '/';
    os << slice.points[0] << delim << slice.points[1] << delim << slice.points[2];
    return os;
}

void Slice::update(uint32_t size) {
    if (points[0] == -1)
        points[0] = size;
    else if (points[1] == -1)
        points[1] = size;
    max_upper = size;
}

#endif // SLICE_HPP
