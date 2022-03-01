/*
 * Copyright (c) 2016-2017, Rafael Ballester-Ripoll
 *                          (Visualization and MultiMedia Lab, University of Zurich),
 *                          rballester@ifi.uzh.ch
 *
 * Licensed under the LGPLv3.0 (https://github.com/rballester/tthresh/blob/master/LICENSE)
 */

#ifndef __SLICE_HPP__
#define __SLICE_HPP__

#include "tthresh.hpp"
#include <string>
#include <sstream>
#include <cmath>

using namespace std;

// Cutout/downsampling modes
enum Reduction { Downsampling, Box, Lanczos };

class Slice {

public:

    int32_t points[3] = {INT32_MAX, INT32_MAX, 1};
    int32_t max_upper = INT32_MAX;
    Reduction reduction = Downsampling;

    Slice(int32_t lower, int32_t stride, int32_t upper);  // Create a slice from its data
    Slice(string description); // Create a slice from its NumPy-like description
    const int32_t get_size(); // Number of elements encompassed by the slice
    const bool is_standard(); // Whether it is the (0,1,-1) slice (equivalent to doing nothing)
    void update(uint32_t size);

    friend ostream& operator<<(ostream& os, const Slice& slice);
};

Slice::Slice(int32_t lower, int32_t upper, int32_t stride) {
       points[0] = lower;
       points[1] = upper;
       points[2] = stride;
}

Slice::Slice(string description) {
    char delim = 0;
    if (description.find(':') != string::npos)
        delim = ':';
    if (description.find('/') != string::npos) {
        if (delim != 0)
            display_error("Slicing argument \""+description+"\" not understood");
        delim = '/';
        reduction = Box;
    }
    if (description.find('l') != string::npos) {
        if (delim != 0)
            display_error("Slicing argument \""+description+"\" not understood");
        delim = 'l';
        reduction = Lanczos;
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
        if (points[1] == INT32_MAX) points[1] = -1;
    }
    else if (points[2] > 0) {
        if (points[0] == INT32_MAX) points[0] = 0;
        if (points[1] == INT32_MAX) points[1] = -1;
    }
    else
        display_error("Slicing argument \""+description+"\" not understood");
    assert(points[2] != 0);
}

const int32_t Slice::get_size() {
    return ceil((points[1]-points[0])/double(points[2]));
}

const bool Slice::is_standard() {
    return points[0] == 0 and (points[1] == -1 or points[1] == max_upper) and points[2] == 1;
}

void Slice::update(uint32_t size) {
    if (points[0] == -1)
        points[0] = size-1;
    else if (points[2] > 0 and points[1] == -1)
        points[1] = size;
    max_upper = size;
}

ostream& operator<<(ostream& os, const Slice& slice)
{
    char delim = ':';
    if (slice.reduction == Box) delim = '/';
    else if (slice.reduction == Lanczos) delim = 'l';
    os << slice.points[0] << delim << slice.points[1] << delim << slice.points[2];
    return os;
}

#endif // SLICE_HPP
