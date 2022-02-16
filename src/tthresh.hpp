/*
 * Copyright (c) 2016-2017, Rafael Ballester-Ripoll
 *                          (Visualization and MultiMedia Lab, University of Zurich),
 *                          rballester@ifi.uzh.ch
 *
 * Licensed under the LGPLv3.0 (https://github.com/rballester/tthresh/blob/master/LICENSE)
 */

#ifndef __TTHRESH_HPP__
#define __TTHRESH_HPP__

#include <vector>
#include <stack>
#include <chrono>
#include <string>
using namespace std;
using namespace std::chrono;

#ifdef WIN32
#include <iso646.h> //for 'and' 'or' ...
#endif

// Size (in bytes) for all I/O buffers
#define CHUNK (1<<18)

// Rows whose squared norm is larger than this will be cropped away
#define AUTOCROP_THRESHOLD (1e-10)

// Compression parameters
enum Mode { none_mode, input_mode, compressed_mode, output_mode, io_type_mode, sizes_mode, target_mode, skip_bytes_mode };
enum Target { eps, rmse, psnr };

// Tensor dimensionality, ranks and sizes. They are only set, never modified
uint8_t n;
vector<uint32_t> r;
vector<size_t> rprod;
vector<uint32_t> s, snew;
vector<size_t> sprod, snewprod;

void cumulative_products(vector<uint32_t>& in, vector<size_t>& out) {
    uint8_t n = s.size();
    out = vector<size_t> (n+1); // Cumulative size products. The i-th element contains s[0]*...*s[i-1]
    out[0] = 1;
    for (uint8_t i = 0; i < n; ++i)
        out[i+1] = out[i]*in[i];
}

stack<high_resolution_clock::time_point> times;

void start_timer(string message) {
    cout << message << flush;
    times.push(std::chrono::high_resolution_clock::now());
}

void stop_timer() {
    if (times.size() < 1) {
        cout << "Error: timer not set" << endl;
        exit(1);
    }
    auto elapsed = std::chrono::high_resolution_clock::now() - times.top();
    times.pop();
    cout << "Elapsed time: " << std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count()/1000. << "ms" << endl << flush;
}

int64_t min(int64_t a, int64_t b) {
    return (a < b) ? a : b;
}

int64_t max(int64_t a, int64_t b) {
    return (a > b) ? a : b;
}

void print_usage() {
    cout << endl;
    cout << "tthresh: a multidimensional data compressor" << endl;
    cout << endl;
    cout << "Usage: tthresh <options>" << endl;
    cout << endl;

    cout << "\t-h                        - Print this usage information and exit" << endl;
    cout << "\t-i <input file>           - Input dataset in raw format (string). Either -i or -o (or both) must be specified" << endl;
    cout << "\t-c <compressed file>      - Name for the compressed result (string)" << endl;
    cout << "\t-o [cutout] <output file> - If given, the compressed file (-c) will be decompressed to this file name (string)" << endl;
    cout << "\t                            \tIf [cutout] is given, a downsampled tensor will be reconstructed." << endl;
    cout << "\t                            \tThis is specified as in NumPy's slicing notation: start:stop:step" << endl;
    cout << "\t                            \tUse the separator ':' for downsampling; '/' for box filter decimation; 'l' for separable Lanczos-2" << endl;
    cout << "\t-v                        - Verbose mode; prints main algorithm steps" << endl;
    cout << "\t-d                        - Print debug information" << endl;
    cout << endl;

    cout << "Compression parameters (needed with -i):" << endl;
    cout << endl;
    cout << "\t-t <type>                 - Input type (can be \"uchar\", \"ushort\", \"int\", \"float\" or \"double\")" << endl;
    cout << "\t-s <x> <y> <z> [...]      - Data sizes (3 or more integers); assumed to be in FORTRAN order (first dimension varies fastest)" << endl;
    cout << "\t-e | -r | -p <target>     - Target accuracy (real); relative error, RMSE or PSNR, respectively" << endl;
    cout << endl;

    cout << "Optional compression parameters:" << endl;
    cout << endl;
    cout << "\t-k <n>                    - Skip n leading bytes, for e.g. removing a header (integer)" << endl;
    cout << endl;

    cout << "Optional decompression parameters:" << endl;
    cout << endl;
    cout << "\t-a                        - Autocrop: restrict reconstruction to its bounding box (resulting sizes will be printed)" << endl;
    cout << endl;

    cout << "Examples:" << endl;
    cout << endl;
    cout << "\ttthresh -i data -t uchar -s 256^3 -p 40 -c data.tthresh -o data.decompressed - Compress and decompress a volume of unsigned chars with PSNR = 40" << endl;
    cout << "\ttthresh -i data -k 16 -t float -s 128 256 64 100 -p 40 -c data.tthresh       - Compress a 4D tensor (e.g. time-dependent volume), skipping the first 16 bytes (header)" << endl;
    cout << "\ttthresh -i data -t double -s 2^20 -p 40 -c data.tthresh                      - Compress a 1D signal with ~1M points by reshaping it into a 2^20 tensor" << endl;
    cout << "\ttthresh -c data.tthresh -o ::2 ::2 ::2 data.decompressed                     - Decompress only the even tensor indices (final size: x8 smaller)" << endl;
    cout << "\ttthresh -c data.tthresh -o :: :: 0 data.decompressed                         - Decompress the first z-slice" << endl;
    cout << "\ttthresh -c data.tthresh -o ll4 ::-1 ::-1 data.decompressed                   - Lanczos downsample (x4) along the x-axis, invert the data along the other two axes" << endl;
    cout << endl;
}

void display_error(string msg) {
    cout << endl;
    cout << "Error: " << msg << endl;
    cout << "Run \"tthresh -h\" for usage information" << endl;
    cout << endl;
    exit(1);
}

bool is_number(string & s) {
    string::const_iterator it = s.begin();
    while (it != s.end() and (isdigit(*it) or *it == '.'))
        ++it;
    return !s.empty() and it == s.end();
}

#endif // TTHRESH_HPP
