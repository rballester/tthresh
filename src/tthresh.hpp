#ifndef __TTHRESH_HPP__
#define __TTHRESH_HPP__

#include <vector>
using namespace std;

enum Mode { none_mode, input_mode, compressed_mode, output_mode, io_type_mode, sizes_mode, target_mode, skip_bytes_mode };
enum Target { eps, rmse, psnr };

typedef long int ind_t; // Used to index bytes and bits

vector<ind_t> sprod;

void cumulative_size_products(vector<int>& s, char n) {
    // Compute the cumulative products (useful later on for index computations)
    sprod = vector<ind_t> (n+1); // Cumulative size products. The i-th element contains s[0]*...*s[i-1]
    sprod[0] = 1;
    for (char dim = 0; dim < n; ++dim)
        sprod[dim+1] = sprod[dim]*s[dim];
}

void print_usage() {
    cout << endl;
    cout << "tthresh: a multidimensional data compressor" << endl;
    cout << "Usage: tthresh <options>" << endl;
    cout << endl;

    cout << "\t-h                    - Print this usage information and exit" << endl;
    cout << "\t-i <input file>       - Input dataset in raw format (string). Either -i or -o (or both) must be specified" << endl;
    cout << "\t-c <compressed file>  - Name for the compressed result (string)" << endl;
    cout << "\t-o <output file>      - If specified, the compressed file (-c) will be decompressed to this file name (string)" << endl;
    cout << "\t-v                    - Verbose mode; prints main algorithm steps" << endl;
    cout << "\t-d                    - Print debug information" << endl;
    cout << endl;

    cout << "Compression parameters (needed with -i):" << endl;
    cout << endl;
    cout << "\t-t <type>             - Input type (can be \"uchar\", \"ushort\", \"int\", \"float\" or \"double\")" << endl;
    cout << "\t-s <x> <y> <z> [...]  - Data sizes (3 or more integers)" << endl;
    cout << "\t-e | -r | -p <target> - Target accuracy (real); relative error, RMSE or PSNR, respectively" << endl;
    cout << endl;

    cout << "Optional compression parameters:" << endl;
    cout << endl;
    cout << "\t-k <n>                - Skip n leading bytes, for e.g. removing a header (integer)" << endl;
    cout << endl;

    cout << "Examples:" << endl;
    cout << endl;
    cout << "\ttthresh -i data -t uchar -s 256^3 -p 40 -c data.tthresh -o data.decompressed - Compress and decompress a volume of unsigned chars with PSNR = 40" << endl;
    cout << "\ttthresh -i data -k 16 -t float -s 128 256 64 100 -p 40 -c data.tthresh       - Compress a 4D tensor (e.g. time-dependent volume), skipping the first 16 bytes (header)" << endl;
    cout << "\ttthresh -i data -t double -s 2^20 -p 40 -c data.tthresh                      - Compress a 1D signal with ~1M points by reshaping it into a 2^20 tensor" << endl;
    cout << "\ttthresh -c data.tthresh -d data.decompressed                                 - Decompress a dataset" << endl;
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
