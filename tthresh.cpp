#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <iomanip>
#include <stdlib.h>
#include <typeinfo>
#include "encode.hpp"
#include "decode.hpp"
#include "tthresh.hpp"
#include "compress.hpp"
#include "decompress.hpp"

using namespace std;

void print_usage()
{
    cout << endl;
    cout << "tthresh: a compressor for 3D and 4D datasets" << endl;
    cout << "Usage: tthresh <options>" << endl;
    cout << endl;
    cout << "\t-h: print this usage information and exit" << endl;
    cout << "\t-i <input file>: input 3D/4D data (string). Either -i or -o (or both) must be specified" << endl;
    cout << "\t-c <compressed file>: name for the compressed result (string)" << endl;
    cout << "\t-o <output file>: if specified, the compressed file (-c) will be decompressed to this file name (string)" << endl;
    cout << "\t-v: verbose mode; prints main algorithm steps" << endl;
    cout << "\t-d: print debug information" << endl;
    cout << endl;
    cout << "Compression parameters (needed with -i):" << endl;
    cout << endl;
    cout << "\t-t <type>: input type (can be \"uchar\", \"ushort\", \"int\", \"float\" or \"double\")" << endl;
    cout << "\t-s <x> <y> <z> [<t>]: the data sizes (3 or 4 integers)" << endl;
    cout << "\t-e | -r | -p <target>: target accuracy (relative error, RMSE or PSNR, respectively)" << endl;
    cout << endl;
    cout << "Optional compression parameters:" << endl;
    cout << endl;
    cout << "\t-k <n>: skip n leading bytes, for e.g. removing a header (integer)" << endl;
    cout << endl;
}

void display_error(string msg)
{
    cout << endl;
    cout << "Error: " << msg << endl;
    cout << "Run \"tthresh -h\" for usage information" << endl;
    cout << endl;
    exit(1);
}

bool is_number(string & s)
{
    string::const_iterator it = s.begin();
    while (it != s.end() and isdigit(*it))
        ++it;
    return !s.empty() and it == s.end();
}

int main(int argc, char *argv[])
{

    /************************************/
    // Check and process all arguments...
    /************************************/

    Mode mode = none_mode;
    Target target = eps;
    double target_value = -1;
    unsigned long int skip_bytes = 0;
    vector < int >s;
    bool input_flag = false, compressed_flag = false, output_flag = false, io_type_flag = false, sizes_flag = false, target_flag = false, skip_bytes_flag = false, verbose_flag = false, debug_flag = false;
    string input_file;
    string compressed_file;
    string output_file;
    string io_type;

    if (argc == 1) {
        print_usage();
        exit(0);
    }

    for (int i = 1; i < argc; ++i) {
        string arg(argv[i]);
        if (arg[0] == '-') {
            mode = none_mode;
        }

        if (mode == none_mode) {
            if (not is_number(arg) and mode == sizes_mode)
                mode = none_mode;
            if (arg[0] != '-')
                display_error("Unrecognized flag \"" + arg + "\"");
            if (arg.size() != 2)
                display_error("Flags must have exactly 1 letter");
            switch (arg[1]) {
            case 'h':
                print_usage();
                exit(0);
            case 'i':
                if (input_flag)
                    display_error("Flag -a already set");
                mode = input_mode;
                input_flag = true;
                continue;
            case 'c':
                if (compressed_flag)
                    display_error("Flag -c already set");
                mode = compressed_mode;
                compressed_flag = true;
                continue;
            case 'o':
                if (output_flag)
                    display_error("Flag -o already set");
                mode = output_mode;
                output_flag = true;
                continue;
            case 't':
                if (io_type_flag)
                    display_error("I/O type already defined");
                mode = io_type_mode;
                io_type_flag = true;
                continue;
            case 's':
                if (sizes_flag)
                    display_error("The sizes (-s) were already set");
                mode = sizes_mode;
                sizes_flag = true;
                continue;
            case 'e':
                if (target_flag)
                    display_error("The target type was already set");
                mode = target_mode;
                target_flag = true;
                target = eps;
                continue;
            case 'r':
                if (target_flag)
                    display_error("The target type was already set");
                mode = target_mode;
                target_flag = true;
                target = rmse;
                continue;
            case 'p':
                if (target_flag)
                    display_error("The target type was already set");
                mode = target_mode;
                target_flag = true;
                target = psnr;
                continue;
            case 'k':
                if (skip_bytes_flag)
                    display_error("Flag -k already set");
                mode = skip_bytes_mode;
                skip_bytes_flag = true;
                continue;
            case 'v':
                if (verbose_flag)
                    display_error("Flag -v already set");
                verbose_flag = true;
                continue;
            case 'd':
                if (debug_flag)
                    display_error("Flag -d already set");
                verbose_flag = true;
                debug_flag = true;
                continue;
            default:
                display_error("Unrecognized flag \"" + arg + "\"");
            }
        } else if (mode == input_mode) {
            input_file = arg;
            mode = none_mode;
        } else if (mode == compressed_mode) {
            compressed_file = arg;
            mode = none_mode;
        } else if (mode == output_mode) {
            output_file = arg;
            mode = none_mode;
        } else if (mode == io_type_mode) {
            io_type = arg;
            mode = none_mode;
        } else if (mode == sizes_mode) {
            stringstream ss(arg);
            int next_size;
            ss >> next_size;
            if (next_size <= 0)
                display_error("Size arguments must be positive integers");
            s.push_back(next_size);
            if (s.size() == 4) {
                mode = none_mode;
            }
        } else if (mode == target_mode) {
            stringstream ss(arg);
            ss >> target_value;
            mode = none_mode;
        } else if (mode == skip_bytes_mode) {
            stringstream ss(arg);
            ss >> skip_bytes;
            mode = none_mode;
        }
    }

    if (input_flag) {
        if (input_file == "")
            display_error("Specify a valid input file name");
        if (!io_type_flag)
            display_error("Specify an IO type (-t)");
        if (!sizes_flag or target_value <= 0)
            display_error("Specify both data sizes (-s) and accuracy target (-e, -r, or -p)");
        if (s.size() < 3)
            display_error("Specify 3 or 4 integer sizes after -s");
    }

    if (!compressed_flag or compressed_file == "")
        display_error("Specify a file name for the compressed data set (-c)");

    if (output_flag) {
        if (output_file == "")
            display_error("Specify a valid output file name");
        if (!input_flag and(io_type_flag or sizes_flag or target_value > 1))
            display_error("Decompression mode only accepts flags -c, -o, -v and -d");
    }

    if (!input_flag and ! output_flag)
        display_error("Specify at least one of the flags -i and -o");

    /***************************/
    // The real work starts here
    /***************************/

    double *data = NULL;
    if (input_flag) {
        data = compress(input_file, compressed_file, io_type, s, target, target_value, skip_bytes, verbose_flag, debug_flag);
    }
    if (output_flag) {
        decompress(compressed_file, output_file, data, verbose_flag, debug_flag);
    }
    delete[](data-skip_bytes);

    return 0;
}
