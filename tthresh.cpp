#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <iomanip>
#include <stdlib.h>
#include <typeinfo>
#include "encode.cpp"
#include "decode.cpp"
#include "tthresh.hpp"
#include "compress.cpp"
#include "decompress.cpp"

using namespace std;

void print_usage() {
    cout << endl;
    cout << "Usage: tthresh <options>" << endl;
    cout << endl;
    cout << "\t-h: print this usage information and exit" << endl;
    cout << "\t-i <input file>: input 3D volume (string). Either -i or -o (or both) must be specified" << endl;
    cout << "\t-z <compressed file>: name for the compressed result (string)" << endl;
    cout << "\t-o <output file>: if specified, the compressed file (-z) will be decompressed to this name (string)" << endl;
    cout << "\t-v: verbose mode; prints main algorithm steps" << endl;
    cout << "\t-d: print debug information" << endl;
    cout << endl;
    cout << "Compression parameters (needed if -i):" << endl;
    cout << endl;
    cout << "\t-t <type>: input type (can be \"uchar\", \"int\", \"float\" or \"double\")" << endl;
    cout << "\t-s <x> <y> <z>: the volume size (3 integers)" << endl;
    cout << "\t-e | -r | -p <target>: target accuracy (relative error, RMSE or PSNR, respectively)" << endl;
    cout << endl;
}

void display_error(string msg) {
    cout << endl;
    cout << "Error: " << msg << endl;
    cout << "Run \"tthresh -h\" for usage information" << endl;
    cout << endl;
    exit(1);
}

int main(int argc, char* argv[]) {

    /************************************/
    // Check and process all arguments...
    /************************************/

    Mode mode = none_mode;
    Target target = eps;
    double target_value = -1;
    int s[3] = {-1, -1, -1};
    int size_index;
    bool input_flag = false, compressed_flag = false, output_flag = false, io_type_flag = false,
            sizes_flag = false, target_flag = false, verbose_flag = false, debug_flag = false;
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
                case 'z':
                    if (compressed_flag)
                        display_error("Flag -z already set");
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
                    size_index = 0;
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
        }
        else if (mode == input_mode) {
            input_file = arg;
            mode = none_mode;
        }
        else if (mode == compressed_mode) {
            compressed_file = arg;
            mode = none_mode;
        }
        else if (mode == output_mode) {
            output_file = arg;
            mode = none_mode;
        }
        else if (mode == io_type_mode) {
            io_type = arg;
            mode = none_mode;
        }
        else if (mode == sizes_mode) {
            stringstream ss(arg);
            ss >> s[size_index];
            if (s[size_index] <= 0)
                display_error("Wrong size argument");
            size_index++;
            if (size_index == 3) {
                mode = none_mode;
            }
        }
        else if (mode == target_mode) {
            stringstream ss(arg);
            ss >> target_value;
            mode = none_mode;
        }
    }

    if (input_flag) {
        if (input_file == "")
            display_error("Specify a valid input file name");
        if (!io_type_flag)
            display_error("Specify an IO type (-t)");
        if (!sizes_flag or target_value <= 0)
            display_error("Specify both volume sizes (-s) and accuracy target (-e, -r, or -p)");
        if (size_index < 3)
            display_error("Specify 3 integer sizes after -s");
    }

    if (!compressed_flag or compressed_file == "")
        display_error("Specify a file name for the compressed data set (-z)");

    if (output_flag) {
        if (output_file == "")
            display_error("Specify a valid output file name");
        if (!input_flag and (io_type_flag or sizes_flag or target_value > 1))
            display_error("Decompression mode only accepts the input (-z) and output file (-o) flags");
    }

    if (!input_flag and !output_flag)
        display_error("Specify at least one of the flags -i and -o");

    /***************************/
    // The real work starts here
    /***************************/

    double* data = NULL;
    if (input_flag) {
        data = compress(input_file,compressed_file,io_type,s,target,target_value,verbose_flag,debug_flag);
    }
    if (output_flag) {
        decompress(compressed_file,output_file,data,verbose_flag,debug_flag);
    }
    delete data;

    return 0;
}
