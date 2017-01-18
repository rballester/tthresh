#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <iomanip>
#include <stdlib.h>
#include <typeinfo>
#include "tthresh.hpp"
#include "compress.hpp"
#include "decompress.hpp"

using namespace std;

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
            if (mode == sizes_mode and not is_number(arg))
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
            stringstream ss1(arg);
            int base, exponent = 1;
            string token;
            int n_parts = 0; // Should be 1 (plain number) or 2 at most (base and exponent)
            while(getline(ss1, token, '^')) {
                n_parts++;
                stringstream ss2(token);
                if (n_parts == 1)
                    ss2 >> base;
                else if (n_parts == 2)
                    ss2 >> exponent;
                if (arg[arg.size()-1] == '^' or n_parts > 2 or not is_number(token) or base <= 0 or exponent <= 0)
                    display_error("Unrecognized sizes: must be positive integers");
            }
            for (int dim = 0; dim < exponent; ++dim)
                s.push_back(base);
        } else if (mode == target_mode) {
            if (not is_number(arg))
                display_error("Numeric argument expected");
            stringstream ss(arg);
            ss >> target_value;
            mode = none_mode;
        } else if (mode == skip_bytes_mode) {
            if (not is_number(arg))
                display_error("Numeric argument expected");
            stringstream ss(arg);
            ss >> skip_bytes;
            mode = none_mode;
        }
    }

    if (input_flag) {
        if (input_file == "")
            display_error("Specify a valid input file name");
        if (!io_type_flag)
            display_error("Specify an I/O type (-t)");
        if (!sizes_flag or target_value < 0)
            display_error("Specify both data sizes (-s) and accuracy target (-e, -r, or -p)");
        if (s.size() < 3)
            display_error("Specify 3 or more integer sizes after -s");
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

    if (input_flag and io_type != "uchar" and io_type != "ushort" and io_type != "int" and io_type != "float" and io_type != "double")
        display_error("Unrecognized I/O type \"" + io_type + "\". Supported are: \"uchar\", \"ushort\", \"int\", \"float\", \"double\"");

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
