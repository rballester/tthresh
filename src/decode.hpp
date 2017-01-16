#ifndef __DECODE_HPP__
#define __DECODE_HPP__

#include <iostream>
#include <fstream>
#include <queue>
#include <map>
#include <climits>
#include <iterator>
#include <algorithm>
#include <unordered_map>

using namespace std;

void decode(ifstream & input_stream, unsigned int bytes_to_read, vector < char >&mask)
{

    /*********************************************/
    // Read dictionary, encoding size and encoding
    /*********************************************/

    char *buffer = new char[bytes_to_read];
    input_stream.read(reinterpret_cast < char *>(buffer), bytes_to_read * sizeof(char));
    unsigned int dict_size = reinterpret_cast < unsigned int *>(buffer)[0];
    unsigned int *key_array = reinterpret_cast < unsigned int *>(buffer) + 1;
    unsigned int *code_array = reinterpret_cast < unsigned int *>(buffer) + 1 + dict_size;
    unsigned int n_bits = reinterpret_cast < unsigned int *>(buffer)[1 + 2 * dict_size];

    unordered_map< unsigned int, unsigned int> tm[28];
    for (int i = 0; i < dict_size; ++i) {
        unsigned int code_len = code_array[i] >> 27;
        unsigned int encoding = code_array[i] & 0x03ffffff;
        tm[code_len][encoding] = key_array[i];
    }

    /*****************************************************/
    // Read translation and decode each symbol on the spot
    /*****************************************************/

    int read_bits = 0;
    int counter = 0;
    int this_code = 0;
    unsigned char current_bit = 0;
    unsigned char write_byte = 0;
    int write_bit = 7;
    int this_code_len = 0;
    unordered_map< unsigned int, unsigned int>::iterator it;
    for (int i = (1+2*dict_size+1)*sizeof(int); i < bytes_to_read; ++i) {
        char c = buffer[i];
        for (int j = 0; j < 8; ++j) {
            if (read_bits == n_bits)
                break;
            char bit = (c >> (7 - j))&1;
            this_code += bit;
            this_code_len++;
            counter++;
            read_bits++;

            it = tm[this_code_len].find(this_code); // See if this corresponds to a symbol
            if (it != tm[this_code_len].end()) {
                // RLE decoding: put as many bits as the decoded integer indicates
                for (int k = 0; k < it->second; ++k) {
                    write_byte |= current_bit << write_bit;
                    write_bit--;
                    if (write_bit < 0) {
                        mask.push_back((write_byte));
                        write_byte = 0;
                        write_bit = 7;
                    }
                }
                current_bit = !current_bit;
                this_code = 0;
                this_code_len = 0;
            }
            this_code <<= 1;
        }
    }
    if (write_bit< 7)
        mask.push_back(write_byte);
    delete[] buffer;
}

#endif // DECODE_HPP
