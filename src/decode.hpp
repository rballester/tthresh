#ifndef __DECODE_HPP__
#define __DECODE_HPP__

#include <iterator>
#include <unordered_map>

using namespace std;

void decode(ind_t bytes_to_read, vector < char >&mask) {

    /*********************************************/
    // Read dictionary, encoding size and encoding
    /*********************************************/

    char *buffer = new char[bytes_to_read];
    read_zlib_stream(reinterpret_cast < unsigned char *>(buffer), bytes_to_read * sizeof(char));
    unsigned int dict_size = reinterpret_cast < unsigned int *>(buffer)[0];
    unsigned int *key_array = reinterpret_cast < unsigned int *>(buffer) + 1;
    unsigned int *code_array = reinterpret_cast < unsigned int *>(buffer) + 1 + dict_size;
    ind_t n_bits = reinterpret_cast < unsigned int *>(buffer)[1 + 2 * dict_size];

    unordered_map< unsigned long int, unsigned long int> tm[28];
    for (unsigned int i = 0; i < dict_size; ++i) {
        unsigned int code_len = code_array[i] >> 27;
        unsigned int encoding = code_array[i] & 0x07ffffff;
        tm[code_len][encoding] = key_array[i];
    }

    /*****************************************************/
    // Read translation and decode each symbol on the spot
    /*****************************************************/

    long int read_bits = 0;
    int this_code = 0;
    unsigned char current_bit = 0;
    unsigned char write_byte = 0;
    int write_bit = 7;
    int this_code_len = 0;
    unordered_map< unsigned long int, unsigned long int>::iterator it;
    for (int i = (1+2*dict_size)*sizeof(int) + sizeof(ind_t); i < bytes_to_read; ++i) {
        for (int j = 7; j >= 0; --j) {
            this_code += (buffer[i] >> j)&1;
            this_code_len++;

            it = tm[this_code_len].find(this_code); // See if this corresponds to a symbol
            if (it != tm[this_code_len].end()) {
                // RLE decoding: put as many bits as the decoded integer indicates. TODO: pass the counters instead!
                for (unsigned int k = 0; k < it->second; ++k) {
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
            read_bits++;
            if (read_bits == n_bits)
                break;
        }
    }
    if (write_bit< 7)
        mask.push_back(write_byte);
    delete[] buffer;
}

#endif // DECODE_HPP
