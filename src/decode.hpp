#ifndef __DECODE_HPP__
#define __DECODE_HPP__

#include <iterator>
#include <unordered_map>

using namespace std;

void decode(vector<ind_t>& rle) {

    zlib_open_rbit();

    // Number of key/code pairs
    int dict_size = zlib_read_bits(sizeof(int)*8);

    // Key/code pairs
    vector< unordered_map<unsigned long int, unsigned long int> > tm;

    for (int i = 0; i < dict_size; ++i) {

        // First, the key's length
        unsigned char key_len = zlib_read_bits(6);

        // Next, the key itself
        unsigned long int key = zlib_read_bits(key_len);

        // Now, the code's length
        unsigned char code_len = zlib_read_bits(6);

        // Finally, the code itself
        unsigned long int code = zlib_read_bits(code_len);

        // Store the key/code pair in the appropriate table
        if (tm.size() <= code_len)
            tm.resize(code_len+1, unordered_map<unsigned long int, unsigned long int>());
        tm[code_len][code] = key;
    }

    // Number of symbols to translate back
    unsigned long int n_symbols = zlib_read_bits(sizeof(n_symbols)*8);

    // Decoding
    unsigned long int code = 0;
    unsigned int code_len = 0;
    unordered_map< unsigned long int, unsigned long int>::iterator it;
    for (unsigned long int i = 0; i < n_symbols; ++i) {
        it = tm[code_len].find(code); // See if this corresponds to a symbol
        while (it == tm[code_len].end()) {
            code <<= 1;
            code += zlib_read_bits(1);
            code_len++;
            if (tm.size() < code_len+1) {
                cout << "Could not translate symbol" << endl;
                exit(1);
            }
            it = tm[code_len].find(code); // See if this corresponds to a symbol
        }
        rle.push_back(it->second);
        code = 0;
        code_len = 0;
    }
}

#endif // DECODE_HPP
