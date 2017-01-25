#ifndef __DECODE_HPP__
#define __DECODE_HPP__

#include <iterator>
#include <unordered_map>

using namespace std;

void decode(vector<ind_t>& counters) {

    zlib_open_rbit();

    // Number of key/code pairs
    int dict_size = 0;
    for (int wbit = 31; wbit >= 0; --wbit)
        dict_size |= zlib_read_bits(1) << wbit;

    // Key/code pairs
    vector< unordered_map<unsigned long int, unsigned long int> > tm;

    for (int i = 0; i < dict_size; ++i) {

        // First, the key's length
        unsigned int key_len = 0;
        for (int wbit = 7; wbit >= 0; --wbit)
            key_len |= zlib_read_bits(1) << wbit;

        // Next, the key itself
        unsigned long int key = 0;
        for (int wbit = key_len-1; wbit >= 0; --wbit)
            key |= zlib_read_bits(1) << wbit;

        // Now, the code's length
        unsigned int code_len = 0;
        for (int wbit = 7; wbit >= 0; --wbit)
            code_len |= zlib_read_bits(1) << wbit;

        // Finally, the code itself
        unsigned long int code = 0;
        for (int wbit = code_len-1; wbit >= 0; --wbit)
            code |= zlib_read_bits(1) << wbit;

        // Store the key/code pair in the appropriate table
        if (tm.size() < code_len+1)
            tm.resize(code_len+1, unordered_map<unsigned long int, unsigned long int>());
        tm[code_len][code] = key;
    }

    // Number of symbols to translate back
    unsigned long int n_symbols = 0;
    for (int wbit = 63; wbit >= 0; --wbit)
        n_symbols |= zlib_read_bits(1) << wbit;

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
        counters.push_back(it->second);
        code = 0;
        code_len = 0;
    }
}

#endif // DECODE_HPP
