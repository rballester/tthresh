/*
 * Copyright (c) 2016-2017, Rafael Ballester-Ripoll
 *                          (Visualization and MultiMedia Lab, University of Zurich),
 *                          rballester@ifi.uzh.ch
 *
 * Licensed under LGPLv3.0 (https://github.com/rballester/tthresh/LICENSE)
 */

#ifndef __DECODE_HPP__
#define __DECODE_HPP__

#include <iterator>
#include <unordered_map>

using namespace std;

void decode(vector<size_t>& rle) {

    zlib_open_rbit();

    // Number of key/code pairs
    uint32_t dict_size = zlib_read_bits(sizeof(uint32_t)*8);

    // Key/code pairs
    vector< unordered_map<uint64_t, size_t> > tm;

    for (uint32_t i = 0; i < dict_size; ++i) {

        // First, the key's length
        uint8_t key_len = zlib_read_bits(6);

        // Next, the key itself
        uint64_t key = zlib_read_bits(key_len);

        // Now, the code's length
        uint8_t code_len = zlib_read_bits(6);

        // Finally, the code itself
        uint64_t code = zlib_read_bits(code_len);

        // Store the key/code pair in the appropriate table
        if (tm.size() <= code_len)
            tm.resize(code_len+1, unordered_map<uint64_t, uint64_t>());
        tm[code_len][code] = key;
    }

    // Number of symbols to translate back
    uint64_t n_symbols = zlib_read_bits(sizeof(n_symbols)*8);

    // Decoding
    uint64_t code = 0;
    uint8_t code_len = 0;
    unordered_map<uint64_t, uint64_t>::iterator it;
    for (uint64_t i = 0; i < n_symbols; ++i) {
        it = tm[code_len].find(code); // See if this corresponds to a symbol
        while (it == tm[code_len].end()) {
            code <<= 1;
            code += zlib_read_bits(1);
            code_len++;
            if (tm.size() < code_len+1U) {
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
