/*
 * Copyright (c) 2016-2017, Rafael Ballester-Ripoll
 *                          (Visualization and MultiMedia Lab, University of Zurich),
 *                          rballester@ifi.uzh.ch
 *
 * Licensed under the LGPLv3.0 (https://github.com/rballester/tthresh/blob/master/LICENSE)
 */

/*
 The arithmetic loop (last part of this function) was provided by
https://marknelson.us/posts/2014/10/19/data-compression-with-arithmetic-coding.html
under the following MIT License:

Copyright (c) 2014 Mark Thomas Nelson

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef __ENCODE_HPP__
#define __ENCODE_HPP__

#include <map>
#include <iterator>
#include "io.hpp"

using namespace std;

constexpr uint8_t CODE_VALUE_BITS = 32;
constexpr uint64_t MAX_CODE = (((uint64_t)1) << CODE_VALUE_BITS)-1;
constexpr uint64_t ONE_FOURTH = (MAX_CODE + ((uint64_t)1))/4;
constexpr uint64_t ONE_HALF = ONE_FOURTH*2;
constexpr uint64_t THREE_FOURTHS = ONE_FOURTH*3;

uint64_t encoding_bits;

inline void put_bit(char bit) {
    write_bits(bit, 1);
    encoding_bits++;
}

inline void put_bit_plus_pending(bool bit, int& pending_bits)
{
  put_bit(bit);
  for ( int i = 0 ; i < pending_bits ; i++ )
    put_bit(!bit);
  pending_bits = 0;
}

uint64_t encode(vector<uint64_t>& rle) {

    // Build table of frequencies/probability intervals
    // key -> (count, lower bound)
    std::map<uint64_t, pair<uint64_t, uint64_t> > frequencies;
    for (uint64_t i = 0; i < rle.size(); ++i)
        ++frequencies[rle[i]].first;
    uint64_t count = 0;
    for (std::map<uint64_t, pair<uint64_t, uint64_t> >::iterator it = frequencies.begin(); it != frequencies.end(); ++it) {
        (it->second).second = count;
        count += (it->second).first;
    }

    encoding_bits = 0;

//    open_wbit();

    //*********
    //********* Write frequencies
    //*********

    // Number of key/frequency pairs
    uint64_t dict_size = frequencies.size();
    write_bits(dict_size, sizeof(uint64_t)*8);
//    cerr << "dict_size: " << dict_size << endl;
    encoding_bits += sizeof(uint64_t)*8;

    // Key/code pairs
    for (std::map<uint64_t, pair<uint64_t, uint64_t> >::iterator it = frequencies.begin(); it != frequencies.end(); ++it) {

        uint64_t key = it->first;
        uint64_t freq = (it->second).first;

        // First, the key's length
        uint8_t key_len = 0;
        uint64_t key_copy = key;
        while (key_copy) {
            key_copy >>= 1;
            key_len++;
        }
        key_len = max(1, key_len); // A 0 still requires 1 bit for us
        write_bits(key_len, 6);

        // Next, the key itself
        write_bits(key, key_len);

        // Now, the frequency's length
        uint8_t freq_len = 0;
        uint64_t freq_copy = freq;
        while (freq_copy) {
            freq_copy >>= 1;
            freq_len++;
        }
        freq_len = max(1, freq_len); // A 0 still requires 1 bit for us
        write_bits(freq_len, 6);

        // Finally, the frequency itself
        write_bits(freq, freq_len);

        encoding_bits += 6 + key_len + 6 + freq_len;
    }

    // Number N of symbols to code
    uint64_t n_symbols = rle.size();
    write_bits(n_symbols, sizeof(uint64_t)*8);
    encoding_bits += sizeof(uint64_t)*8;

    //*********
    //********* Write the encoding
    //*********

    int pending_bits = 0;
    uint64_t low = 0;
    uint64_t high = MAX_CODE;

    uint64_t rle_pos = 0;
    for ( ; ; ) {
      uint64_t c = rle[rle_pos];
      rle_pos++;

      uint64_t phigh = frequencies[c].second + frequencies[c].first;
      uint64_t plow = frequencies[c].second;

      uint64_t range = high - low + 1;
      high = low + (range * phigh / n_symbols) - 1;
      low = low + (range * plow / n_symbols);

      for ( ; ; ) {
        if ( high < ONE_HALF )
          put_bit_plus_pending(0, pending_bits);
        else if ( low >= ONE_HALF )
          put_bit_plus_pending(1, pending_bits);
        else if ( low >= ONE_FOURTH && high < THREE_FOURTHS ) {
          pending_bits++;
          low -= ONE_FOURTH;
          high -= ONE_FOURTH;
        } else
          break;
        high <<= 1;
        high++;
        low <<= 1;
        high &= MAX_CODE;
        low &= MAX_CODE;
      }

      if (rle_pos == n_symbols)
        break;
    }
    pending_bits++;
    if ( low < ONE_FOURTH )
      put_bit_plus_pending(0, pending_bits);
    else
      put_bit_plus_pending(1, pending_bits);

    write_bits(0UL, CODE_VALUE_BITS-2); // Trailing zeros
    encoding_bits += CODE_VALUE_BITS-2;

//    close_wbit();

    return encoding_bits;
}

#endif // ENCODE_HPP
