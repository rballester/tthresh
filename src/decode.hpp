/*
 * Copyright (c) 2016-2017, Rafael Ballester-Ripoll
 *                          (Visualization and MultiMedia Lab, University of Zurich),
 *                          rballester@ifi.uzh.ch
 *
 * Licensed under the LGPLv3.0 (https://github.com/rballester/tthresh/blob/master/LICENSE)
 */

/*
 The arithmetic loop (last part of this function) was provided by
http://marknelson.us/2014/10/19/data-compression-with-arithmetic-coding
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

#ifndef __DECODE_HPP__
#define __DECODE_HPP__

#include <iterator>

using namespace std;

void decode(vector<size_t>& rle) {

//    open_rbit();

    //*********
    //********* Read the frequencies
    //*********

    // Number of key/frequency pairs
    uint64_t dict_size = read_bits(sizeof(uint64_t)*8);

    // Pairs of key -> probability's lower bound
    std::map<uint64_t, uint64_t> lowers;

    uint64_t count = 0;
    for (uint64_t i = 0; i < dict_size; ++i) {

        // First, the key's length
        uint8_t key_len = read_bits(6);

        // Next, the key itself
        uint64_t key = read_bits(key_len);

        // Now, the frequency's length
        uint8_t freq_len = read_bits(6);

        // Finally, the frequency itself
        uint64_t freq = read_bits(freq_len);
        lowers[count] = key;
        count += freq;
    }

    // Number of symbols to translate back
    uint64_t n_symbols = read_bits(sizeof(uint64_t)*8);

    lowers[n_symbols] = 0; // The last upper bound

    //*********
    //********* Read the encoding
    //*********

    uint64_t high = MAX_CODE;
    uint64_t low = 0;
    uint64_t value = 0;
    value = read_bits(CODE_VALUE_BITS);

    for ( ; ; ) {

      uint64_t range = high - low + 1;
      uint64_t scaled_value =  ((value - low + 1) * n_symbols - 1 ) / range;

      std::map<uint64_t, uint64_t>::iterator it = lowers.upper_bound(scaled_value);
      uint64_t phigh = it->first;
      it--;
      uint64_t c = it->second;
      rle.push_back(c);
      uint64_t plow = it->first;

      high = low + (range*phigh)/n_symbols -1;
      low = low + (range*plow)/n_symbols;

      for( ; ; ) {

        if ( high < ONE_HALF ) {
          //do nothing, bit is a zero
        } else if ( low >= ONE_HALF ) {
          value -= ONE_HALF;  //subtract one half from all three code values
          low -= ONE_HALF;
          high -= ONE_HALF;
        } else if ( low >= ONE_FOURTH && high < THREE_FOURTHS ) {
          value -= ONE_FOURTH;
          low -= ONE_FOURTH;
              high -= ONE_FOURTH;
            } else
                break;
            low <<= 1;
            high <<= 1;
            high++;
            value <<= 1;
            value += read_bits(1) ? 1 : 0;
        }

        if (rle.size() == n_symbols)
            break;
    }
}

#endif // DECODE_HPP
