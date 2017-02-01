#ifndef __ENCODE_HPP__
#define __ENCODE_HPP__

// Parts of this code were taken from https://rosettacode.org/wiki/Huffman_coding
// (released under the GNU Free Documentation License 1.2, http://www.gnu.org/licenses/fdl-1.2.html)
// which is a C++ Huffman encoding implementation.

#include <iostream>
#include <fstream>
#include <queue>
#include <map>
#include <iterator>
#include <algorithm>
#include <cstring>
#include "zlib_io.hpp"

using namespace std;

typedef std::vector<bool> HuffCode;
typedef std::map<size_t, uint64_t> HuffCodeMap;

class INode {
public:
    const size_t
    f;

    virtual ~ INode() {
    } protected:
    INode(size_t f):f(f) {
    }
};

class InternalNode:public INode {
public:
    INode * const
    left;
    INode *const
    right;

    InternalNode(INode * c0, INode * c1):INode(c0->f + c1->f), left(c0), right(c1) {
    } ~InternalNode() {
        delete left;
        delete right;
    }
};

class LeafNode:public INode {
public:
    const size_t
    c;

    LeafNode(size_t f, size_t c):INode(f), c(c) {
    }};

struct NodeCmp {
    bool operator  () (const INode * lhs, const INode * rhs) const {
        return lhs->f > rhs->f;
    }};

INode *BuildTree(std::map < size_t, uint32_t >&frequencies)
{
    std::priority_queue < INode *, std::vector < INode * >, NodeCmp > trees;

    for (std::map<size_t, uint32_t>::iterator it = frequencies.begin(); it != frequencies.end(); ++it) {
        trees.push(new LeafNode(it->second, (uint32_t) it->first));
    }
    while (trees.size() > 1) {
        INode *childR = trees.top();
        trees.pop();

        INode *childL = trees.top();
        trees.pop();

        INode *parent = new InternalNode(childR, childL);
        trees.push(parent);
    }
    return trees.top();
}

void GenerateCodes(const INode * node, const HuffCode & prefix, HuffCodeMap & outCodes, std::map<size_t, uint8_t>& code_lens)
{
    if (const LeafNode * lf = dynamic_cast < const LeafNode * >(node)) {
        uint64_t binary = 0;
        if (prefix.size() > sizeof(binary)*8) {
            cout << "Error: encoding too large" << endl;
            exit(1);
        }
        for (int i = prefix.size()-1; i >= 0; --i)
            binary |= prefix[prefix.size()-1-i] << i;
        outCodes[lf->c] = binary;
        code_lens[lf->c] = prefix.size();
    } else if (const InternalNode * in = dynamic_cast < const InternalNode * >(node)) {
        HuffCode leftPrefix = prefix;
        leftPrefix.push_back(false);
        GenerateCodes(in->left, leftPrefix, outCodes, code_lens);

        HuffCode rightPrefix = prefix;
        rightPrefix.push_back(true);
        GenerateCodes(in->right, rightPrefix, outCodes, code_lens);
    }
}

void encode(vector<size_t>& rle) {

    std::map<size_t, uint8_t> code_lens;
    std::map<size_t, uint32_t> frequencies;
    for (size_t i = 0; i < rle.size(); ++i)
        ++frequencies[rle[i]];

    /*******************************/
    // Create the Huffman dictionary
    /*******************************/

    INode *root = BuildTree(frequencies);

    HuffCodeMap codes;
    GenerateCodes(root, HuffCode(), codes, code_lens);
    delete root;

    if (frequencies.size() == 1) { // If there's only one symbol, we still need one bit for it
        codes[frequencies.begin()->first] = 0;
        code_lens[frequencies.begin()->first] = 1;
    }

    /**************************************************************************/
    // Save the dictionary + encoding information:
    // (1) Number of key/code pairs (uint32_t)
    // (2) For each key/code pair:
    //         length(key) (uint8_t)
    //         key (sequence of bits)
    //         length(code) (uint8_t)
    //         code (sequence of bits)
    // (3) Number N of keys to code (uint64_t)
    // (4) N codes (sequence of bits)
    /**************************************************************************/

    zlib_open_wbit();

    // Number of key/code pairs
    uint32_t dict_size = codes.size();
    zlib_write_bits(dict_size, sizeof(int)*8);

    // Key/code pairs
    for (HuffCodeMap::const_iterator it = codes.begin(); it != codes.end(); ++it) {

        uint64_t key = it->first; // TODO change key to ind_t

        // First, the key's length
        uint8_t key_len = 0;
        uint64_t key_copy = key;
        while (key_copy) {
            key_copy >>= 1;
            key_len++;
        }
        key_len = max(1, key_len); // A 0 still requires 1 bit for us
        zlib_write_bits(key_len, 6);

        // Next, the key itself
        zlib_write_bits(key, key_len);

        // Now, the code's length
        zlib_write_bits(code_lens[key], 6);

        // Finally, the code itself
        zlib_write_bits(it->second, code_lens[key]);
    }

    // Number N of symbols to code
    uint64_t n_symbols = rle.size();
    zlib_write_bits(n_symbols, sizeof(n_symbols)*8);

    // Now the N codes
    for (size_t i = 0; i < rle.size(); ++i)
        zlib_write_bits(codes[rle[i]], code_lens[rle[i]]);

    zlib_close_wbit();
}

#endif // ENCODE_HPP
