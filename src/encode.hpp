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

typedef std::vector < bool > HuffCode;
typedef std::map < unsigned long int, unsigned long int > HuffCodeMap;

class INode {
public:
    const unsigned long int
    f;

    virtual ~ INode() {
    } protected:
    INode(unsigned long int f):f(f) {
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
    const unsigned long int
    c;

    LeafNode(unsigned long int f, unsigned long int c):INode(f), c(c) {
    }};

struct NodeCmp {
    bool operator  () (const INode * lhs, const INode * rhs) const {
        return lhs->f > rhs->f;
    }};

INode *BuildTree(std::map < unsigned long int, int >&frequencies)
{
    std::priority_queue < INode *, std::vector < INode * >, NodeCmp > trees;

    for (std::map < unsigned long int, int >::iterator it = frequencies.begin(); it != frequencies.end(); ++it) {
        trees.push(new LeafNode(it->second, (int) it->first));
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

void GenerateCodes(const INode * node, const HuffCode & prefix, HuffCodeMap & outCodes, std::map<unsigned long int, unsigned char>& code_lens)
{
    if (const LeafNode * lf = dynamic_cast < const LeafNode * >(node)) {
        unsigned long int binary = 0;
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

void encode(vector<unsigned long int>& counters) {

    std::map<unsigned long int, unsigned char> code_lens;
    std::map<unsigned long int, int >frequencies;
    for (unsigned long int i = 0; i < counters.size(); ++i)
        ++frequencies[counters[i]];

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
    // (1) Number of key/code pairs (int)
    // (2) For each key/code pair:
    //         length(key) (char)
    //         key (sequence of bits)
    //         length(code) (char)
    //         code (sequence of bits)
    // (3) Number N of keys to code (ind_t)
    // (4) N codes (sequence of bits)
    /**************************************************************************/

    zlib_open_wbit();

    // Number of key/code pairs
    unsigned int dict_size = codes.size();
    zlib_write_bit(dict_size, sizeof(int)*8);

    // Key/code pairs
    for (HuffCodeMap::const_iterator it = codes.begin(); it != codes.end(); ++it) {

        unsigned long int key = it->first; // TODO change key to ind_t

        // First, the key's length
        unsigned char key_len = 0;
        unsigned long int key_copy = key;
        while (key_copy != 0) {
            key_copy >>= 1;
            key_len++;
        }
        if (key_len == 0) // A 0 still requires 1 bit
            key_len = 1;
        zlib_write_bit(key_len, 6);

        // Next, the key itself
        zlib_write_bit(key, key_len);

        // Now, the code's length
        zlib_write_bit(code_lens[key], 6);

        // Finally, the code itself
        zlib_write_bit(it->second, code_lens[key]);
    }

    // Number N of symbols to code
    unsigned long int n_symbols = counters.size();
    zlib_write_bit(n_symbols, sizeof(n_symbols)*8);

    // Now the N codes
    for (unsigned long int i = 0; i < counters.size(); ++i)
        zlib_write_bit(codes[counters[i]], code_lens[counters[i]]);

    zlib_close_wbit();
}

#endif // ENCODE_HPP
