#ifndef __ENCODE_HPP__
#define __ENCODE_HPP__

// Parts of this code were taken from https://rosettacode.org/wiki/Huffman_coding
// (released under the GNU Free Documentation License 1.2, http://www.gnu.org/licenses/fdl-1.2.html)
// which is a C++ Huffman encoding implementation. Our version does a run-length encoding first
// in order to compress a sequence of presence bits in the Tucker core.

#include <iostream>
#include <fstream>
#include <queue>
#include <map>
#include <iterator>
#include <algorithm>
#include <cstring>

using namespace std;

typedef
std::vector < bool > HuffCode;
typedef
std::map < int, HuffCode > HuffCodeMap;

class INode {
public:
    const int
    f;

    virtual ~ INode() {
    } protected:
    INode(int f):f(f) {
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
    const int
    c;

    LeafNode(int f, int c):INode(f), c(c) {
    }};

struct NodeCmp {
    bool operator  () (const INode * lhs, const INode * rhs) const {
        return lhs->f > rhs->f;
    }};

INode *BuildTree(std::map < int, int >&frequencies)
{
    std::priority_queue < INode *, std::vector < INode * >, NodeCmp > trees;

    for (std::map < int, int >::iterator it = frequencies.begin(); it != frequencies.end(); ++it) {
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

void GenerateCodes(const INode * node, const HuffCode & prefix, HuffCodeMap & outCodes)
{
    if (const LeafNode * lf = dynamic_cast < const LeafNode * >(node)) {
        outCodes[lf->c] = prefix;
    } else if (const InternalNode * in = dynamic_cast < const InternalNode * >(node)) {
        HuffCode leftPrefix = prefix;
        leftPrefix.push_back(false);
        GenerateCodes(in->left, leftPrefix, outCodes);

        HuffCode rightPrefix = prefix;
        rightPrefix.push_back(true);
        GenerateCodes(in->right, rightPrefix, outCodes);
    }
}

void encode(vector < char >&mask, vector < char >&compressed_mask) {

    /*********************************************/
    // Perform run-length encoding into counters[]
    /*********************************************/

    char last_bit = 0;
    int counter = 0;
    vector < int >counters;
    std::map < int, int >frequencies;
    for (unsigned int i = 0; i < mask.size(); ++i) {
        char c = mask[i];
        for (int j = 7; j >= 0; --j) {
            char bit = (c >> j) & 1;
            if (bit == last_bit)
                counter++;
            else {
                counters.push_back(counter);
                ++frequencies[counter];
                counter = 1;
                last_bit = bit;
            }
        }
    }
    counters.push_back(counter);
    ++frequencies[counter];

    /*******************************/
    // Create the Huffman dictionary
    /*******************************/

    INode *root = BuildTree(frequencies);

    HuffCodeMap codes;
    GenerateCodes(root, HuffCode(), codes);
    delete root;

    if (frequencies.size() == 1) // If there's only one symbol, we still need one bit for it
        codes[frequencies.begin()->first].push_back(false);

    /***************************************************************************/
    // Translate each symbol. Save the code using 27 bits and its length using 5
    /***************************************************************************/

    ind_t n_bits = 0;
    unsigned int key_array[codes.size()];
    unsigned int code_array[codes.size()];
    counter = 0;
    for (HuffCodeMap::const_iterator it = codes.begin(); it != codes.end(); ++it) {
        int key = (int) (it->first);
        int cost = it->second.size();
        key_array[counter] = key;
        unsigned int encoding = 0;
        if (it->second.size() > 27) {
            cout << "Encoding too large (" << it->second.size() << " bits)" << endl;
            exit(1);
        }
        for (unsigned int i = 0; i < it->second.size(); ++i)
            encoding |= it->second[i] << (it->second.size()-1-i);
        encoding |= (it->second.size() << 27);
        code_array[counter] = encoding;
        n_bits += cost * frequencies[key];
        ++counter;
    }

    /**********************************************/
    // Save the dictionary and the translated codes
    /**********************************************/

    int dict_size = codes.size();
    compressed_mask = vector < char >((1 + 2 * dict_size) * sizeof(int) + sizeof(n_bits));
    memcpy(&compressed_mask[0], reinterpret_cast < char *>(&dict_size), sizeof(int));
    memcpy(&compressed_mask[0] + (1) * sizeof(int), reinterpret_cast < char *>(key_array), dict_size * sizeof(int));
    memcpy(&compressed_mask[0] + (1 + dict_size) * sizeof(int), reinterpret_cast < char *>(code_array), dict_size * sizeof(int));
    memcpy(&compressed_mask[0] + (1 + 2 * dict_size) * sizeof(int), reinterpret_cast < char *>(&n_bits), sizeof(n_bits));

    char compressed_mask_wbyte = 0;
    char compressed_mask_wbit = 7;
    for (unsigned int i = 0; i < counters.size(); ++i) {
        for (unsigned int j = 0; j < codes[counters[i]].size(); ++j) {
            compressed_mask_wbyte |= codes[counters[i]][j] << compressed_mask_wbit;
            compressed_mask_wbit--;
            if (compressed_mask_wbit < 0) {
                compressed_mask.push_back(compressed_mask_wbyte);
                compressed_mask_wbit = 7;
                compressed_mask_wbyte = 0;
            }
        }
    }
    if (compressed_mask_wbit < 7)
        compressed_mask.push_back(compressed_mask_wbyte);
}

#endif // ENCODE_HPP
