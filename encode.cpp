// This code is partly from of https://rosettacode.org/wiki/Huffman_coding (released under the GNU Free Documentation License 1.2, http://www.gnu.org/licenses/fdl-1.2.html), which is a Huffman encoding implementation for C++. This version does a run-length encoding first, in order to efficiently compress a bit-stream.

#include <iostream>
#include <fstream>
#include <queue>
#include <map>
#include <iterator>
#include <algorithm>

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

class InternalNode:
public INode {
  public:
    INode * const
     left;
    INode *const
     right;

  InternalNode(INode * c0, INode * c1):
    INode(c0->f + c1->f), left(c0), right(c1) {
    } ~InternalNode() {
	delete left;
	delete right;
    }
};

class LeafNode:
public INode {
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
    std::priority_queue < INode *, std::vector < INode * >,
	NodeCmp > trees;

    for (std::map < int, int >::iterator it = frequencies.begin();
	 it != frequencies.end(); ++it) {
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

void
GenerateCodes(const INode * node, const HuffCode & prefix,
	      HuffCodeMap & outCodes)
{
    if (const LeafNode * lf = dynamic_cast < const LeafNode * >(node)) {
	outCodes[lf->c] = prefix;
    } else if (const InternalNode * in =
	       dynamic_cast < const InternalNode * >(node)) {
	HuffCode leftPrefix = prefix;
	leftPrefix.push_back(false);
	GenerateCodes(in->left, leftPrefix, outCodes);

	HuffCode rightPrefix = prefix;
	rightPrefix.push_back(true);
	GenerateCodes(in->right, rightPrefix, outCodes);
    }
}

void encode()
{

    /*********************************************/
    // Perform run-length encoding into counters[]
    /*********************************************/

    ifstream input("tmp/mask.raw", ios::binary);
    std::vector < char >
	contents((istreambuf_iterator < char >(input)),
		 istreambuf_iterator < char >());
    char last_bit = 1;
    int counter = -1;
    vector<int> counters;
    std::map<int, int> frequencies;
    for (int i = 0; i < contents.size(); ++i) {
        char c = contents[i];
        for (int j = 7; j >= 0; --j) {
            char bit = (c >> j)&1;
            if (bit == last_bit)
                counter++;
            else {
                counters.push_back(counter + 1);
                ++frequencies[counter+1];
                counter = 0;
                last_bit = bit;
            }
        }
    }
    if (counter > 0) {
        counters.push_back(counter + 1);
        ++frequencies[counter+1];
    }

    /*******************************/
    // Create the Huffman dictionary
    /*******************************/

    INode *root = BuildTree(frequencies);

    HuffCodeMap codes;
    GenerateCodes(root, HuffCode(), codes);
    delete root;

    if (frequencies.size() == 1)	// Fix: if there's only one symbol, we still need one bit for it
        codes[frequencies.begin()->first].push_back(false);

    /***************************************************************************/
    // Translate each symbol. Save the code using 24 bits and its length using 8
    /***************************************************************************/

    int n_bits = 0;
    unsigned int key_array[codes.size()];
    unsigned int code_array[codes.size()];
    counter = 0;
    for (HuffCodeMap::const_iterator it = codes.begin(); it != codes.end(); ++it) {
        int key = (int) (it->first);
        int cost = it->second.end() - it->second.begin();
        key_array[counter] = key;
        unsigned int encoding = 0;
        for (int i = 0; i < it->second.size(); ++i)
            encoding |= it->second[i] * (1 << (it->second.size() - 1 - i));
        if (encoding >= 1 << 24) {
            cout << "Encoding too large" << endl;
            exit(1);
        }
        encoding |= (it->second.size() << 24);
        code_array[counter] = encoding;
        n_bits += cost * frequencies[key];
        ++counter;
    }

    /**********************************************/
    // Save the dictionary and the translated codes
    /**********************************************/

    ofstream output("tmp/mask.compressed", ios::out | ios::binary);
    int dict_size = codes.size();
    output.write(reinterpret_cast < char *>(&dict_size), sizeof(int));
    output.write(reinterpret_cast < char *>(key_array),
		 sizeof(int) * codes.size());
    output.write(reinterpret_cast < char *>(code_array),
		 sizeof(int) * codes.size());
    output.write(reinterpret_cast < char *>(&n_bits), sizeof(int));
    vector < char > translation;
    char translation_wbyte = 0;
    char translation_wbit = 7;
    for (int i = 0; i < counters.size(); ++i) {
        for (int j = 0; j < codes[counters[i]].size(); ++j) {
            translation_wbyte |= codes[counters[i]][j] << translation_wbit;
            translation_wbit--;
            if (translation_wbit < 0) {
            translation.push_back(translation_wbyte);
            translation_wbit = 7;
            translation_wbyte = 0;
            }
        }
    }
    if (translation_wbit < 7) {
	translation.push_back(translation_wbyte);
    }
    std::copy(translation.begin(), translation.end(),
	      std::ostreambuf_iterator < char >(output));
    output.close();

}
