#include <iostream>
#include <fstream>
#include <queue>
#include <map>
#include <climits>
#include <iterator>
#include <algorithm>

using namespace std;

void decode(string input_file, string output_file) {

    /*********************************************/
    // Read dictionary, encoding size and encoding
    /*********************************************/

    ifstream input(input_file.c_str(), ios::binary );
    std::vector<char> buffer((istreambuf_iterator<char>(input)), istreambuf_iterator<char>());
	int dict_size = reinterpret_cast<int*>( &buffer[0] )[0];
	int *key_array = reinterpret_cast<int*>( &buffer[0] )+1;
	int *code_array = reinterpret_cast<int*>( &buffer[0] )+1+dict_size;
    int n_bits = reinterpret_cast<int*>( &buffer[0] )[1+2*dict_size];
	char *compression = &buffer[(1+2*dict_size+1)*sizeof(int)];
	int compression_size = buffer.size()-(1+2*dict_size+1)*sizeof(int);

	int* table[24];
	for (int i = 0; i < 24; ++i) {
		table[i] = new int[1<<i];
		for (int j = 0; j < 1<<i; ++j)
			table[i][j] = -1;
	}
	for (int i = 0; i < dict_size; ++i) {
		int code_len = code_array[i] >> 24;
		int encoding = code_array[i] & 0x00ffffff;
		table[code_len][encoding] = key_array[i];
	}

    /*****************************************************/
    // Read translation and decode each symbol on the spot
    /*****************************************************/

    int read_bits = 0;
	std::vector<char> result;
	int counter = 0;    
	int this_code = 0;
	unsigned char current_bit = 1;
	unsigned char write_byte = 0;
    int write_counter = 7;
	int this_code_len = 0;
    for (int i = 0; i < compression_size; ++i) {
		char c = compression[i];
		for (int j = 0; j < 8; ++j) {
            if (read_bits == n_bits)
				break;
            char bit = ((c & (128 >> j)) != 0);
			this_code += bit;
			this_code_len++;
			counter++;
			read_bits++;

			if (table[this_code_len][this_code] != -1) {
                for (int k = 0; k < table[this_code_len][this_code]; ++k) {
                    write_byte |= current_bit<<write_counter;
                    write_counter--;
                    if (write_counter < 0) {
                        result.push_back((write_byte));
                        write_byte = 0;
                        write_counter = 7;
                    }
                }
				current_bit = !current_bit;
				this_code = 0;
				this_code_len = 0;
			}
			this_code <<= 1;
		}
	}
    if (write_counter < 7) {
		result.push_back(write_byte);
	}

    ofstream output(output_file.c_str(), ios::out | ios::binary);
	std::copy(result.begin(), result.end(), std::ostreambuf_iterator<char>(output));
    output.close();

    for (int i = 0; i < 24; ++i)
        delete table[i];

}
